import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional, Dict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import rff.layers
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import WandbLogger
from skimage.metrics import structural_similarity as ssim
from torch import nn

from config_utils import (
    ExperimentConfig,
    MLPConfig,
    SirenConfig,
    ReLUConfig,
    WireConfig,
    FinerConfig,
)
from data_utils import resize_to_square, tensor_to_image
from datasets import Signal
from metrics import psnr, get_wasserstein, prepare_spectrum_generators
from spectum_utils import get_fft, get_spectrum


class SineLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega: list[float] | float = 30.0,
        beta=1.0,
        weight_constant=6.0,
        is_linear=False,
        apply_omega_when_linear=False,
        boost_bias=False,
    ):
        super().__init__()
        if is_linear and not apply_omega_when_linear:
            # Don't apply gradient boosting on the last layer
            # (not used; we follow the original SIREN implementation)
            omega = 1.0
        if isinstance(omega, list) and in_features != len(omega):
            raise ValueError("Omega must be provided for each input feature.")
        if isinstance(omega, list) and not is_first:
            raise ValueError("Omega can only be a list for the first layer.")

        self.omega = omega
        self.is_first = is_first
        self.boost_bias = boost_bias
        self.beta = beta
        self.weight_constant = weight_constant
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation = lambda x: x
        if not is_linear:
            self.activation = torch.sin

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                num_input = self.linear.weight.size(-1)
                w = torch.distributions.beta.Beta(self.beta, self.beta).rsample(
                    self.linear.weight.shape
                )
                w = w * 2 - 1
                w = w / num_input
                self.linear.weight.copy_(w)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(self.weight_constant / self.in_features) / self.omega,
                    np.sqrt(self.weight_constant / self.in_features) / self.omega,
                )

    def forward(self, x):
        if not self.boost_bias:
            if isinstance(self.omega, list):
                x = torch.clone(x)
                for i, omega in self.omega:
                    x[:, i] = omega * x[:, i]
            else:
                x = self.omega * x
            return self.activation(self.linear(x))
        else:
            return self.activation(self.omega * self.linear(x))


class BaseMLP(ABC, nn.Module):
    @property
    @abstractmethod
    def net(self) -> nn.Sequential:
        pass

    @net.setter
    def net(self, net):
        self.net = net

    @property
    @abstractmethod
    def spectral_parameters(self) -> tuple:
        pass

    @spectral_parameters.setter
    def spectral_parameters(self, spectral_parameters):
        self.spectral_parameters = spectral_parameters

    def forward(self, coords):
        return self.net(coords)

    def forward_in_batches(self, coords, batch_size=2**19) -> torch.Tensor:
        return inference_in_batches(
            self.net,
            coords,
            batch_size=batch_size,
            device=self.net.parameters().__next__().device,
        )


class Siren(BaseMLP):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=False,
        first_omega_0=30.0,
        hidden_omega=30.0,
        beta=1.0,
        weight_constant=6.0,
        apply_omega_when_linear=False,
        boost_bias=False,
    ):
        super().__init__()

        self._spectral_parameters = (first_omega_0, beta, weight_constant)
        self._net = []
        self._net.append(
            SineLayer(
                in_features,
                hidden_features,
                beta=beta,
                is_first=True,
                omega=first_omega_0,
            )
        )

        for i in range(hidden_layers):
            self._net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega=hidden_omega,
                    weight_constant=weight_constant,
                    boost_bias=boost_bias,
                )
            )
        self._net.append(
            SineLayer(
                hidden_features,
                out_features,
                is_first=False,
                omega=hidden_omega,
                is_linear=outermost_linear,
                apply_omega_when_linear=apply_omega_when_linear,
                boost_bias=boost_bias,
            )
        )

        self._net = nn.Sequential(*self._net)

    @property
    def net(self) -> nn.Sequential:
        return self._net

    @property
    def spectral_parameters(self) -> tuple:
        return self._spectral_parameters


## WIRE Ref：https://github.com/vishwa91/wire
class ComplexGaborLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega0=10.0,
        sigma0=40.0,
        trainable=False,
    ):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        self.in_features = in_features

        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat

        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)

        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        return torch.exp(1j * omega - scale.abs().square())


class Wire(BaseMLP):
    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        hidden_layers,
        first_omega_0=20.0,
        hidden_omega_0=20.0,
        scale=10.0,
    ):
        # hidden_omega_0 = first_omega_0
        super().__init__()
        self._spectral_parameters = (first_omega_0, scale)
        # All results in the paper were with the default complex 'gabor' nonlinearity
        self.nonlin = ComplexGaborLayer

        # Since complex numbers are two real numbers, reduce the number of hidden parameters by 2
        hidden_features = int(hidden_features / np.sqrt(2))
        dtype = torch.cfloat
        self.complex = True

        self._net = []
        self._net.append(
            self.nonlin(
                in_features,
                hidden_features,
                is_first=True,
                omega0=first_omega_0,
                sigma0=scale,
            )
        )

        for i in range(hidden_layers):
            self._net.append(
                self.nonlin(
                    hidden_features, hidden_features, omega0=hidden_omega_0, sigma0=10
                )
            )

        final_linear = nn.Linear(hidden_features, out_features, dtype=dtype)
        self._net.append(final_linear)

        class RealLayer(nn.Module):
            def forward(self, x):
                return x.real / 2 + 0.5

        self._net.append(RealLayer())
        self._net = nn.Sequential(*self._net)

    @property
    def net(self) -> nn.Sequential:
        return self._net

    @property
    def spectral_parameters(self) -> tuple:
        return self._spectral_parameters


class FinerLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega_0=30.0,
        first_bias_scale=None,
        scale_req_grad=False,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
        self.scale_req_grad = scale_req_grad
        self.first_bias_scale = first_bias_scale
        if self.first_bias_scale != None:
            self.init_first_bias()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def init_first_bias(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.bias.uniform_(-self.first_bias_scale, self.first_bias_scale)

    def generate_scale(self, x):
        if self.scale_req_grad:
            scale = torch.abs(x) + 1
        else:
            with torch.no_grad():
                scale = torch.abs(x) + 1
        return scale

    def forward(self, input):
        x = self.linear(input)
        scale = self.generate_scale(x)
        out = torch.sin(self.omega_0 * scale * x)
        return out


class Finer(BaseMLP):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        first_omega_0=30,
        hidden_omega_0=30.0,
        bias=True,
        first_bias_scale=None,
        scale_req_grad=False,
    ):
        self._spectral_parameters = (first_omega_0, first_bias_scale)
        super().__init__()
        self._net = []
        self._net.append(
            FinerLayer(
                in_features,
                hidden_features,
                is_first=True,
                omega_0=first_omega_0,
                first_bias_scale=first_bias_scale,
                scale_req_grad=scale_req_grad,
            )
        )

        for i in range(hidden_layers):
            self._net.append(
                FinerLayer(
                    hidden_features,
                    hidden_features,
                    omega_0=hidden_omega_0,
                    scale_req_grad=scale_req_grad,
                )
            )

        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(
                -np.sqrt(6 / hidden_features) / hidden_omega_0,
                np.sqrt(6 / hidden_features) / hidden_omega_0,
            )
        self._net.append(final_linear)
        self._net = nn.Sequential(*self._net)

    @property
    def net(self) -> nn.Sequential:
        return self._net

    @property
    def spectral_parameters(self) -> tuple:
        return self._spectral_parameters


class FourierFeaturesMLP(BaseMLP):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        sigma=1.0,
    ):
        super().__init__()

        self._spectral_parameters = (sigma,)
        self._net = []
        self._net.append(
            rff.layers.GaussianEncoding(
                sigma=sigma, input_size=in_features, encoded_size=hidden_features // 2
            )
        )

        for i in range(hidden_layers):
            self._net.append(torch.nn.Linear(hidden_features, hidden_features))
            self._net.append(torch.nn.ReLU())
        self._net.append(torch.nn.Linear(hidden_features, out_features))

        self._net = nn.Sequential(*self._net)

    @property
    def net(self) -> nn.Sequential:
        return self._net

    @property
    def spectral_parameters(self) -> tuple:
        return self._spectral_parameters


class INRBase(pl.LightningModule):
    def __init__(
        self,
        dataset,
        experiment_config: ExperimentConfig,
        gt_example_image: Optional[torch.Tensor] = None,
        spectrum_generators: Optional[dict] = None,
        initial_wasserstein: Optional[Dict[str, float]] = None,
        initial_wasserstein_se: Optional[Dict[str, float]] = None,
        initial_wasserstein_time: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.experiment_config = experiment_config
        self.dataset = dataset
        if spectrum_generators is None:
            spectrum_generators = prepare_spectrum_generators(
                experiment_config.spectrum_config
            )

        self.net = init_net_from_config(experiment_config.model_config, dataset)

        self.spectrum_generators = spectrum_generators
        if initial_wasserstein is None:
            initial_wasserstein = dict()
            initial_wasserstein_se = dict()
            initial_wasserstein_time = dict()
        elif None == initial_wasserstein_se or None == initial_wasserstein_time:
            raise ValueError(
                "Either all or none of initial_wasserstein, initial_wasserstein_se, initial_wasserstein_time should be provided."
            )
        self.initial_wasserstein = initial_wasserstein
        self.initial_wasserstein_se = initial_wasserstein_se
        self.initial_wasserstein_time = initial_wasserstein_time
        self.gt_spectra_normalised = None
        self.gt_example_image = gt_example_image
        if gt_example_image is not None:
            self._calculate_gt_spectrum(gt_example_image)

    def _calculate_gt_spectrum(self, gt_image: torch.Tensor):
        gt_spectra_normalised = {
            key: fun(gt_image).cpu().numpy()
            for key, fun in self.spectrum_generators.items()
        }
        self.gt_spectra_normalised = {
            key: gt_spectra_normalised[key] / np.sum(gt_spectra_normalised[key])
            for key in gt_spectra_normalised
        }

    @property
    def wandb_logger(self) -> WandbLogger:
        return self.logger

    def calculate_initial_wasserstein_distance(
        self,
        spectrum_key,
    ):
        raise NotImplementedError()

    def eval_initial_wasserstein(self) -> Dict[str, float]:
        total_time = 0
        metrics = dict()
        for name, spectrum_fun in self.spectrum_generators.items():
            print(f"Evaluating spectrum using {name}...")
            if name in self.initial_wasserstein:
                distance = self.initial_wasserstein[name]
                standard_error = self.initial_wasserstein_se[name]
                w_time = self.initial_wasserstein_time[name]
            else:
                distance, standard_error, w_time = (
                    self.calculate_initial_wasserstein_distance(name)
                )
            metrics.update(
                {
                    f"initial_wasserstein_{name}": distance,
                    f"initial_wasserstein_{name}_SE": standard_error,
                    f"initial_wasserstein_time_{name}": w_time,
                }
            )
            total_time += w_time
        metrics["initial_wasserstein_time_total"] = total_time
        self.initial_wasserstein.update(metrics)
        self.wandb_logger.log_metrics(metrics, step=self.global_step)

        return metrics

    def _log_summary_table(self, metrics: dict):
        table_keys = sorted(metrics.keys())
        table_data = [metrics[key] for key in metrics]
        self.wandb_logger.log_metrics(metrics, step=self.global_step)

        self.wandb_logger.log_table(
            key="summary_table",
            columns=table_keys,
            data=[table_data],
            step=self.global_step,
        )

    def validation_step(self, _):
        # everything is done in on_validation_epoch_end
        pass

    def test_step(self, _):
        # everything is done in on_test_epoch_end
        pass


class INR(INRBase):
    def __init__(
        self,
        dataset: Signal,
        spectrum_generators: Optional[dict],
        experiment_config: ExperimentConfig,
        initial_wasserstein: Optional[Dict[str, float]] = None,
        initial_wasserstein_se: Optional[Dict[str, float]] = None,
        initial_wasserstein_time: Optional[Dict[str, float]] = None,
    ):
        super().__init__(
            dataset=dataset,
            experiment_config=experiment_config,
            spectrum_generators=spectrum_generators,
            gt_example_image=dataset.signal_channel_first,
            initial_wasserstein=initial_wasserstein,
            initial_wasserstein_se=initial_wasserstein_se,
            initial_wasserstein_time=initial_wasserstein_time,
        )
        self.batch_size = self.experiment_config.batch_size

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.squeeze(0)
        y = y.squeeze(0)
        y_pred = self.net(x)
        loss = F.mse_loss(y_pred, y)
        self.wandb_logger.log_metrics({"train_loss": loss}, step=self.global_step)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.experiment_config.optimizer.learning_rate
        )
        if self.experiment_config.optimizer.scheduler_name is None:
            return optimizer
        assert self.experiment_config.optimizer.scheduler_name == "CosineAnnealingLR"
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, **self.experiment_config.optimizer.scheduler_args
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_validation_epoch_end(self):
        self._validate_model(conserve_memory=True)

    def on_test_epoch_end(self):
        self._validate_model(conserve_memory=False)

    def log_wasserstein(self):
        fig, axs = plt.subplots(4, 4, figsize=(16, 16))
        axs = axs.flatten()
        results = {}
        metrics = {}
        _, out, out_train, out_test = self._infer_full_img()
        for ax, (name, spectrum_fun) in zip(axs, self.spectrum_generators.items()):
            gt_spec = self.gt_spectra_normalised[name]
            out_spec = spectrum_fun(out).numpy()
            results[name + "_gt"] = gt_spec
            results[name] = out_spec
            ax.plot(gt_spec, label="gt")
            ax.plot(out_spec / sum(out_spec), label="out")
            ax.set_title(name)
            w = get_wasserstein(gt_spec, out_spec)
            metrics[f"wasserstein_{name}"] = w
        axs[0].legend()
        fig.suptitle(self.experiment_config.model_config.config_name)
        if self.global_step == 0:
            self.wandb_logger.log_image(
                key="spectra", images=[fig], step=self.global_step
            )

            # Save weights
            # weights_dir = Path(self.experiment_config.results_path) / "embedding_weights"
            # weights_dir.mkdir(exist_ok=True, parents=True)
            # if isinstance(self.net, Siren):
            #     embedding: SineLayer = self.net._net[0]
            #     weights = embedding.linear.weight.cpu().detach().data
            #     np.save(weights_dir / "weight_0.npy", weights)
            # elif isinstance(self.net, FourierFeaturesMLP):
            #     embedding = self.net._net[0]
            #     weights = embedding.b.cpu().detach().data
            #     np.save(weights_dir / "weight_0.npy", weights)

        spectra_dir = Path(self.experiment_config.results_path) / "spectra_img"
        spectra_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(spectra_dir / f"spectra_{self.global_step}.png")
        plt.close(fig)

        for name, spectrum in results.items():
            df = pd.DataFrame.from_dict({"spectrum": spectrum}, orient="columns")
            spectra_dir_csv = Path(self.experiment_config.results_path) / "spectra_csv"
            spectra_dir_csv.mkdir(exist_ok=True, parents=True)
            df.to_csv(spectra_dir_csv / f"spectrum_{name}_{self.global_step}.csv")

        return metrics

    def _validate_model(self, conserve_memory=True):
        metrics = dict()
        if self.global_step == 0:
            if self.experiment_config.spectrum_config.eval_before_training:
                metrics = self.eval_initial_wasserstein()
            else:
                for name in self.initial_wasserstein:
                    metrics.update(
                        {
                            f"initial_wasserstein_{name}": self.initial_wasserstein[
                                name
                            ],
                            f"initial_wasserstein_{name}_SE": self.initial_wasserstein_se[
                                name
                            ],
                            f"initial_wasserstein_time_{name}": self.initial_wasserstein_time[
                                name
                            ],
                        }
                    )

            self._log_gt()

        w_metrics = self.log_wasserstein()
        metrics.update(w_metrics)

        _, out, out_train, out_test = self._infer_full_img()
        residuals = out - self.dataset.signal_channel_first
        train_residuals = out_train - self.dataset.train_pixels
        test_residuals = out_test - self.dataset.test_pixels

        metrics.update(
            self._get_metrics(self.dataset.signal_channel_first, out, residuals)
        )
        metrics["ssim"] = ssim(
            self.dataset.signal_channel_first.numpy(),
            out.numpy(),
            channel_axis=0,
            data_range=2,
        )
        metrics.update(
            self._get_metrics(
                self.dataset.train_pixels, out_train, train_residuals, suffix="train"
            )
        )
        metrics.update(
            self._get_metrics(
                self.dataset.test_pixels, out_test, test_residuals, suffix="test"
            )
        )

        self._log_residual_figure(out, residuals)

        self._log_summary_table(metrics)
        self.wandb_logger.log_image(
            key="output",
            images=[tensor_to_image(out, limit_size=conserve_memory)],
            step=self.global_step,
            caption=["model_output"],
        )
        if self.global_step == 0:
            scaled_output = out - out.min()
            scaled_output = scaled_output / scaled_output.max()
            self.wandb_logger.log_image(
                key="scaled_output",
                images=[
                    tensor_to_image(scaled_output, limit_size=False, rescale=False)
                ],
                step=self.global_step,
                caption=["scaled_model_output"],
            )
        outputs_dir = Path(self.experiment_config.results_path) / "outputs"
        outputs_dir.mkdir(exist_ok=True, parents=True)
        img = tensor_to_image(out, limit_size=False)
        img.save(outputs_dir / f"output_{self.global_step}.png")

        # Save weights
        # weights_dir = Path(self.experiment_config.results_path) / "embedding_weights"
        # weights_dir.mkdir(exist_ok=True, parents=True)
        # if isinstance(self.net, Siren):
        #     embedding: SineLayer = self.net._net[0]
        #     weights = embedding.linear.weight.cpu().detach().data
        #     np.save(weights_dir / "weight_last.npy", weights)
        # elif isinstance(self.net, FourierFeaturesMLP):
        #     embedding = self.net._net[0]
        #     weights = embedding.b.cpu().detach().data
        #     np.save(weights_dir / "weight_last.npy", weights)

    def _log_gt(self):
        self.wandb_logger.log_image(
            key="ground truth",
            images=[
                tensor_to_image(
                    self.dataset.signal_channel_first,
                )
            ],
            step=self.global_step,
        )

    def calculate_initial_wasserstein_distance(
        self,
        spectrum_key,
    ):
        with torch.no_grad():
            return calculate_initial_wasserstein_distance(
                self.experiment_config,
                self.dataset,
                self.spectrum_generators[spectrum_key],
                evals=self.experiment_config.spectrum_config.initial_evals,
                gt_spectrum=self.gt_spectra_normalised[spectrum_key],
            )

    def _infer_full_img(self):
        out = self.net.forward_in_batches(
            self.dataset.coords, batch_size=self.batch_size
        )
        out_train = out[self.dataset.train_idx]
        out_test = out[self.dataset.test_idx]
        img = out.reshape(self.dataset.shape)
        img = torch.movedim(img, -1, 0)  # move channels to the beginning
        return out, img, out_train, out_test

    @staticmethod
    def _get_metrics(gt, out, residuals, suffix=""):
        if suffix:
            suffix = f"_{suffix}"
        return {
            f"psnr{suffix}": psnr(gt.numpy(), out.numpy()),
            f"MAE{suffix}": torch.abs(residuals).mean(),
            f"max_residual{suffix}": torch.abs(residuals).max(),
        }

    def _log_residual_figure(self, out, residuals, key="residuals"):
        if len(residuals.shape) == 4:
            # extract first frame from video
            residuals = residuals[:, 0]
            out = out[:, 0]

        residuals_fft = np.abs(get_fft(resize_to_square(residuals)))
        residuals_fft_spectrum = get_spectrum(residuals_fft, drop_bias=False)
        residual_sum = np.abs(residuals).sum(0)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(str(self.net.spectral_parameters))
        axes[0].imshow(out.permute(1, 2, 0).clip(-1, 1) / 2 + 0.5)
        pos = axes[1].imshow(residual_sum, cmap="inferno", vmin=0)
        fig.colorbar(pos, ax=axes[1])
        pos = axes[2].imshow(np.log(1 + residuals_fft), cmap="inferno")
        fig.colorbar(pos, ax=axes[2])
        axes[3].plot(residuals_fft_spectrum)
        axes[3].set_title("spectrum of residuals")

        residual_dir = Path(self.experiment_config.results_path) / "residuals"
        residual_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(residual_dir / f"residuals_plot_{self.global_step}.png")
        plt.close(fig)

        residuals_csv = residual_dir / f"residuals_{self.global_step}.csv"
        pd.DataFrame({"residuals": residuals_fft_spectrum[1:]}).to_csv(
            residuals_csv, index=False
        )

        max_residual = residual_sum.max()
        img = tensor_to_image(residual_sum[None] / max_residual, rescale=False)
        img.save(residual_dir / f"residuals_{self.global_step}_{max_residual}.png")


def inference_in_batches(model, x, batch_size=2**19, device="cuda"):
    """Run full inference and move results to cpu."""
    with torch.no_grad():
        out = []
        for i in range(0, x.shape[0], batch_size):
            out.append(model(x[i : i + batch_size].to(device=device)).to(device="cpu"))
        return torch.cat(out, dim=0)


def init_net_from_config(model_config: MLPConfig, dataset):
    if isinstance(model_config, SirenConfig):
        model = Siren(
            in_features=dataset.dims,
            hidden_features=model_config.hidden_features,
            hidden_layers=model_config.hidden_layers,
            out_features=dataset.channels,
            first_omega_0=model_config.omega_0,
            beta=model_config.beta,
            weight_constant=model_config.weight_constant,
            outermost_linear=True,
            apply_omega_when_linear=model_config.apply_omega_when_linear,
            boost_bias=model_config.boost_bias,
        )
        return model
    elif isinstance(model_config, ReLUConfig):
        return FourierFeaturesMLP(
            in_features=2,
            hidden_features=model_config.hidden_features,
            hidden_layers=model_config.hidden_layers,
            out_features=dataset.channels,
            sigma=model_config.sigma,
        )
    elif isinstance(model_config, WireConfig):
        return Wire(
            in_features=2,
            out_features=dataset.channels,
            hidden_features=model_config.hidden_features,
            hidden_layers=model_config.hidden_layers,
            first_omega_0=model_config.omega_0,
            scale=model_config.scale,
            hidden_omega_0=model_config.hidden_omega,
        )
    elif isinstance(model_config, FinerConfig):
        return Finer(
            in_features=2,
            out_features=dataset.channels,
            hidden_features=model_config.hidden_features,
            hidden_layers=model_config.hidden_layers,
            first_omega_0=model_config.omega_0,
            first_bias_scale=model_config.first_bias_scale,
            hidden_omega_0=model_config.hidden_omega,
        )
    else:
        raise ValueError(f"Unknown model type: {model_config}")


def calculate_initial_wasserstein_distance(
    expeperiment_config: ExperimentConfig,
    dataset,
    spectrum_fun: Callable,
    evals: int,
    gt_spectrum: Optional[torch.Tensor],
):
    if gt_spectrum is None:
        gt_spectrum = spectrum_fun(dataset.signal_channel_first)

    spectrum_generators = prepare_spectrum_generators(
        expeperiment_config.spectrum_config
    )
    t0 = time.time()
    distances = []
    for i in range(evals):
        model = INR(
            dataset=dataset,
            spectrum_generators=spectrum_generators,
            experiment_config=expeperiment_config,
        ).to(device=expeperiment_config.device)

        _, out, _, _ = model._infer_full_img()
        out_spec = spectrum_fun(out).cpu().numpy()
        del out
        w = get_wasserstein(gt_spectrum, out_spec)
        distances.append(w)

    total_time = time.time() - t0
    standard_error = np.std(distances) / np.sqrt(evals)
    return np.mean(distances), standard_error, total_time
