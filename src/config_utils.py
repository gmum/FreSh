import dataclasses
import hashlib
import itertools
import os
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, List

from omegaconf import OmegaConf, ListConfig, DictConfig
from sortedcontainers import SortedDict


class TrainingType(Enum):
    ONE = "one"  # train one (baseline) configuration
    ALL = "all"  # train all configurations from ExperimentConfig.possible_configs
    BEST = "best"  # train one configuration from ExperimentConfig.possible_configs selected based on wasserstein distance
    SAVE_OUT = "save"  # saves output of uninitialised model

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value

        raise NotImplementedError(
            f"Cannot compare {self.__class__} with {other.__class__}"
        )

    def __str__(self):
        return self.value


class ComparableObject:

    def __iter__(self):
        return (getattr(self, field.name) for field in dataclasses.fields(self))

    def __lt__(self, other):
        for a, b in zip(self, other):
            # we treat None as the smallest possible value
            if a is None and b is not None:
                return True
            if a is not None and b is None:
                return False

            if isinstance(a, SortedDict) and isinstance(b, SortedDict):
                a = list(a.items())
                b = list(b.items())

            if a == b:
                continue
            return a < b
        return False


@dataclass(frozen=True, order=True)
class MLPConfig:
    hidden_features: int
    hidden_layers: int

    @property
    def config_name(self):
        return f"{self.hidden_layers}_{self.hidden_features}"

    def dict(self):
        return asdict(self)


class InputEncoding(Enum):
    Positional = 0
    RandomFourierFeatures = 1

    @staticmethod
    def from_string(s: str):
        if s == "rff":
            return InputEncoding.RandomFourierFeatures
        elif s == "pos":
            return InputEncoding.Positional
        else:
            raise ValueError(f"Unknown input encoding: {s}")

    def __str__(self):
        if self == InputEncoding.Positional:
            return "pos"
        elif self == InputEncoding.RandomFourierFeatures:
            return "rff"


@dataclass(frozen=True, order=True)
class SirenConfig(MLPConfig):
    weight_constant: float
    beta: float
    omega_0: float
    apply_omega_when_linear: bool
    boost_bias: bool

    @property
    def config_name(self):
        generic_name = super().config_name
        return "_".join(
            [
                f"{self.omega_0:.2f}",
                str(self.beta),
                str(self.weight_constant),
                "Siren",
                generic_name,
            ]
        )


@dataclass(frozen=True, order=True)
class WireConfig(MLPConfig):
    omega_0: float
    scale: float
    hidden_omega: float

    @property
    def config_name(self):
        generic_name = super().config_name
        return "_".join(
            [
                f"{self.omega_0:.2f}",
                f"{self.scale:.2f}",
                "WIRE",
                generic_name,
            ]
        )


@dataclass(frozen=True, order=True)
class FinerConfig(MLPConfig):
    omega_0: float
    first_bias_scale: Optional[float]
    hidden_omega: float

    @property
    def config_name(self):
        generic_name = super().config_name

        bias = f"{self.first_bias_scale}"
        # if self.first_bias_scale is not None:
        #     bias = f"{self.first_bias_scale:.2f}"
        return "_".join(
            [
                f"{self.omega_0:.2f}",
                bias,
                "Finer",
                generic_name,
            ]
        )


@dataclass(frozen=True, order=True)
class ReLUConfig(MLPConfig):
    sigma: float

    @property
    def config_name(self):
        generic_name = super().config_name
        return "_".join([str(self.sigma), "ReLU", generic_name])


@dataclass(frozen=True, order=False)
class LoggerConfig(ComparableObject):
    group: str
    save_dir: str
    extra_config: SortedDict
    val_check_interval: int

    def dict(self):
        d = asdict(self)
        return d


@dataclass(frozen=True, order=False)
class SpectrumConfig(ComparableObject):
    use_baseline: bool
    eval_before_training: bool
    resize_before_crop: bool
    resize_sizes: List[int | float]
    crop_sizes: List[int | float]
    selection_methods: List[str]
    initial_evals: int


@dataclass(frozen=True, order=False)
class OptimizerConfig(ComparableObject):
    learning_rate: float
    scheduler_name: str | None = None
    scheduler_args: dict | None = None


@dataclass(frozen=True, order=False)
class ExperimentConfig(ComparableObject):
    model_type: str
    seed: int
    epochs: int
    learning_rate: float
    optimizer: OptimizerConfig
    batch_size: int
    training_type: TrainingType
    model_config: MLPConfig
    spectrum_config: SpectrumConfig
    logger_config: LoggerConfig
    possible_configs: Optional[tuple[MLPConfig]]
    dataset_name: Optional[str]
    dataset_path: Optional[Path]
    dataset_id: str
    dataset_source: str
    checkpoints_base_path: str
    results_base_path: str
    checkpoints_frequency: int
    fast_dev_run: bool
    accelerator: str
    device: str
    config_selection_method: str = ""
    config_hash = -1

    def __post_init__(self):
        s = str(self)
        h = int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)
        object.__setattr__(self, "config_hash", str(h))

    @property
    def checkpoints_path(self):
        return f"{self.checkpoints_base_path}/{self.logger_config.group}_{self.dataset_id}_{self.seed}_{self.model_config.config_name}_{self.config_hash}"

    @property
    def results_path(self):
        return f"{self.results_base_path}/{self.logger_config.group}_{self.dataset_id}_{self.seed}_{self.model_config.config_name}_{self.config_hash}"

    def dict(self):
        d = asdict(self)
        del d["possible_configs"]
        return d

    def __hash__(self):
        return int(self.config_hash)

    def change_dataset(self, *, dataset_name=None, dataset_path=None):
        dataset_id = dataset_name
        dataset_source = "scikit"
        if dataset_path is not None:
            dataset_id = Path(dataset_path).name.split(".")[0]
            dataset_source = Path(dataset_path).absolute().parent.name

        return dataclasses.replace(
            self,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            dataset_id=dataset_id,
            dataset_source=dataset_source,
        )

    def __iter__(self):
        return (getattr(self, field.name) for field in dataclasses.fields(self))

    def __lt__(self, other):
        for a, b in zip(self, other):
            # treat None as the smallest possible value
            if a is None and b is not None:
                return True
            if a is not None and b is None:
                return False

            if a == b:
                continue
            return a < b

    def prepare_run_configs(self, datasets: List[str]):
        run_configs = []
        if self.training_type == TrainingType.ONE:
            for dataset in datasets:
                run_config = self.change_dataset(dataset_path=dataset)
                run_configs.append(run_config)
        elif (
            self.training_type == TrainingType.ALL
            or self.training_type == TrainingType.SAVE_OUT
        ):
            for dataset in datasets:
                for model_config in self.possible_configs:
                    run_config = dataclasses.replace(self, model_config=model_config)
                    run_config = run_config.change_dataset(dataset_path=dataset)
                    run_configs.append(run_config)
        elif self.training_type == TrainingType.BEST:
            for dataset in datasets:
                run_config = self.change_dataset(dataset_path=dataset)

                for selection_method in run_config.spectrum_config.selection_methods:
                    run_configs.append(
                        dataclasses.replace(
                            run_config, config_selection_method=selection_method
                        )
                    )

        return sorted(run_configs)


def _omega_to_dataclass(omega: ListConfig | DictConfig) -> ExperimentConfig:
    if omega.model.type == "siren":
        model_config = SirenConfig(
            hidden_features=omega.model.hidden_features,
            hidden_layers=omega.model.hidden_layers,
            weight_constant=omega.model.weight_constant,
            beta=omega.model.beta,
            omega_0=omega.model.omega_0,
            apply_omega_when_linear=omega.model.apply_omega_when_linear,
            boost_bias=omega.model.boost_bias,
        )
        omega_0_values = omega.model.param_grid.omega_0
        beta_values = omega.model.param_grid.beta_values
        weight_constant_values = omega.model.param_grid.weight_constant
        configs = tuple(
            itertools.product(omega_0_values, beta_values, weight_constant_values)
        )
        possible_configs = tuple(
            SirenConfig(
                hidden_features=omega.model.hidden_features,
                hidden_layers=omega.model.hidden_layers,
                weight_constant=weight_constant,
                beta=beta,
                omega_0=omega_0,
                apply_omega_when_linear=omega.model.apply_omega_when_linear,
                boost_bias=omega.model.boost_bias,
            )
            for omega_0, beta, weight_constant in configs
        )
    elif omega.model.type == "relu":
        model_config = ReLUConfig(
            hidden_features=omega.model.hidden_features,
            hidden_layers=omega.model.hidden_layers,
            sigma=omega.model.sigma,
        )
        possible_configs = tuple(
            ReLUConfig(
                hidden_features=omega.model.hidden_features,
                hidden_layers=omega.model.hidden_layers,
                sigma=sigma,
            )
            for sigma in omega.model.param_grid.sigma
        )
    elif omega.model.type == "wire":
        model_config = WireConfig(
            hidden_features=omega.model.hidden_features,
            hidden_layers=omega.model.hidden_layers,
            omega_0=omega.model.omega_0,
            scale=omega.model.scale,
            hidden_omega=omega.model.hidden_omega,
        )
        omega_0_values = omega.model.param_grid.omega_0
        scale_values = omega.model.param_grid.scale
        configs = tuple(itertools.product(omega_0_values, scale_values))
        possible_configs = tuple(
            WireConfig(
                hidden_features=omega.model.hidden_features,
                hidden_layers=omega.model.hidden_layers,
                hidden_omega=omega.model.hidden_omega,
                omega_0=omega_0,
                scale=scale,
            )
            for omega_0, scale in configs
        )
    elif omega.model.type == "finer":
        model_config = FinerConfig(
            hidden_features=omega.model.hidden_features,
            hidden_layers=omega.model.hidden_layers,
            omega_0=omega.model.omega_0,
            first_bias_scale=omega.model.first_bias_scale,
            hidden_omega=omega.model.hidden_omega,
        )
        omega_0_values = omega.model.param_grid.omega_0
        scale_values = omega.model.param_grid.first_bias_scale
        configs = tuple(itertools.product(omega_0_values, scale_values))
        possible_configs = tuple(
            FinerConfig(
                hidden_features=omega.model.hidden_features,
                hidden_layers=omega.model.hidden_layers,
                hidden_omega=omega.model.hidden_omega,
                omega_0=omega_0,
                first_bias_scale=scale,
            )
            for omega_0, scale in configs
        )
    else:
        raise ValueError(f"Unknown model type: {omega.model.type}")

    scheduler = omega.optimizer.get("scheduler", dict())
    optimizer_config = OptimizerConfig(
        learning_rate=omega.optimizer.learning_rate,
        scheduler_name=scheduler.get("name", None),
        scheduler_args=scheduler.get("args", None),
    )

    if omega.training_type == "all":
        training_type = TrainingType.ALL
    elif omega.training_type == "best":
        training_type = TrainingType.BEST
    elif omega.training_type == "one":
        training_type = TrainingType.ONE
    elif omega.training_type == "save":
        training_type = TrainingType.SAVE_OUT
    else:
        raise ValueError(f"Unknown training type: {omega.training_type}")

    logger_config = LoggerConfig(
        group=omega.logger.group,
        save_dir=omega.logger.save_dir,
        extra_config=SortedDict(),
        val_check_interval=omega.logger.val_check_interval,
    )

    spectrum_config = SpectrumConfig(
        use_baseline=omega.spectrum.use_baseline,
        eval_before_training=omega.spectrum.eval_before_training,
        resize_before_crop=omega.spectrum.resize_before_crop,
        resize_sizes=omega.spectrum.resize_sizes,
        crop_sizes=omega.spectrum.crop_sizes,
        selection_methods=omega.spectrum.selection_methods,
        initial_evals=omega.spectrum.initial_evals,
    )

    dataset_name = None
    if not OmegaConf.is_missing(omega.dataset, "name"):
        dataset_name = omega.dataset.name
    dataset_path = None
    if not OmegaConf.is_missing(omega.dataset, "path"):
        dataset_path = Path(omega.dataset.path)

    if dataset_name is not None and dataset_path is not None:
        raise ValueError("Provide only dataset path or dataset name.")
    dataset_id = dataset_name
    dataset_source = "scikit"
    if dataset_path is not None:
        dataset_id = Path(dataset_path).name.split(".")[0]
        dataset_source = Path(dataset_path).absolute().parent.name

    experiment_config = ExperimentConfig(
        seed=omega.seed,
        model_type=omega.model.type,
        learning_rate=omega.optimizer.learning_rate,
        model_config=model_config,
        possible_configs=possible_configs,
        training_type=training_type,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        dataset_id=dataset_id,
        dataset_source=dataset_source,
        logger_config=logger_config,
        checkpoints_base_path=omega.checkpoints_path,
        checkpoints_frequency=omega.checkpoints_frequency,
        results_base_path=omega.results_path,
        fast_dev_run=omega.fast_dev_run,
        epochs=omega.epochs,
        batch_size=omega.batch_size,
        accelerator=omega.accelerator,
        device="cuda" if omega.accelerator == "gpu" else "cpu",
        spectrum_config=spectrum_config,
        optimizer=optimizer_config,
    )
    return experiment_config


def load_config(yaml_file, cli_args=None) -> ExperimentConfig:
    if cli_args is None:
        cli_args = []
    yaml_conf = OmegaConf.load(yaml_file)
    if yaml_conf.get("defaults", False):
        dir_name = os.path.dirname(yaml_file)
        defaults = [
            OmegaConf.load(str(os.path.join(dir_name, f))) for f in yaml_conf.defaults
        ]
        defaults = OmegaConf.merge(*defaults)
        yaml_conf = OmegaConf.merge(defaults, yaml_conf)
    cli_conf = OmegaConf.from_cli(cli_args)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(conf)
    return _omega_to_dataclass(conf)
