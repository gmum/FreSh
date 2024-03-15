import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pytorch_lightning as pl

from config_utils import ExperimentConfig, TrainingType, load_config
from datasets import load_dataset
from metrics import prepare_spectrum_generators
from models import INR
from utils import _log_dataset_props, run_trainer, initialise_logger


def prepare_argparser():
    parser = argparse.ArgumentParser(description="Train a baseline INR or a FreSh INR.")
    parser.add_argument("--conf", default=None, type=str)
    parser.add_argument("--config_id", default=0, type=int)
    parser.add_argument(
        "--dataset_sources",
        default=None,
        type=str,
        help="List of folders with images.",
    )
    parser.add_argument("--datasets", default=None, type=str, help="List of files.")
    parser.add_argument("--count", action="store_true")
    return parser


def train_from_config(
    experiment_config: ExperimentConfig,
    initial_wasserstein=None,
    initial_wasserstein_se=None,
    initial_wasserstein_time=None,
):
    pl.seed_everything(experiment_config.seed)
    dataset = load_dataset(experiment_config)

    spectrum_generators = prepare_spectrum_generators(experiment_config.spectrum_config)
    model = INR(
        dataset=dataset,
        spectrum_generators=spectrum_generators,
        experiment_config=experiment_config,
        initial_wasserstein=initial_wasserstein,
        initial_wasserstein_se=initial_wasserstein_se,
        initial_wasserstein_time=initial_wasserstein_time,
    )

    wandb_logger = initialise_logger(experiment_config)
    _log_dataset_props(wandb_logger, dataset)
    for name, value in experiment_config.logger_config.extra_config.items():
        wandb_logger.experiment.config[name] = value
    wandb_logger.experiment.config["config_hash"] = experiment_config.config_hash

    run_trainer(
        wandb_logger,
        model,
        dataset,
        experiment_config.fast_dev_run,
        accelerator=experiment_config.accelerator,
        checkpoints_path=experiment_config.checkpoints_path,
        checkpoints_frequency=experiment_config.checkpoints_frequency,
        max_steps=experiment_config.epochs,
        val_check_interval=experiment_config.logger_config.val_check_interval,
    )


def prep_datasets(opt: argparse.Namespace):
    if opt.dataset_sources is not None:
        dataset_sources = eval(opt.dataset_sources)
        datasets = []
        for dataset_source in dataset_sources:
            datasets.extend(
                [
                    str(x.absolute())
                    for x in Path(dataset_source).iterdir()
                    if x.is_file()
                ]
            )
    elif opt.datasets is not None:
        datasets = eval(opt.datasets)
    else:
        raise ValueError("Either --dataset_sources or --datasets should be provided")

    return sorted(datasets)


def prepare_best_model_config(run_config: ExperimentConfig):
    spectrum_generators = prepare_spectrum_generators(run_config.spectrum_config)
    dataset = load_dataset(run_config)

    wasserstein_logs = dict()
    outputs = dict()
    distances = []
    config_selection_method = run_config.config_selection_method
    for config in run_config.possible_configs:
        pl.seed_everything(run_config.seed)
        initial_wasserstein = dict()
        initial_wasserstein_se = dict()
        initial_wasserstein_time = dict()
        model = INR(
            dataset=dataset,
            spectrum_generators=spectrum_generators,
            experiment_config=dataclasses.replace(
                experiment_config, model_config=config
            ),
        )

        distance, distance_se, compute_time = (
            model.calculate_initial_wasserstein_distance(config_selection_method)
        )
        initial_wasserstein[config_selection_method] = distance
        initial_wasserstein_se[config_selection_method] = distance_se
        initial_wasserstein_time[config_selection_method] = compute_time
        wasserstein_logs[config] = (
            initial_wasserstein,
            initial_wasserstein_se,
            initial_wasserstein_time,
        )
        outputs[config] = config
        distances.append(distance)

    best_idx = np.argmin(distances)
    best_config = run_config.possible_configs[best_idx]
    initial_wasserstein, initial_wasserstein_se, _ = (
        wasserstein_logs[best_config]
    )
    initial_wasserstein_total_time = 0
    for _, (_, _, time) in wasserstein_logs.items():
        initial_wasserstein_total_time += time[config_selection_method]
    return (
        best_config,
        initial_wasserstein,
        initial_wasserstein_se,
        {config_selection_method: initial_wasserstein_total_time},
    )


if __name__ == "__main__":
    parser = prepare_argparser()
    opt, extras = parser.parse_known_args()
    experiment_config = load_config(opt.conf, cli_args=extras)
    datasets = prep_datasets(opt)

    run_configs = experiment_config.prepare_run_configs(datasets=datasets)
    if opt.count:
        print(f"Total number of configurations: {len(run_configs)}")
        exit(0)

    run_config = run_configs[opt.config_id]
    extra_config = run_config.logger_config.extra_config.copy()
    extra_config["config_id"] = opt.config_id
    logger_config = dataclasses.replace(
        run_config.logger_config, extra_config=extra_config
    )
    run_config = dataclasses.replace(run_config, logger_config=logger_config)
    if experiment_config.training_type in {TrainingType.ALL, TrainingType.ONE}:
        train_from_config(run_config)
    elif experiment_config.training_type == TrainingType.BEST:
        (
            best_config,
            initial_wasserstein,
            initial_wasserstein_se,
            initial_wasserstein_time,
        ) = prepare_best_model_config(run_config)

        run_config = dataclasses.replace(run_config, model_config=best_config)
        logger_config = dataclasses.replace(
            run_config.logger_config,
            extra_config=run_config.logger_config.extra_config,
        )
        train_from_config(
            run_config,
            initial_wasserstein=initial_wasserstein,
            initial_wasserstein_se=initial_wasserstein_se,
            initial_wasserstein_time=initial_wasserstein_time,
        )
    elif experiment_config.training_type == TrainingType.SAVE_OUT:
        output_root = Path("./model_outputs")
        dataset_root = Path("./datasets_numpy")
        pl.seed_everything(run_config.seed)
        dataset = load_dataset(run_config)
        model = INR(
            dataset=dataset,
            spectrum_generators=dict(),
            experiment_config=run_config,
        )

        output_folder = (
            output_root
            / run_config.model_type
            / run_config.dataset_source
            / run_config.dataset_id
            / f"{model.net.spectral_parameters}"
        )
        output_folder.mkdir(parents=True, exist_ok=True)
        for i in range(experiment_config.spectrum_config.initial_evals):
            model = INR(
                dataset=dataset,
                spectrum_generators=dict(),
                experiment_config=run_config,
            )

            _, out, _, _ = model._infer_full_img()
            out = out.cpu().detach().numpy()
            np.save(output_folder / f"{i}.npy", out)

        dataset_output_folder = dataset_root / run_config.dataset_source
        dataset_output_folder.mkdir(parents=True, exist_ok=True)
        dataset_file = dataset_output_folder / f"{run_config.dataset_id}.npy"
        if not dataset_file.exists():
            np.save(dataset_file, dataset.signal_channel_first)
