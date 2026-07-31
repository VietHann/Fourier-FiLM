#!/usr/bin/env python3
"""Train circular-DoA FiLM-conditioned MCMamba on the McNet-SSF data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
import yaml


ROOT = Path("/project/anhlt/0607")
UPSTREAM_SRC = ROOT / "Track2/deep-non-linear-filter/src"
EXPERIMENT_ROOT = Path(__file__).resolve().parent
PAPER_REPRODUCTION = ROOT / "jnf_mamba_experiment/paper_reproduction"
RUN_ROOT = EXPERIMENT_ROOT / "runs"
sys.path.insert(0, str(UPSTREAM_SRC))
sys.path.insert(0, str(EXPERIMENT_ROOT))

from data.datamodule import HDF5DataModule  # noqa: E402
from mcmamba_ssf_circular_film import MCMambaSSFCircularFiLM  # noqa: E402
from models.exp_ssf import SSFExp  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-rate", type=int, choices=(8000, 16000), required=True)
    parser.add_argument("--devices", type=int, default=3)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--accumulate-grad-batches", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument(
        "--version", default="mcmamba_ssf_circular_film_sp1_bs12_500ep"
    )
    parser.add_argument("--ckpt-path", type=Path)
    parser.add_argument("--original-var-pos-data-dir", type=Path, required=True)
    parser.add_argument("--n-interferers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_interferers < 1:
        raise ValueError("n-interferers must be positive")
    if args.devices < 1 or args.accumulate_grad_batches < 1:
        raise ValueError("devices and accumulate-grad-batches must be positive")

    sample_rate = args.sample_rate
    with (UPSTREAM_SRC / "config/ssf_config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    data_dir = args.original_var_pos_data_dir.resolve()
    stem = f"ch3_sp{args.n_interferers}_var_target"
    hdf5_path = data_dir / f"prep_mix_{stem}.hdf5"
    metadata_path = data_dir / f"prep_mix_meta_{stem}.json"
    prep_files = {
        "train_data": str(hdf5_path),
        "train_meta": str(metadata_path),
        "val_data": str(hdf5_path),
        "val_meta": str(metadata_path),
    }
    for path in prep_files.values():
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    config["data"].update(
        {
            "n_channels": 3,
            "batch_size": args.per_device_batch_size,
            "prep_files": prep_files,
            "fs": sample_rate,
            "meta_frame_length": 3 * sample_rate,
            "stft_length_samples": round(0.032 * sample_rate),
            "stft_shift_samples": round(0.016 * sample_rate),
            "n_workers": args.workers,
        }
    )
    config["training"].update(
        {
            "devices": args.devices,
            "strategy": "ddp" if args.devices > 1 else "auto",
            "max_epochs": args.max_epochs,
        }
    )
    network_config = {
        "n_channels": 3,
        "n_cond_emb_dim": 180,
        "freq_hidden": 128,
        "narrow_hidden": 256,
        "subband_hidden": 384,
        "fullband_hidden": 128,
        "projection_size": 64,
        "subband_noisy_radius": 3,
        "subband_embedding_radius": 2,
        "temporal_context": [5, 5],
        "output_activation": "tanh",
        "reference_channel": 0,
        "bidirectional": True,
        "mamba_d_state": 16,
        "mamba_d_conv": 4,
        "mamba_expand": 2,
        "doa_fourier_harmonics": 4,
        "doa_condition_dim": 128,
    }
    config["network"] = network_config
    config["derived"] = {
        "model": "MCMamba-SSF-Circular-FiLM",
        "paper": "https://arxiv.org/abs/2409.10376",
        "conditioning_reference": "https://arxiv.org/abs/2605.18442",
        "adaptation": (
            "Published offline MCMamba stages with circular Fourier DoA "
            "encoding and post-LayerNorm FiLM in both spatial stages"
        ),
        "micro_batch_global_size": args.devices * args.per_device_batch_size,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "effective_global_batch_size": (
            args.devices
            * args.per_device_batch_size
            * args.accumulate_grad_batches
        ),
        "sample_rate": sample_rate,
        "train_n_interferers": args.n_interferers,
        "train_total_speakers": args.n_interferers + 1,
        "data_profile": "upstream_data_gen_var_pos",
        "steering": (
            "Four-harmonic circular DoA encoding drives zero-initialized "
            "FiLM scale/bias in full-band and narrow-band spatial Bi-Mamba"
        ),
        "comparison_control": "Same SSF data, loss, optimizer, and scheduler as McNet-SSF",
    }

    pl.seed_everything(config.get("seed", 123), workers=True)
    run_dir = RUN_ROOT / f"{sample_rate // 1000}k" / args.version
    run_dir.mkdir(parents=True, exist_ok=True)
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        subprocess.run(
            [
                sys.executable,
                str(PAPER_REPRODUCTION / "validate_original_var_pos_dataset.py"),
                "--sample-rate",
                str(sample_rate),
                "--data-dir",
                str(data_dir),
                "--n-interferers",
                str(args.n_interferers),
                "--output",
                str(run_dir / "preflight_data_audit.json"),
            ],
            check=True,
        )
        with (run_dir / "effective_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        with (run_dir / "provenance.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "model_class": (
                        "mcmamba_ssf_circular_film.MCMambaSSFCircularFiLM"
                    ),
                    "experiment_class": "models.exp_ssf.SSFExp",
                    "data_module_class": "data.datamodule.HDF5DataModule",
                    "architecture_paper": "https://arxiv.org/abs/2409.10376",
                    "official_mcmamba_code": None,
                    "note": (
                        "The paper has no public reference implementation and no "
                        "SSF steering; this run is a documented circular-DoA "
                        "FiLM-conditioned adaptation."
                    ),
                },
                handle,
                indent=2,
            )

    logger = pl_loggers.TensorBoardLogger(
        save_dir=str(run_dir), name="tensorboard", version=0, log_graph=False
    )
    checkpoint = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        filename="best-{epoch:03d}-{monitor_loss:.6f}",
        monitor="monitor_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    datamodule = HDF5DataModule(**config["data"])
    model = MCMambaSSFCircularFiLM(**network_config)
    experiment = SSFExp(
        model=model,
        stft_length=config["data"]["stft_length_samples"],
        stft_shift=config["data"]["stft_shift_samples"],
        **config["experiment"],
    )
    training = config["training"]
    trainer = pl.Trainer(
        enable_model_summary=True,
        logger=logger,
        devices=training["devices"],
        log_every_n_steps=100,
        max_epochs=training["max_epochs"],
        gradient_clip_val=training["gradient_clip_val"],
        gradient_clip_algorithm=training["gradient_clip_algorithm"],
        strategy=training["strategy"],
        accelerator=training["accelerator"],
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[checkpoint, ModelSummary(max_depth=2)],
        default_root_dir=str(run_dir),
    )
    trainer.fit(
        experiment,
        datamodule,
        ckpt_path=str(args.ckpt_path.resolve()) if args.ckpt_path else None,
    )


if __name__ == "__main__":
    main()
