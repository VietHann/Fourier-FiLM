#!/usr/bin/env python3
"""Validate completed native-rate output from the upstream var-pos generator."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np


ROOT = Path("/project/anhlt/0607")
UPSTREAM = ROOT / "Track2/deep-non-linear-filter"
UPSTREAM_GENERATOR = UPSTREAM / "src/data/data_gen_var_pos.py"
EXPERIMENT_ROOT = ROOT / "jnf_mamba_experiment/paper_reproduction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-rate", type=int, choices=(8000, 16000), required=True)
    parser.add_argument("--n-interferers", type=int, default=5)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_interferers < 1:
        raise ValueError("n_interferers must be positive")
    rate = args.sample_rate
    data_dir = (
        args.data_dir.resolve()
        if args.data_dir
        else EXPERIMENT_ROOT / "data_original_var_pos_parallel" / f"{rate // 1000}k"
    )
    stem = f"prep_mix_ch3_sp{args.n_interferers}_var_target"
    hdf5_path = data_dir / f"{stem}.hdf5"
    metadata_path = data_dir / f"prep_mix_meta_ch3_sp{args.n_interferers}_var_target.json"
    launch_path = data_dir / f"{stem}.launch.json"
    plan_path = data_dir / f"{stem}.plan.jsonl"
    for path in (hdf5_path, metadata_path, launch_path, plan_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with launch_path.open(encoding="utf-8") as handle:
        launch = json.load(handle)
    current_hash = hashlib.sha256(UPSTREAM_GENERATOR.read_bytes()).hexdigest()
    if launch.get("status") != "complete":
        raise ValueError(f"Generator launch status is not complete: {launch.get('status')}")
    if launch.get("upstream_commit") != "ddbd620":
        raise ValueError("Unexpected upstream commit in launch provenance")
    if launch.get("upstream_generator_sha256") != current_hash:
        raise ValueError("Upstream generator hash differs from the launch record")
    if launch.get("native_sample_rate") != rate:
        raise ValueError("Launch record has the wrong native sample rate")
    if launch.get("upstream_recipe", {}).get("n_interfering_speakers") != args.n_interferers:
        raise ValueError("Launch record has the wrong interferer count")
    if launch.get("workers") != 16:
        raise ValueError("Expected the audited 16-worker executor")
    if launch.get("plan_sha256") != hashlib.sha256(plan_path.read_bytes()).hexdigest():
        raise ValueError("Frozen job plan hash mismatch")

    commit = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "--short", "HEAD"], text=True
    ).strip()
    tracked_diff = subprocess.run(
        ["git", "-C", str(UPSTREAM), "diff", "--quiet", "--"], check=False
    ).returncode
    if commit != "ddbd620" or tracked_diff != 0:
        raise ValueError("Upstream repository provenance is no longer clean")

    with metadata_path.open(encoding="utf-8") as handle:
        metadata_root = json.load(handle)

    expected = {
        "train": {"count": 54_000, "per_angle": 300, "split": "train-100"},
        "val": {"count": 2_700, "per_angle": 15, "split": "dev"},
        "test": {"count": 1_800, "per_angle": 10, "split": "test"},
    }
    evidence = {
        "status": "valid",
        "sample_rate": rate,
        "upstream_commit": commit,
        "upstream_generator_sha256": current_hash,
        "upstream_tracked_diff": False,
        "source_corpus": "Libri2Mix isolated s1/s2",
        "n_interferers": args.n_interferers,
        "n_speakers_total": args.n_interferers + 1,
        "stages": {},
    }
    max_samples = 12 * rate
    with h5py.File(hdf5_path, "r") as storage:
        for stage, stage_expected in expected.items():
            count = int(stage_expected["count"])
            metadata = metadata_root.get(stage)
            if not isinstance(metadata, dict) or len(metadata) != count:
                raise ValueError(f"{stage} metadata count mismatch")
            expected_keys = {str(index) for index in range(count)}
            if set(metadata) != expected_keys:
                raise ValueError(f"{stage} metadata indices are not contiguous")
            angles = Counter()
            for index in range(count):
                sample = metadata[str(index)]
                direction = int(sample["target_dir"])
                if int(sample["target_angle"]) != direction:
                    raise ValueError(f"{stage}[{index}] target angle mismatch")
                angles[direction] += 1
                n_samples = int(sample["n_samples"])
                if not 0 < n_samples <= max_samples:
                    raise ValueError(f"{stage}[{index}] invalid sample count")
                source_paths = [sample["target_file"]]
                source_paths.extend(
                    sample[f"interf{i}_file"] for i in range(args.n_interferers)
                )
                if f"interf{args.n_interferers}_file" in sample:
                    raise ValueError(f"{stage}[{index}] has too many interferers")
                for source_path in source_paths:
                    resolved = Path(source_path).resolve()
                    expected_fragment = (
                        f"/LibriMix/Libri2Mix/wav{rate // 1000}k/min/"
                        f"{stage_expected['split']}/"
                    )
                    if expected_fragment not in str(resolved):
                        raise ValueError(
                            f"{stage}[{index}] source belongs to the wrong native split"
                        )
                    if resolved.parent.name not in ("s1", "s2"):
                        raise ValueError(f"{stage}[{index}] source is not isolated s1/s2")
            expected_angles = {
                angle: int(stage_expected["per_angle"])
                for angle in range(-180, 180, 2)
            }
            if dict(angles) != expected_angles:
                raise ValueError(f"{stage} does not have the exact 2-degree DoA grid")

            expected_shape = (count, 3, 3, max_samples)
            if storage[stage].shape != expected_shape:
                raise ValueError(f"{stage} HDF5 shape mismatch: {storage[stage].shape}")
            if int(storage.attrs.get(f"completed_{stage}", -1)) != count:
                raise ValueError(f"{stage} is not marked complete")
            check_indices = np.unique(np.linspace(0, count - 1, 96, dtype=int))
            min_component_energy = float("inf")
            for index in check_indices:
                n_samples = int(metadata[str(int(index))]["n_samples"])
                audio = storage[stage][int(index), :, :, :n_samples]
                if not np.isfinite(audio).all():
                    raise ValueError(f"{stage}[{index}] contains non-finite samples")
                energy = np.sum(np.square(audio, dtype=np.float64), axis=(1, 2))
                if np.any(energy <= 0.0):
                    raise ValueError(f"{stage}[{index}] contains an empty component")
                min_component_energy = min(min_component_energy, float(np.min(energy)))
            evidence["stages"][stage] = {
                "samples": count,
                "directions": 180,
                "samples_per_direction": int(stage_expected["per_angle"]),
                "hdf5_shape": list(expected_shape),
                "checked_audio_examples": int(len(check_indices)),
                "minimum_checked_component_energy": min_component_energy,
            }

    output_text = json.dumps(evidence, indent=2)
    print(output_text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
