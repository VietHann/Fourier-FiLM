# MCMamba-SSF Circular-FiLM improvement code snapshot

Source experiment:

`/project/anhlt/0607/jnf_mamba_experiment/mcmamba_ssf_circular_film_experiment_20260729`

Copied on 2026-07-31 without modifying the source experiment or its running
training job.

## Main experiment files

- `mcmamba_ssf_circular_film.py`: model implementation, including circular
  Fourier DoA encoding, the DoA MLP, zero-initialized FiLM heads, and
  post-LayerNorm FiLM in both spatial Bi-Mamba stages.
- `train_mcmamba_ssf_circular_film.py`: training entrypoint and improvement
  hyperparameters.
- `run_mcmamba_ssf_circular_film_sp1_8k_bs12.sh`: launch recipe used by the
  source experiment.
- `effective_config.yaml`: resolved configuration from the active run.
- `README.md`: architecture and controlled-comparison notes.

## Included local dependencies

- `upstream_snapshot/config/ssf_config.yaml`
- `upstream_snapshot/data/datamodule.py`
- `upstream_snapshot/data/dataset.py`
- `upstream_snapshot/data/data_gen_var_pos.py`
- `upstream_snapshot/models/exp_ssf.py`
- `upstream_snapshot/models/exp_enhancement.py`
- `upstream_snapshot/utils/log_images.py`
- `paper_reproduction/validate_original_var_pos_dataset.py`

These are the exact local files imported or opened by the training entrypoint
and its SSF training/validation path. The upstream source commit is
`ddbd620d8e31f1e20bb93ef09bcf4c3f40063c7a`.

## Baseline reference

`baseline_reference/` contains the four source files from the original
MCMamba-SSF experiment so the Circular-FiLM changes can be reviewed against
the unchanged baseline.

Checkpoints, TensorBoard events, logs, datasets, and Python cache files are
intentionally excluded.
