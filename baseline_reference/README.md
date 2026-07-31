# MCMamba-SSF experiment

This is an isolated experiment derived from the McNet-SSF setup. It does not
modify or resume the existing McNet-SSF run.

The four BLSTM stages were replaced by offline Bi-Mamba stages using the output
and hidden dimensions reported in:

- W. Ren et al., "Leveraging Joint Spectral and Spatial Learning with MAMBA for
  Multichannel Speech Enhancement", arXiv:2409.10376.

The published MCMamba is a non-steered speech-enhancement model and the paper
does not provide public source code. For this SSF experiment, the 180-way
target-DoA encoding is projected and added to the first frequency-axis
Bi-Mamba representation. This adaptation is recorded in each run's
`provenance.json` and `effective_config.yaml`.

For a controlled McNet-SSF comparison, the dataset, STFT, CRM target, loss,
optimizer, scheduler, and effective global batch size remain unchanged.
