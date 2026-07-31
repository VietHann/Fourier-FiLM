# MCMamba-SSF Circular-FiLM experiment

This is an isolated experiment derived from the running MCMamba-SSF setup. It
does not modify or resume that run.

The audio path retains the four offline Bi-Mamba stages: full-band spatial,
narrow-band spatial, sub-band spectral, and full-band spectral. Target DoA
conditioning is changed as follows:

- The existing 180-way one-hot vector is mapped through a fixed circular
  Fourier basis with harmonics 1 through 4.
- A two-layer MLP encodes the resulting eight sine/cosine values.
- Separate zero-initialized FiLM heads generate scale and bias for the
  full-band and narrow-band spatial stages.
- FiLM is applied after LayerNorm as `(1 + scale) * normalized + bias`.
- Spectral stages remain unconditioned.

The circular basis makes the wrap-around relationship between `-180°` and
`178°` explicit and permits continuous/soft directional inputs. FiLM provides
stronger feature-wise conditioning than adding one repeated embedding before
normalization.

The dataset, STFT, CRM target, loss, optimizer, scheduler, Mamba dimensions,
and effective global batch size are unchanged for a controlled comparison.
