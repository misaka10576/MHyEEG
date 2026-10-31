# BERN on SEED-IV

This directory is an independent, paper-constrained reimplementation of Zhou
et al., *BERN: A Novel Framework for Enhanced Emotion Recognition Through the
Integration of EEG and Eye Movement Features* (IEEE TCDS, 2025,
[DOI 10.1109/TCDS.2025.3590031](https://doi.org/10.1109/TCDS.2025.3590031)).
It is not the authors' source code and does not claim line-by-line replication.

Everything created by this implementation—code, configuration, cached
ImageNet weights, checkpoints, predictions, metrics and reports—stays under
`BERN/`. The existing `SEED_IV/` directory is read only.

## Protocol

Each of the 15 subjects and 3 sessions is an independent within-subject run.
Trials 1–16 are training data and trials 17–24 are held-out test data. Epoch
selection uses the last-occurring trial of each class among trials 1–16 as
four complete validation trials. A model is trained on the other 12 trials
with validation-loss early stopping. It is then reinitialized and trained for
the selected number of epochs on all 16 training trials. Test data is
standardized and evaluated exactly once only after final training.

EEG statistics are fitted separately for every electrode/band and eye
statistics for every feature. Selection statistics use only the 12 selection
trials; final statistics are refitted using all 16 training trials. Non-finite
values are mean-imputed (zero after z-score). The three modes share identical
trial splits.

The formal defaults are Adam, learning rate `1e-4`, batch size 32, weight decay
`0.01`, at most 500 epochs, early-stopping patience 50, ordinary cross entropy,
FP32, deterministic execution and seed 2024. The default feature is `de_LDS`;
`--eeg-feature-kind de_movingAve` is available for the unresolved smoothing
choice.

## Model

The public interface is:

```python
from BERN import BERN

model = BERN(mode="fusion", pretrained=True)
logits = model(eeg, eye)  # eeg [B,62,5], eye [B,31], logits [B,4]
```

The EEG branch maps all 62 channels onto the standard 5×9×9 SEED grid,
bilinearly resizes it to 5×112×112, applies a `(5,3,3)` 3-D convolution to
produce 64×56×56, and passes the result through the four ResNet-18 residual
stages and a 512→128 projection. The residual stages load the official
ImageNet-1K ResNet-18 weights; the incompatible image stem and classifier are
not loaded. This implementation has no `torchvision` runtime dependency.

The eye branch applies four 31→15→31 BN/ReLU/Sigmoid residual blocks followed
by `Conv1d(31,128,1)`, BN and ReLU. Fusion independently projects the EEG and
eye embeddings into eight 32-dimensional tokens, then applies two four-head
EEG-query/eye-key-value cross-attention layers. Each layer uses pre-normalized
inputs, an attention residual and output normalization. The two-layer output
is flattened and projected from 256 back to 128 before classification.
See [method_provenance.md](docs/method_provenance.md) for shapes, formulae and
the boundary between paper facts and necessary assumptions.

## Run

Run from the repository root in the existing environment:

```bash
conda run --no-capture-output -n mhyeeg \
  python -m unittest discover -s BERN/tests -v
```

Populate and verify the official weight cache (the runner also does this
automatically on first EEG/fusion run):

```bash
conda run --no-capture-output -n mhyeeg python -c \
  "from BERN.model import EEGEncoder; print(EEGEncoder(pretrained=True).pretrained_report)"
```

Run the gated GPU Fusion smoke test for subject 1/session 1:

```bash
conda run --no-capture-output -n mhyeeg \
  python -m BERN.run --mode fusion --subject 1 --session 1 \
  --smoke-test --resume
```

`--smoke-test` processes only two batches per epoch and writes below
`BERN/outputs/vector_attention/smoke`; its accuracy is not a scientific
result. Formal training must not be started if either layer's normalized
attention entropy exceeds 0.9995.

After the smoke gate passes, run the one full subject/session check and then
resume all 45 affected Fusion experiments:

```bash
conda run --no-capture-output -n mhyeeg \
  python -m BERN.run --mode fusion --subject 1 --session 1 --resume
conda run --no-capture-output -n mhyeeg \
  python -m BERN.run --mode fusion --all-runs --resume
```

Every run writes its resolved configuration, selection history/best
checkpoint, final checkpoint, per-window logits and predictions, and metrics
to:

```text
BERN/outputs/vector_attention/fusion/subject_<NN>/session_<N>/
```

Completed `metrics.json` files are skipped with `--resume`; interrupted
selection and final stages continue from optimizer-bearing last checkpoints.
Those larger temporary checkpoints are removed after their stage completes.

Validate exact paired 45-run coverage, architecture identifiers, split
integrity, finite outputs, test-prediction provenance and attention
diagnostics, then generate the comparison report:

```bash
conda run --no-capture-output -n mhyeeg \
  python -m BERN.compare_vector_attention
```

The new checkpoints and results never overwrite the original 135 runs.
`BERN/reports/vector_attention/` contains accuracy, Macro-F1, subject/session
breakdowns, all 45 paired deltas, win/tie/loss counts, the paper gap and
attention diagnostics. The original Eye 47.69% and EEG 36.10% results remain
fixed comparisons. Acceptance is based on non-degenerate attention, protocol
correctness, complete artifacts and reproducibility—not forcing 83.33%.

## Tests

The standard-library `unittest` suite covers:

- all channel/grid locations and zero-filled gaps;
- exact, non-overlapping complete-trial splits;
- leakage-safe standardization;
- three model modes and paper-constrained intermediate dimensions;
- complete official residual-layer weight coverage;
- 8×32 vector tokens, 4×8×8 head weights, modality sensitivity and two-layer gradients;
- finite entropy, uniform-deviation and attention-variance diagnostics;
- finite one-step/two-stage training, selected-epoch final retraining;
- checkpoint, prediction, resume, validation and aggregation artifacts.

## Licensing

Project-specific code in this directory is provided for research
reimplementation. Third-party attribution and license copies for the
ResNet-derived structure and consulted LibEER implementation are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/`.
