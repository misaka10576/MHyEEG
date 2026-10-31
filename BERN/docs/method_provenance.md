# Method provenance and unresolved choices

This document records what is explicit in the BERN paper, what was
cross-checked against local projects, and what this reimplementation had to
infer. The implementation should be described as a **paper-constrained
reimplementation**, never as the authors' exact code.

## Primary paper

Zhou et al., *BERN: A Novel Framework for Enhanced Emotion Recognition Through
the Integration of EEG and Eye Movement Features*, IEEE Transactions on
Cognitive and Developmental Systems (2025),
[DOI 10.1109/TCDS.2025.3590031](https://doi.org/10.1109/TCDS.2025.3590031).

### Explicit constraints used here

- SEED-IV EEG and eye-movement features, four emotion classes.
- The subject-dependent protocol trains on the first 16 trials and tests on
  the final 8 trials in each subject/session.
- EEG features use five frequency bands and 62 electrodes arranged spatially,
  a 3-D convolutional front end, ResNet-18 residual feature extraction and a
  128-dimensional representation.
- The eye branch uses four 31→15→31 residual feature blocks and maps 31
  features to a 128-dimensional representation with a 1-D convolution.
- EEG is the query and eye movement supplies key/value to two cross-attention
  fusion layers with residual additions.
- Adam, learning rate `1e-4`, batch size 32, weight decay/L2 `0.01`, a maximum
  of 500 epochs and validation-loss early stopping with patience 50.
- Reported SEED-IV accuracies used for comparison: Eye 73.07%, EEG 81.33%,
  Fusion 83.33%.

Ordinary cross entropy is used because the paper does not state class weights,
label smoothing or an auxiliary loss. Adam's `weight_decay=0.01` realizes the
stated L2 term once; no duplicate explicit penalty is added.

## Locally cross-checked dataset facts

The standard SEED/SEED-IV order of 62 electrodes and its 9×9 grid agree
between:

- `torcheeg/torcheeg/datasets/constants/emotion_recognition/seed_iv.py`
  (`SEED_IV_CHANNEL_LIST`, `SEED_IV_LOCATION_LIST`);
- `LibEER/LibEER/data_utils/constants/seed.py`
  (`SEED_CHANNEL_NAME`);
- the MAT feature dimensions in local `SEED_IV/eeg_feature_smooth`.

The 24 labels for each session agree between the existing local loader,
TorchEEG's SEED-IV feature loader and LibEER's loader. BERN duplicates the
small constants rather than importing either repository, keeping runtime
dependencies isolated.

## Necessary assumptions

### Smoothed feature variant

The paper says “DE features” but does not disambiguate the dataset's
`de_LDS` and `de_movingAve` variables. The default is `de_LDS`; the other is a
CLI switch. All reports include the selected feature name.

### Validation and final retraining

No validation protocol is disclosed. Within trials 1–16, this implementation
selects the last-occurring complete trial from each of the four classes for
validation. The other 12 trials select the best epoch. The model and optimizer
are then reinitialized, standardization is refitted, and all 16 trials are
used for exactly that many final epochs. Trials 17–24 never influence
standardization, stopping or epoch choice.

### EEG 3-D convolution

The input grid has shape `[B,5,9,9]`. It is bilinearly resized to
`[B,5,112,112]`, viewed as `[B,1,5,112,112]`, then convolved with:

```text
Conv3d(1, 64, kernel=(5,3,3), stride=(1,2,2), padding=(0,1,1))
```

This covers all five bands and a 3×3 spatial neighborhood and yields the
figure-compatible `[B,64,1,56,56]`, squeezed to `[B,64,56,56]`. BN, ReLU and
2-D max pooling precede ResNet-18 `layer1`…`layer4`. Only those residual-stage
keys are loaded from PyTorch's official ImageNet-1K weights. Global average
pooling and `Linear(512,128)` finish the branch.

### Cross-attention tokenization

The paper gives fusion equations but not token shapes or a head count. Using a
single 128-vector as one token would make softmax attention identically one.
The original reimplementation treated each feature coordinate as a scalar
token; this structurally weakened the score scale and produced near-uniform
attention. The corrected `vector_token_v2` variant independently projects
each 128-dimensional modality embedding into eight learned,
32-dimensional vector tokens.

For each of two independent layers, the fused/EEG tokens supply the query and
the unchanged eye tokens supply key and value. PyTorch
`MultiheadAttention(embed_dim=32, num_heads=4, dropout=0, batch_first=True)`
splits each token into four eight-dimensional heads and applies the standard
\(\sqrt{8}\) per-head scaling:

\[
q_h=W^h_q\operatorname{LN}(x),\quad
k_h=W^h_k\operatorname{LN}(e),\quad
v_h=W^h_v\operatorname{LN}(e)
\]

\[
A_h=\operatorname{softmax}\left(\frac{q_hk_h^\top}{\sqrt{8}}\right)
\in\mathbb{R}^{B\times4\times8\times8}
\]

\[
x'=\operatorname{LN}_{out}\left(x+\operatorname{MHA}(
\operatorname{LN}(x),\operatorname{LN}(e),\operatorname{LN}(e))\right).
\]

The output of layer 1 becomes the query state of layer 2; eye tokens do not
change. No FFN or auxiliary loss is added. The final `[B,8,32]` tensor is
flattened to 256 and projected to 128 for the existing classifier. This is a
documented inference, not a claim about unpublished author code.

During the single held-out forward pass, each layer records normalized
entropy, mean absolute deviation relative to the uniform probability, and
variance across keys. These are saved atomically with predictions so resume
cannot cause a second test evaluation.

### Reproducibility

The paper does not publish a seed. The default is 2024, a value also used by
the local LibEER workflows. Formal runs use CUDA FP32, deterministic
algorithms where PyTorch supports them, cuDNN benchmarking off and no AMP.
Each subject/session is independently initialized with the same seed.

## Leakage boundary

Selection code receives only the 12 training trials and 4 validation trials;
its function has no test-array argument. Final training receives only trials
1–16. Only after the final checkpoint exists are trials 17–24 transformed
with final-training statistics and passed through the model once.
`BERN.summarize` rejects split mismatches, test counts other than one,
non-finite values, missing predictions and predictions from non-test trials.
