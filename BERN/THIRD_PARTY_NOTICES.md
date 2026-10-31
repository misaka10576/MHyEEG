# Third-party notices

## PyTorch / torchvision ResNet-18

`BERN/model.py` contains a small local implementation of the standard
ResNet-18 basic block and four residual stages, structurally compatible with
the official PyTorch/torchvision ImageNet-1K checkpoint. No torchvision Python
package is imported. The official weight URL is:

`https://download.pytorch.org/models/resnet18-f37072fd.pth`

PyTorch/torchvision are distributed under a BSD-style license. A copy is
included at [licenses/TORCHVISION.txt](licenses/TORCHVISION.txt).

## LibEER

The local LibEER repository was consulted to cross-check SEED channel order,
session labels and conventional seed choice. BERN does not import LibEER or
copy its model implementations. LibEER is MIT-licensed; its license is
included at [licenses/LIBEER.txt](licenses/LIBEER.txt).

## TorchEEG

The local TorchEEG repository was consulted to cross-check the SEED-IV 62
electrode 9×9 layout and labels. The relevant constants are data facts
duplicated locally to avoid a runtime dependency. See the repository-level
`torcheeg/LICENSE` for its license.
