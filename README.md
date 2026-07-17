## Hypercomplex Multimodal Emotion Recognition from EEG and Peripheral Physiological Signals :performing_arts:

Official PyTorch repository for the papers:
1) Hypercomplex Multimodal Emotion Recognition from EEG and Peripheral Physiological Signals, ICASSPW 2023. [[IEEEXplore](https://ieeexplore.ieee.org/abstract/document/10193329)][[ArXiv Preprint](https://arxiv.org/abs/2310.07648)]
2) Hierarchical Hypercomplex Network for Multimodal Emotion Recognition, MLSP 2024. [[IEEEXplore](https://ieeexplore.ieee.org/abstract/document/10734815)][[ArXiv Preprint](https://arxiv.org/abs/2409.09194)]
3) PHemoNet: A Multimodal Network for Physiological Signals, RTSI 2024. [[IEEEXplore](https://ieeexplore.ieee.org/abstract/document/10761462)][[ArXiv Preprint](https://arxiv.org/abs/2410.00010)]

Authors: 

Eleonora Lopez, Eleonora Chiarantano, [Eleonora Grassucci](https://sites.google.com/uniroma1.it/eleonoragrassucci/home-page), [Aurelio Uncini](https://www.uncini.com/), and [Danilo Comminiello](https://danilocomminiello.site.uniroma1.it/) from [ISPAMM Lab](https://sites.google.com/uniroma1.it/ispamm/) 🏘️

### 📰 News
- [2025.05.15] Released pretrained weights 💣
- [2025.05.14] Updated code with H2 and PHemoNet models from MLSP and RTSI papers! 👩🏻‍💻
- [2024.07] Extension papers have been accepted at MLSP and RTSI 2024!
- [2023.11.11] Code is available for HyperFuseNet! 👩🏼‍💻
- [2023.04.14] The paper has been accepted for presentation at ICASSP workshop 2023 🎉!

### Overview :blush:

### 📚 Papers & Models

| Model               | Paper                                                                                                                          | Arousal F1 | Arousal Acc | Valence F1 | Valence Acc | Highlights | Weights | 
|----------------------|-------------------------------------------------------------------------------------------------------------------------------|------------|-------------|------------|-------------|------------|---------|
| 🥇 **H2**           | MLSP 2024 [[IEEEXplore](https://ieeexplore.ieee.org/abstract/document/10734815)][[ArXiv](https://arxiv.org/abs/2409.09194)]    | **0.557**  | **56.91**   | **0.685**  | **67.87** | Hierarchical model with PHC-based encoders in modality-specific domains, achieves **best performance** | [Arousal](https://drive.google.com/file/d/1xvC5mVaoHG2UINJv-jJ8_pR1R5f2z1oG/view?usp=sharing) - [Valence](https://drive.google.com/file/d/1tBTmbxswkNTa9e_7_1RPnRQZSD-Kr9vf/view?usp=sharing)
| 🥈 **PHemoNet**     | RTSI 2024 [[IEEEXplore](https://ieeexplore.ieee.org/abstract/document/10761462)][[ArXiv](https://arxiv.org/abs/2410.00010)]    | 0.401      | 42.54       | 0.505      | 50.77 | PHM-based encoders with modality-specifc domains and revised hypercomplex fusion module | [Arousal](https://drive.google.com/file/d/1d8tF93EtHXOmC0IOa_ID0gn9JYxAHxhF/view?usp=sharing) - [Valence](https://drive.google.com/file/d/1b-KUJ_mhJhSG8AAHBeqE_39e8z67nwiJ/view?usp=sharing)
| 🥉 **HyperFuseNet** | ICASSPW 2023 [[IEEEXplore](https://ieeexplore.ieee.org/abstract/document/10193329)][[ArXiv](https://arxiv.org/abs/2310.07648)] | 0.397      | 41.56       | 0.436      | 44.30 | Introduces hypercomplex fusion module | [Arousal](https://drive.google.com/file/d/1VrOiBj2t_xwn-MIUPxSYVz_5gUZYhS6F/view?usp=sharing) - [Valence](https://drive.google.com/file/d/1XgqthdUTKYrWy7Vh10MceJVt94KN2DbU/view?usp=sharing)

### How to use :scream:

#### Install requirements

`pip install -r requirements.txt`

#### Data preprocessing

1) Download the data from the [official website](https://mahnob-db.eu/hci-tagging/).
2) Preprocess the data: `python data/preprocessing.py`
   - This will create a folder for each subject with CSV files containing the preprocessed data and save everything inside `args.save_path`.
   
4) Create torch files with augmented and split data: `python data/create_dataset.py`
   - This performs data splitting and augmentation from the preprocessed data in step 2.
   - You can specify which label to consider by setting the parameter `label_kind` to either `Arsl` or `Vlnc`.
   - The data is saved as .pt files which are used for training.

#### Training

To reproduce the results, use the corresponding configuration file for each model and task:

- `configs/h2.yml` → H2 model
- `configs/cross_attention_h2.yml` → H2 with cross-modal attention fusion
- `configs/phemonet.yml` → PHemoNet
- `configs/hyperfusenet_arousal.yml` → HyperFuseNet for valence
- `configs/hyperfusenet_valence.yml` → HyperFuseNet for arousal

Run training with:
```
python main.py --train_file_path /path/to/arsl_or_vlnc_train.pt --test_file_path /path/to/arsl_or_vlnc_test.pt --config configs/config.yml
```

To do a sweep (used in HyperFuseNet paper) run: `python sweep.py`

Experiments will be directly tracked on [Weight&Biases](https://wandb.ai/).

#### Cross-modal attention H2 on AutoDL

`CrossAttentionH2` keeps the four H2 encoders and PHM classifier, but replaces
the concatenation fusion with modality-level cross-attention. Each query
modality can attend only to the other three modalities.

Run it from the repository root:

```bash
python main.py \
  --config configs/cross_attention_h2.yml \
  --train_file_path /path/to/train_augmented_data_Arsl.pt \
  --test_file_path /path/to/test_data_Arsl.pt
```

The expected model inputs are Eye `[B, 4, 600]`, GSR `[B, 1, 1280]`, EEG
`[B, 10, 1280]`, and ECG `[B, 3, 1280]`. Attention settings can be changed in
the YAML file through `attention_dim`, `attention_heads`, `attention_layers`,
and `attention_dropout`.

For diagnostics:

```python
logits, info = model(eye, gsr, eeg, ecg, return_attention=True)
cross_attention = info["cross_attention"]
pooling_weights = info["pooling_weights"]
```

`cross_attention` has shape
`[layer, batch, query_modality, head, other_modality]`, while
`pooling_weights` has shape `[batch, modality]`. The modality order is Eye,
GSR, EEG, and ECG.

The dataset-free checks can be run with:

```bash
python -m unittest tests.test_cross_attention
```

### Cite

Please cite our works if you found this repo useful 🫶

- H2 model:
```
@inproceedings{lopez2024hierarchical,
  title={Hierarchical hypercomplex network for multimodal emotion recognition},
  author={Lopez, Eleonora and Uncini, Aurelio and Comminiello, Danilo},
  booktitle={2024 IEEE 34th International Workshop on Machine Learning for Signal Processing (MLSP)},
  pages={1--6},
  year={2024},
  organization={IEEE}
}
```
- PHemoNet:
```
@inproceedings{lopez2024phemonet,
  title={PHemoNet: A Multimodal Network for Physiological Signals},
  author={Lopez, Eleonora and Uncini, Aurelio and Comminiello, Danilo},
  booktitle={2024 IEEE 8th Forum on Research and Technologies for Society and Industry Innovation (RTSI)},
  pages={260--264},
  year={2024},
  organization={IEEE}
}
```
- HyperFuseNet:
```
@inproceedings{lopez2023hypercomplex,
  title={Hypercomplex Multimodal Emotion Recognition from EEG and Peripheral Physiological Signals},
  author={Lopez, Eleonora and Chiarantano, Eleonora and Grassucci, Eleonora and Comminiello, Danilo},
  booktitle={2023 IEEE International Conference on Acoustics, Speech, and Signal Processing Workshops (ICASSPW)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

### Want more of hypercomplex models? :busts_in_silhouette:

Check out:

* Multi-view hypercomplex learning for breast cancer screening, _under review at TMI_, 2022 [[Paper](https://arxiv.org/abs/2204.05798)][[GitHub](https://github.com/ispamm/PHBreast/)]
* PHNNs: Lightweight neural networks via parameterized hypercomplex convolutions, _IEEE Transactions on Neural Networks and Learning Systems_, 2022 [[Paper](https://ieeexplore.ieee.org/document/9983846)][[GitHub](https://github.com/elegan23/hypernets)].
* Hypercomplex Image-to-Image Translation, _IJCNN_, 2022 [[Paper](https://ieeexplore.ieee.org/document/9892119)][[GitHub](https://github.com/ispamm/HI2I)]
