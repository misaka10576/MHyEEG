"""Leakage-safe, subject/session-specific SEED-IV feature loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy.io import loadmat


SESSION_LABELS: Mapping[int, tuple[int, ...]] = {
    1: (1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3),
    2: (2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1),
    3: (1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0),
}
CLASS_NAMES = ("neutral", "sad", "fear", "happy")
EEG_FEATURE_KINDS = ("de_LDS", "de_movingAve")

# The order and grid agree with the local TorchEEG SEED-IV constants and the
# local LibEER SEED constants. Keep these values local so BERN has no runtime
# dependency on either project.
CHANNELS = (
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ",
    "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4",
    "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ",
    "O2", "CB2",
)
LOCATION_ROWS = (
    ("-", "-", "-", "FP1", "FPZ", "FP2", "-", "-", "-"),
    ("-", "-", "-", "AF3", "-", "AF4", "-", "-", "-"),
    ("F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8"),
    ("FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8"),
    ("T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8"),
    ("TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8"),
    ("P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8"),
    ("-", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "-"),
    ("-", "-", "CB1", "O1", "OZ", "O2", "CB2", "-", "-"),
)
CHANNEL_LOCATIONS = {
    channel: (row, column)
    for row, values in enumerate(LOCATION_ROWS)
    for column, channel in enumerate(values)
    if channel != "-"
}


@dataclass(frozen=True)
class RunArrays:
    """All aligned windows for one subject and one session."""

    eeg: np.ndarray
    eye: np.ndarray
    labels: np.ndarray
    trials: np.ndarray
    window_in_trial: np.ndarray
    subject: int
    session: int

    def subset(self, trial_ids: Sequence[int]) -> "RunArrays":
        mask = np.isin(self.trials, np.asarray(trial_ids))
        return RunArrays(
            eeg=self.eeg[mask],
            eye=self.eye[mask],
            labels=self.labels[mask],
            trials=self.trials[mask],
            window_in_trial=self.window_in_trial[mask],
            subject=self.subject,
            session=self.session,
        )

    def __len__(self) -> int:
        return int(self.labels.shape[0])


@dataclass(frozen=True)
class TrialSplit:
    selection_train: tuple[int, ...]
    validation: tuple[int, ...]
    final_train: tuple[int, ...]
    test: tuple[int, ...]

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "selection_train": list(self.selection_train),
            "validation": list(self.validation),
            "final_train": list(self.final_train),
            "test": list(self.test),
        }


@dataclass
class FeatureStandardizer:
    eeg_mean: np.ndarray
    eeg_std: np.ndarray
    eye_mean: np.ndarray
    eye_std: np.ndarray

    @classmethod
    def fit(cls, arrays: RunArrays) -> "FeatureStandardizer":
        if not len(arrays):
            raise ValueError("Cannot fit a standardizer on an empty training set.")
        eeg = np.where(np.isfinite(arrays.eeg), arrays.eeg, np.nan)
        eye = np.where(np.isfinite(arrays.eye), arrays.eye, np.nan)
        eeg_mean = np.nanmean(eeg, axis=0, dtype=np.float64)
        eeg_std = np.nanstd(eeg, axis=0, dtype=np.float64)
        eye_mean = np.nanmean(eye, axis=0, dtype=np.float64)
        eye_std = np.nanstd(eye, axis=0, dtype=np.float64)
        eeg_mean = np.nan_to_num(eeg_mean, nan=0.0)
        eye_mean = np.nan_to_num(eye_mean, nan=0.0)
        eeg_std = np.nan_to_num(eeg_std, nan=1.0)
        eye_std = np.nan_to_num(eye_std, nan=1.0)
        eeg_std[eeg_std < 1e-6] = 1.0
        eye_std[eye_std < 1e-6] = 1.0
        return cls(eeg_mean, eeg_std, eye_mean, eye_std)

    def transform(self, arrays: RunArrays) -> RunArrays:
        eeg = (arrays.eeg.astype(np.float64) - self.eeg_mean) / self.eeg_std
        eye = (arrays.eye.astype(np.float64) - self.eye_mean) / self.eye_std
        return RunArrays(
            eeg=np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
            eye=np.nan_to_num(eye, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
            labels=arrays.labels,
            trials=arrays.trials,
            window_in_trial=arrays.window_in_trial,
            subject=arrays.subject,
            session=arrays.session,
        )

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "eeg_mean": self.eeg_mean,
            "eeg_std": self.eeg_std,
            "eye_mean": self.eye_mean,
            "eye_std": self.eye_std,
        }


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, arrays: RunArrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return len(self.arrays)

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.arrays.eeg[index]),
            torch.from_numpy(self.arrays.eye[index]),
            torch.tensor(int(self.arrays.labels[index]), dtype=torch.long),
            torch.tensor(int(self.arrays.trials[index]), dtype=torch.int16),
            torch.tensor(int(self.arrays.window_in_trial[index]), dtype=torch.int16),
        )


def subject_mat_path(dataset_root: str | Path, modality: str, subject: int, session: int) -> Path:
    if subject not in range(1, 16):
        raise ValueError("subject must be in 1..15")
    if session not in SESSION_LABELS:
        raise ValueError("session must be in 1..3")
    folder = Path(dataset_root) / f"{modality}_feature_smooth" / str(session)
    matches = sorted(folder.glob(f"{subject}_*.mat"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one subject {subject} MAT file in {folder}, found {len(matches)}."
        )
    return matches[0]


def load_run(
    dataset_root: str | Path,
    subject: int,
    session: int,
    eeg_feature_kind: str = "de_LDS",
) -> RunArrays:
    """Load a single subject/session without changing source data."""
    if eeg_feature_kind not in EEG_FEATURE_KINDS:
        raise ValueError(f"eeg_feature_kind must be one of {EEG_FEATURE_KINDS}")
    eeg_path = subject_mat_path(dataset_root, "eeg", subject, session)
    eye_path = subject_mat_path(dataset_root, "eye", subject, session)
    if eeg_path.name != eye_path.name:
        raise ValueError(f"EEG/eye file names do not pair: {eeg_path.name}, {eye_path.name}")

    eeg_names = [f"{eeg_feature_kind}{trial}" for trial in range(1, 25)]
    eye_names = [f"eye_{trial}" for trial in range(1, 25)]
    eeg_mat = loadmat(eeg_path, variable_names=eeg_names)
    eye_mat = loadmat(eye_path, variable_names=eye_names)
    eeg_chunks: list[np.ndarray] = []
    eye_chunks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    trials: list[np.ndarray] = []
    windows: list[np.ndarray] = []
    for trial, label in enumerate(SESSION_LABELS[session], start=1):
        eeg_name, eye_name = f"{eeg_feature_kind}{trial}", f"eye_{trial}"
        if eeg_name not in eeg_mat or eye_name not in eye_mat:
            raise KeyError(f"Missing paired variables {eeg_name}/{eye_name}")
        eeg = np.asarray(eeg_mat[eeg_name], dtype=np.float32).transpose(1, 0, 2)
        eye = np.asarray(eye_mat[eye_name], dtype=np.float32).T
        if eeg.ndim != 3 or eeg.shape[1:] != (62, 5):
            raise ValueError(f"{eeg_path}:{eeg_name} has shape {eeg.shape}, expected [N,62,5]")
        if eye.ndim != 2 or eye.shape[1] != 31 or eye.shape[0] != eeg.shape[0]:
            raise ValueError(f"{eye_path}:{eye_name} is not aligned with EEG: {eye.shape}")
        count = eeg.shape[0]
        eeg_chunks.append(eeg)
        eye_chunks.append(eye)
        labels.append(np.full(count, label, dtype=np.int64))
        trials.append(np.full(count, trial, dtype=np.int8))
        windows.append(np.arange(count, dtype=np.int16))
    return RunArrays(
        eeg=np.concatenate(eeg_chunks),
        eye=np.concatenate(eye_chunks),
        labels=np.concatenate(labels),
        trials=np.concatenate(trials),
        window_in_trial=np.concatenate(windows),
        subject=subject,
        session=session,
    )


def make_trial_split(session: int) -> TrialSplit:
    """Use one last-occurring complete training trial per class for validation."""
    labels = SESSION_LABELS[session]
    first_sixteen = tuple(range(1, 17))
    validation = tuple(
        max(trial for trial in first_sixteen if labels[trial - 1] == label)
        for label in range(4)
    )
    selection_train = tuple(trial for trial in first_sixteen if trial not in validation)
    split = TrialSplit(
        selection_train=selection_train,
        validation=tuple(sorted(validation)),
        final_train=first_sixteen,
        test=tuple(range(17, 25)),
    )
    validate_trial_split(split, session)
    return split


def validate_trial_split(split: TrialSplit, session: int) -> None:
    sets = {
        "selection_train": set(split.selection_train),
        "validation": set(split.validation),
        "test": set(split.test),
    }
    if len(split.selection_train) != 12 or len(split.validation) != 4:
        raise ValueError("Selection split must contain 12 train and 4 validation trials.")
    if set(split.final_train) != set(range(1, 17)) or sets["test"] != set(range(17, 25)):
        raise ValueError("Final train/test trials must be exactly 1..16/17..24.")
    if any(sets[a] & sets[b] for a, b in (("selection_train", "validation"), ("selection_train", "test"), ("validation", "test"))):
        raise ValueError("Train, validation and test trials overlap.")
    validation_labels = [SESSION_LABELS[session][trial - 1] for trial in split.validation]
    if sorted(validation_labels) != list(range(4)):
        raise ValueError("Validation must contain exactly one complete trial from each class.")


def eeg_to_grid(eeg: torch.Tensor) -> torch.Tensor:
    """Map ``[..., 62, 5]`` features to ``[..., 5, 9, 9]`` with zero gaps."""
    if eeg.shape[-2:] != (62, 5):
        raise ValueError(f"Expected EEG shape [...,62,5], received {tuple(eeg.shape)}")
    output = eeg.new_zeros((*eeg.shape[:-2], 5, 9, 9))
    for channel_index, channel in enumerate(CHANNELS):
        row, column = CHANNEL_LOCATIONS[channel]
        output[..., :, row, column] = eeg[..., channel_index, :]
    return output


def make_loader(
    arrays: RunArrays,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        WindowDataset(arrays),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
