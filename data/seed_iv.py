"""SEED-IV aligned EEG and eye-movement feature loading."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from scipy.io import loadmat


SESSION_LABELS = {
    1: (1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3),
    2: (2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1),
    3: (1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0),
}
CLASS_NAMES = ("neutral", "sad", "fear", "happy")
EEG_FEATURE_KINDS = (
    "de_LDS",
    "de_movingAve",
    "psd_LDS",
    "psd_movingAve",
)
SUBJECT_FOLDS = (
    (1, 2, 3),
    (4, 5, 6),
    (7, 8, 9),
    (10, 11, 12),
    (13, 14, 15),
)


@dataclass(frozen=True)
class SeedIVArrays:
    eeg: np.ndarray
    eye: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    sessions: np.ndarray
    trials: np.ndarray


def get_subject_split(fold: int) -> Dict[str, Tuple[int, ...]]:
    """Return a deterministic 9/3/3 train/validation/test subject split."""
    if not 0 <= fold < len(SUBJECT_FOLDS):
        raise ValueError(f"fold must be between 0 and {len(SUBJECT_FOLDS) - 1}.")

    test_subjects = SUBJECT_FOLDS[fold]
    validation_subjects = SUBJECT_FOLDS[(fold + 1) % len(SUBJECT_FOLDS)]
    excluded = set(test_subjects + validation_subjects)
    train_subjects = tuple(
        subject for subject in range(1, 16) if subject not in excluded
    )
    return {
        "train": train_subjects,
        "validation": validation_subjects,
        "test": test_subjects,
    }


def _subject_id(path: Path) -> int:
    try:
        return int(path.stem.split("_", maxsplit=1)[0])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Cannot parse subject ID from {path.name}.") from error


def load_seed_iv_features(
    dataset_root: str,
    eeg_feature_kind: str = "de_LDS",
) -> SeedIVArrays:
    """Load all aligned four-second EEG and eye feature windows."""
    if eeg_feature_kind not in EEG_FEATURE_KINDS:
        raise ValueError(
            f"eeg_feature_kind must be one of {EEG_FEATURE_KINDS}, "
            f"received {eeg_feature_kind!r}."
        )

    root = Path(dataset_root)
    eeg_root = root / "eeg_feature_smooth"
    eye_root = root / "eye_feature_smooth"
    if not eeg_root.is_dir() or not eye_root.is_dir():
        raise FileNotFoundError(
            f"{root} must contain eeg_feature_smooth and eye_feature_smooth."
        )

    eeg_chunks = []
    eye_chunks = []
    label_chunks = []
    subject_chunks = []
    session_chunks = []
    trial_chunks = []

    for session in range(1, 4):
        eeg_files = sorted(
            (eeg_root / str(session)).glob("*.mat"),
            key=_subject_id,
        )
        if len(eeg_files) != 15:
            raise ValueError(
                f"Session {session} must contain 15 EEG MAT files; "
                f"found {len(eeg_files)}."
            )

        for eeg_path in eeg_files:
            subject = _subject_id(eeg_path)
            eye_path = eye_root / str(session) / eeg_path.name
            if not eye_path.is_file():
                raise FileNotFoundError(
                    f"Missing eye feature file paired with {eeg_path}."
                )

            eeg_names = [
                f"{eeg_feature_kind}{trial}" for trial in range(1, 25)
            ]
            eye_names = [f"eye_{trial}" for trial in range(1, 25)]
            eeg_mat = loadmat(eeg_path, variable_names=eeg_names)
            eye_mat = loadmat(eye_path, variable_names=eye_names)

            for trial, label in enumerate(SESSION_LABELS[session], start=1):
                eeg_name = f"{eeg_feature_kind}{trial}"
                eye_name = f"eye_{trial}"
                if eeg_name not in eeg_mat or eye_name not in eye_mat:
                    raise KeyError(
                        f"Missing {eeg_name} or {eye_name} for subject "
                        f"{subject}, session {session}."
                    )

                # Official shapes are EEG [62, window, 5] and Eye [31, window].
                eeg = np.asarray(eeg_mat[eeg_name], dtype=np.float32).transpose(
                    1, 0, 2
                )
                eye = np.asarray(eye_mat[eye_name], dtype=np.float32).T
                if eeg.ndim != 3 or eeg.shape[1:] != (62, 5):
                    raise ValueError(
                        f"{eeg_path}:{eeg_name} has invalid shape {eeg.shape}."
                    )
                if eye.ndim != 2 or eye.shape[1] != 31:
                    raise ValueError(
                        f"{eye_path}:{eye_name} has invalid shape {eye.shape}."
                    )
                if eeg.shape[0] != eye.shape[0]:
                    raise ValueError(
                        f"Window mismatch for subject {subject}, session "
                        f"{session}, trial {trial}: EEG={eeg.shape[0]}, "
                        f"Eye={eye.shape[0]}."
                    )

                window_count = eeg.shape[0]
                eeg_chunks.append(eeg)
                eye_chunks.append(eye)
                label_chunks.append(
                    np.full(window_count, label, dtype=np.int64)
                )
                subject_chunks.append(
                    np.full(window_count, subject, dtype=np.int16)
                )
                session_chunks.append(
                    np.full(window_count, session, dtype=np.int8)
                )
                trial_chunks.append(
                    np.full(window_count, trial, dtype=np.int8)
                )

    return SeedIVArrays(
        eeg=np.concatenate(eeg_chunks, axis=0),
        eye=np.concatenate(eye_chunks, axis=0),
        labels=np.concatenate(label_chunks),
        subjects=np.concatenate(subject_chunks),
        sessions=np.concatenate(session_chunks),
        trials=np.concatenate(trial_chunks),
    )


def standardize_from_training_subjects(
    arrays: SeedIVArrays,
    train_subjects: Sequence[int],
) -> SeedIVArrays:
    """Z-score features with training subjects only and mean-impute missing eye values."""
    train_mask = np.isin(arrays.subjects, train_subjects)
    if not train_mask.any():
        raise ValueError("No training samples found for the requested subjects.")

    eeg = arrays.eeg.copy()
    eye = arrays.eye.copy()
    eeg[~np.isfinite(eeg)] = np.nan
    eye[~np.isfinite(eye)] = np.nan

    eeg_mean = np.nanmean(eeg[train_mask], axis=0, dtype=np.float64)
    eeg_std = np.nanstd(eeg[train_mask], axis=0, dtype=np.float64)
    eye_mean = np.nanmean(eye[train_mask], axis=0, dtype=np.float64)
    eye_std = np.nanstd(eye[train_mask], axis=0, dtype=np.float64)
    eeg_std = np.where(eeg_std < 1e-6, 1.0, eeg_std)
    eye_std = np.where(eye_std < 1e-6, 1.0, eye_std)

    eeg = ((eeg - eeg_mean) / eeg_std).astype(np.float32)
    eye = ((eye - eye_mean) / eye_std).astype(np.float32)
    # A missing value becomes the training-set mean, which is zero after z-score.
    eeg = np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0)
    eye = np.nan_to_num(eye, nan=0.0, posinf=0.0, neginf=0.0)

    return SeedIVArrays(
        eeg=eeg,
        eye=eye,
        labels=arrays.labels,
        subjects=arrays.subjects,
        sessions=arrays.sessions,
        trials=arrays.trials,
    )


class SeedIVDataset(torch.utils.data.Dataset):
    """A subject-filtered view over aligned SEED-IV feature arrays."""

    def __init__(self, arrays: SeedIVArrays, subjects: Sequence[int]):
        self.arrays = arrays
        self.indices = np.flatnonzero(np.isin(arrays.subjects, subjects))
        if not len(self.indices):
            raise ValueError("The selected subjects contain no samples.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        sample_index = self.indices[index]
        inputs = {
            "eeg": torch.from_numpy(self.arrays.eeg[sample_index]),
            "eye": torch.from_numpy(self.arrays.eye[sample_index]),
        }
        label = int(self.arrays.labels[sample_index])
        return inputs, label


def _class_distribution(dataset: SeedIVDataset) -> np.ndarray:
    labels = dataset.arrays.labels[dataset.indices]
    return np.bincount(labels, minlength=len(CLASS_NAMES))


def build_seed_iv_loaders(
    dataset_root: str,
    fold: int,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    eeg_feature_kind: str = "de_LDS",
    seed: int = 0,
):
    """Build leakage-safe subject-independent loaders for one of five folds."""
    print("----Loading SEED-IV dataset----")
    split = get_subject_split(fold)
    arrays = load_seed_iv_features(dataset_root, eeg_feature_kind)
    arrays = standardize_from_training_subjects(arrays, split["train"])

    train_dataset = SeedIVDataset(arrays, split["train"])
    validation_dataset = SeedIVDataset(arrays, split["validation"])
    test_dataset = SeedIVDataset(arrays, split["test"])

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    train_distribution = _class_distribution(train_dataset)
    inverse_frequency = train_distribution.sum() / train_distribution
    class_weights = inverse_frequency / inverse_frequency.sum()

    print(f"Fold: {fold}")
    print(f"Training subjects: {split['train']}")
    print(f"Validation subjects: {split['validation']}")
    print(f"Test subjects: {split['test']}")
    print(f"Training samples: {len(train_dataset)} {train_distribution.tolist()}")
    print(
        "Validation samples: "
        f"{len(validation_dataset)} "
        f"{_class_distribution(validation_dataset).tolist()}"
    )
    print(
        f"Test samples: {len(test_dataset)} "
        f"{_class_distribution(test_dataset).tolist()}"
    )
    print("--------------------------------")

    return train_loader, validation_loader, test_loader, class_weights
