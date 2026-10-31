from __future__ import annotations

import unittest

import numpy as np
import torch

from BERN.data import (
    CHANNELS,
    CHANNEL_LOCATIONS,
    FeatureStandardizer,
    LOCATION_ROWS,
    RunArrays,
    SESSION_LABELS,
    eeg_to_grid,
    make_trial_split,
)


class DataProtocolTests(unittest.TestCase):
    def test_all_62_channels_map_to_the_expected_9_by_9_cells(self):
        eeg = torch.arange(62 * 5, dtype=torch.float32).reshape(1, 62, 5) + 1
        grid = eeg_to_grid(eeg)
        self.assertEqual(tuple(grid.shape), (1, 5, 9, 9))
        occupied = set(CHANNEL_LOCATIONS.values())
        self.assertEqual(len(occupied), 62)
        for index, channel in enumerate(CHANNELS):
            row, column = CHANNEL_LOCATIONS[channel]
            torch.testing.assert_close(grid[0, :, row, column], eeg[0, index])
        for row in range(9):
            for column in range(9):
                if (row, column) not in occupied:
                    self.assertTrue(torch.equal(grid[0, :, row, column], torch.zeros(5)))

    def test_location_rows_and_channel_order_are_bijective(self):
        row_channels = tuple(
            channel for row in LOCATION_ROWS for channel in row if channel != "-"
        )
        self.assertEqual(set(row_channels), set(CHANNELS))
        self.assertEqual(len(row_channels), len(CHANNELS))

    def test_validation_is_last_occurring_trial_per_class_and_disjoint(self):
        for session, labels in SESSION_LABELS.items():
            split = make_trial_split(session)
            self.assertEqual(len(split.selection_train), 12)
            self.assertEqual(len(split.validation), 4)
            self.assertEqual(
                sorted(labels[trial - 1] for trial in split.validation),
                [0, 1, 2, 3],
            )
            for trial in split.validation:
                label = labels[trial - 1]
                self.assertEqual(
                    trial,
                    max(i for i in range(1, 17) if labels[i - 1] == label),
                )
            self.assertFalse(set(split.selection_train) & set(split.validation))
            self.assertFalse(set(split.final_train) & set(split.test))

    def test_standardizer_uses_only_the_arrays_passed_to_fit(self):
        eeg = np.zeros((4, 62, 5), dtype=np.float32)
        eye = np.zeros((4, 31), dtype=np.float32)
        eeg[:2] = 2
        eye[:2] = 3
        eeg[2:] = 1000
        eye[2:] = 1000
        arrays = RunArrays(
            eeg=eeg,
            eye=eye,
            labels=np.zeros(4, dtype=np.int64),
            trials=np.array([1, 1, 17, 17], dtype=np.int8),
            window_in_trial=np.array([0, 1, 0, 1], dtype=np.int16),
            subject=1,
            session=1,
        )
        scaler = FeatureStandardizer.fit(arrays.subset([1]))
        self.assertTrue(np.allclose(scaler.eeg_mean, 2))
        self.assertTrue(np.allclose(scaler.eye_mean, 3))
        train = scaler.transform(arrays.subset([1]))
        self.assertTrue(np.allclose(train.eeg, 0))
        self.assertTrue(np.allclose(train.eye, 0))


if __name__ == "__main__":
    unittest.main()
