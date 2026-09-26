"""Boundary and statistics checks; also runnable with unittest discovery."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from datasets import Dataset

from lerobot.datasets.dataset_reader import DatasetReader
from lerobot.scripts.action_chunk_stats import compute_stats, fit_per_timestamp, iter_action_chunks


class ActionChunkTests(unittest.TestCase):
    def setUp(self):
        self.cfg = SimpleNamespace(
            policy=SimpleNamespace(action_delta_indices=[0, 1, 2], use_relative_actions=False),
            reward_model=None,
            rename_map={},
        )
        self.dataset = SimpleNamespace(
            features={"action": {"names": ["joint", "gripper"]}, "observation.state": {}},
            hf_dataset=Dataset.from_dict({
                "action": [[10.0, 1.0], [20.0, 2.0], [30.0, 3.0], [100.0, 9.0]],
                "observation.state": [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
            }),
            absolute_to_relative_idx=None,
            episodes=None,
            meta=SimpleNamespace(total_episodes=2, episodes=[
                {"dataset_from_index": 0, "dataset_to_index": 3},
                {"dataset_from_index": 3, "dataset_to_index": 4},
            ]),
        )

    def test_boundaries_match_lerobot_reader(self):
        self.cfg.policy.action_delta_indices = [-1, 0, 2, 4]
        reader = object.__new__(DatasetReader)
        reader._meta = self.dataset.meta
        reader.delta_indices = {"action": self.cfg.policy.action_delta_indices}
        for index, (chunk, padding) in enumerate(iter_action_chunks(self.cfg, self.dataset)):
            queries, masks = reader._get_query_indices(index, 0 if index < 3 else 1)
            expected = torch.tensor(self.dataset.hf_dataset[queries["action"]]["action"])
            torch.testing.assert_close(chunk, expected)
            torch.testing.assert_close(padding, masks["action_is_pad"])

    def test_drop_starts_preserves_episode_tail(self):
        self.cfg.policy.drop_n_last_frames = 1
        chunks = list(iter_action_chunks(self.cfg, self.dataset))
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[-1][0][:, 0].tolist(), [20, 30, 30])
        self.assertEqual(chunks[-1][1].tolist(), [False, False, True])

    def test_filtered_episode_uses_absolute_index_mapping(self):
        self.dataset.episodes = [1]
        self.dataset.hf_dataset = self.dataset.hf_dataset.select([3])
        self.dataset.absolute_to_relative_idx = {3: 0}
        chunks = list(iter_action_chunks(self.cfg, self.dataset))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][0][:, 0].tolist(), [100, 100, 100])
        self.assertEqual(chunks[0][1].tolist(), [False, True, True])

    def test_relative_actions_use_anchor_and_preserve_gripper(self):
        self.cfg.policy.use_relative_actions = True
        self.cfg.policy.relative_exclude_joints = ["gripper"]
        chunks = list(iter_action_chunks(self.cfg, self.dataset))
        self.assertEqual(chunks[1][0].tolist(), [[18, 2], [28, 3], [28, 3]])

    def test_masked_stats_and_empty_horizon(self):
        self.cfg.policy.action_delta_indices = [0, 1, 2, 3]
        stats = compute_stats(iter_action_chunks(self.cfg, self.dataset))
        self.assertEqual(stats["count"].tolist(), [4, 2, 1, 0])
        self.assertEqual(stats["mean"][:3, 0].tolist(), [40, 25, 30])
        self.assertEqual(stats["std"][1:3, 0].tolist(), [5, 0])
        self.assertTrue(np.isnan(stats["mean"][3]).all())
        self.assertTrue(np.isnan(stats["per_timestamp_mean"][3]).all())
        self.assertTrue(np.isnan(stats["per_timestamp_std"][3]).all())
        np.testing.assert_allclose(stats["q01"][:3, 0], [10.3, 20.1, 30])
        np.testing.assert_allclose(stats["q99"][:3, 0], [97.9, 29.9, 30])
        ratio = stats["per_timestamp_std"] / np.maximum(stats["std"], 1e-8)
        for key in ("q01", "q99"):
            expected = stats["per_timestamp_mean"] + (stats[key] - stats["mean"]) * ratio
            np.testing.assert_allclose(stats[f"per_timestamp_{key}"], expected)

    def test_smooth_fits_recover_known_curves(self):
        t = np.arange(30, dtype=np.float64)[:, None]
        mean = 2 + 0.5 * t
        std = 0.2 + 1.3 * np.sqrt(t + 0.4)
        fitted_mean, fitted_std = fit_per_timestamp(mean, std)
        np.testing.assert_allclose(fitted_mean, mean, atol=1e-10)
        np.testing.assert_allclose(fitted_std, std, atol=1e-4)

    def test_constant_dimension_and_single_position(self):
        stats = compute_stats(iter([(torch.tensor([[7.0, 0.0]]), torch.tensor([False]))]))
        np.testing.assert_allclose(stats["per_timestamp_mean"], [[7, 0]])
        np.testing.assert_allclose(stats["per_timestamp_std"], [[1e-6, 1e-6]])
        np.testing.assert_allclose(stats["per_timestamp_q01"], [[7, 0]])
        np.testing.assert_allclose(stats["per_timestamp_q99"], [[7, 0]])

    def test_padding_values_do_not_affect_statistics(self):
        chunks = [
            (torch.tensor([[1.0], [float("nan")]]), torch.tensor([False, True])),
            (torch.tensor([[3.0], [5.0]]), torch.tensor([False, False])),
        ]
        stats = compute_stats(iter(chunks))
        np.testing.assert_allclose(stats["mean"], [[2], [5]])
        self.assertEqual(stats["count"].tolist(), [2, 1])

    def test_all_starts_dropped(self):
        self.cfg.policy.drop_n_last_frames = 3
        with self.assertRaisesRegex(ValueError, "No eligible"):
            list(iter_action_chunks(self.cfg, self.dataset))


if __name__ == "__main__":
    unittest.main()
