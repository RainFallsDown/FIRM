import unittest

import numpy as np

from firm_sim.perturbations import (
    apply_depth_noise,
    apply_rgb_noise,
    get_perturbation_level,
    perturbation_axes,
    perturbation_levels,
    sample_perturbation,
)


class PerturbationsTest(unittest.TestCase):
    def test_disclosed_levels_are_available(self):
        self.assertEqual(
            perturbation_levels(),
            ("nominal", "low", "medium", "medium_high", "high"),
        )
        medium_high = get_perturbation_level("medium_high")
        self.assertEqual(medium_high.object_translation_m, 0.030)
        self.assertEqual(medium_high.fixture_translation_m, 0.015)
        self.assertEqual(medium_high.object_yaw_deg, 15.0)
        self.assertEqual(medium_high.pose_position_noise_m, 0.0075)
        self.assertEqual(medium_high.pose_rotation_noise_deg, 3.75)
        self.assertEqual(medium_high.rgb_noise, 7.5 / 255.0)
        self.assertEqual(medium_high.depth_noise_m, 0.00375)
        high = get_perturbation_level("high")
        self.assertEqual(high.object_translation_m, 0.040)
        self.assertEqual(high.fixture_translation_m, 0.020)
        self.assertEqual(high.object_yaw_deg, 20.0)
        self.assertEqual(high.pose_position_noise_m, 0.010)
        self.assertEqual(high.pose_rotation_noise_deg, 5.0)
        self.assertEqual(high.rgb_noise, 10.0 / 255.0)
        self.assertEqual(high.depth_noise_m, 0.005)

    def test_axis_isolation_and_seed_replay(self):
        first = sample_perturbation("medium", "object_translation", seed=17)
        second = sample_perturbation("medium", "object_translation", seed=17)
        self.assertEqual(first, second)
        self.assertAlmostEqual(np.linalg.norm(first.object_translation_xy_m), 0.020)
        self.assertEqual(first.fixture_translation_xy_m, (0.0, 0.0))
        self.assertEqual(first.object_yaw_deg, 0.0)
        self.assertEqual(first.rgb_noise, 0.0)

    def test_combined_enables_every_dimension(self):
        sample = sample_perturbation("high", "combined", seed=3)
        self.assertAlmostEqual(np.linalg.norm(sample.object_translation_xy_m), 0.040)
        self.assertAlmostEqual(np.linalg.norm(sample.fixture_translation_xy_m), 0.020)
        self.assertEqual(abs(sample.object_yaw_deg), 20.0)
        self.assertGreater(sample.pose_position_noise_m, 0.0)
        self.assertGreater(sample.rgb_noise, 0.0)
        self.assertGreater(sample.depth_noise_m, 0.0)

    def test_perception_noise_is_bounded_and_reproducible(self):
        rgb = np.full((12, 10, 3), 128, dtype=np.uint8)
        noisy_rgb = apply_rgb_noise(rgb, 10.0 / 255.0, seed=9)
        self.assertTrue(np.array_equal(noisy_rgb, apply_rgb_noise(rgb, 10.0 / 255.0, seed=9)))
        self.assertLessEqual(np.abs(noisy_rgb.astype(int) - rgb.astype(int)).max(), 10)

        depth = np.full((12, 10), 0.75, dtype=np.float32)
        noisy_depth = apply_depth_noise(depth, 0.005, seed=9)
        self.assertLessEqual(np.abs(noisy_depth - depth).max(), 0.005001)
        self.assertGreaterEqual(noisy_depth.min(), 0.0)

    def test_invalid_axis_is_rejected(self):
        self.assertIn("combined", perturbation_axes())
        with self.assertRaisesRegex(KeyError, "Unknown perturbation axis"):
            sample_perturbation("low", "everything", seed=0)


if __name__ == "__main__":
    unittest.main()
