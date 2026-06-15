import unittest
from pathlib import Path

from firm_sim.exceptions import PlaceholderTaskError


class PipelineTest(unittest.TestCase):
    def test_env_connects_and_closes_headless(self):
        try:
            from firm_sim.env import BenchmarkEnv, BenchmarkEnvConfig
        except ModuleNotFoundError as exc:
            self.skipTest(f"PyBullet is not available in this interpreter: {exc}")

        env = BenchmarkEnv(BenchmarkEnvConfig(gui=False))
        env.connect()
        self.assertTrue(env.connected)
        env.close()
        self.assertFalse(env.connected)

    def test_evaluator_handles_placeholder_task(self):
        try:
            from firm_sim.evaluation import Evaluator
        except ModuleNotFoundError as exc:
            self.skipTest(f"PyBullet is not available in this interpreter: {exc}")

        evaluator = Evaluator(output_dir=Path("outputs/test_eval"))
        with self.assertRaises(PlaceholderTaskError):
            evaluator.evaluate(
                task_name="tape_manipulation",
                policy_name="no_op",
                episodes=1,
                seed=0,
            )

    def test_manual_insertion_scene_resets(self):
        try:
            from firm_sim.env import BenchmarkEnv, BenchmarkEnvConfig
            from firm_sim.registry import make_task
        except ModuleNotFoundError as exc:
            self.skipTest(f"PyBullet is not available in this interpreter: {exc}")

        env = BenchmarkEnv(BenchmarkEnvConfig(gui=False))
        task = make_task("instruction_manual_insertion")
        observation = env.reset(task)
        self.assertEqual(observation["task"], "instruction_manual_insertion")
        self.assertIn("manual", task.asset_ids)
        self.assertIn("receptacle_floor", task.asset_ids)
        env.close()


if __name__ == "__main__":
    unittest.main()
