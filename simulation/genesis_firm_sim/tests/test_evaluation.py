import json
import unittest
import numpy as np
from firm_sim.perturbations import generate_conditions, noisy_observation, Condition
from firm_sim.evaluation.metrics import summarize


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.conditions = list(generate_conditions(["tape_manipulation"], seed=7))

    def test_schedule_is_paired_and_isolated(self):
        self.assertEqual(len(self.conditions), 300)
        self.assertEqual(
            [c.id for c in self.conditions],
            [c.id for c in generate_conditions(["tape_manipulation"], seed=7)],
        )
        for c in self.conditions:
            self.assertEqual(c.id, Condition(**json.loads(json.dumps(c.to_dict()))).id)
            self.assertLessEqual(np.linalg.norm(c.object_xy), 0.04 * c.delta + 1e-12)
            self.assertLessEqual(np.linalg.norm(c.fixture_xy), 0.02 * c.delta + 1e-12)
            self.assertLessEqual(abs(c.object_yaw_deg), 20 * c.delta)
            self.assertLessEqual(abs(c.fixture_yaw_deg), 10 * c.delta)
            self.assertLessEqual(np.linalg.norm(c.camera_xyz), 0.01 * c.delta + 1e-12)
            self.assertLessEqual(
                np.linalg.norm(c.camera_rotvec_deg), 5 * c.delta + 1e-12
            )
            if c.family != "object":
                self.assertEqual(np.linalg.norm(c.object_xy), 0)
            if c.family != "fixture":
                self.assertEqual(np.linalg.norm(c.fixture_xy), 0)
            if c.family != "scene" or c.delta == 0:
                self.assertEqual(c.rgb_sigma, 0)
                self.assertEqual(len(c.distractors), 0)

    def test_gaussian_rgb_and_bounded_depth(self):
        c = next(c for c in self.conditions if c.family == "scene" and c.delta == 1)
        image = np.full((300, 300, 3), 0.5, dtype=np.float32)
        depth = np.ones((300, 300), dtype=np.float32)
        depth[0, 0] = np.nan
        a, b = noisy_observation(image, depth, c, 1)
        self.assertAlmostEqual(float(np.std(a - image)), 10 / 255, places=3)
        self.assertLessEqual(float(np.nanmax(abs(b - depth))), 0.005001)
        self.assertTrue(np.isnan(b[0, 0]))
        self.assertTrue(np.array_equal(a, noisy_observation(image, depth, c, 1)[0]))
        nominal = next(
            c for c in self.conditions if c.family == "scene" and c.delta == 0
        )
        self.assertTrue(
            np.array_equal(image, noisy_observation(image, depth, nominal, 1)[0])
        )

    def test_paper_example_and_equal_family_weight(self):
        counts = [18, 16, 14, 10, 6]
        results = [
            {
                "condition_id": c.id,
                "status": "completed",
                "success": c.trial < counts[int(c.delta * 4)],
            }
            for c in self.conditions
        ]
        report = summarize(self.conditions, results)["tape_manipulation"]
        self.assertAlmostEqual(report["pr_auc"], 0.65)
        with self.assertRaises(ValueError):
            summarize(self.conditions, results[:-1])
        with self.assertRaises(ValueError):
            summarize(self.conditions, results + results[:1])
        results[0]["status"] = "error"
        with self.assertRaises(ValueError):
            summarize(self.conditions, results)


class AdapterTest(unittest.TestCase):
    def test_world_transform_and_particle_rotation(self):
        from firm_sim.evaluation.environment import transform_entity

        class Rigid:
            def get_pos(self, *, relative):
                assert relative is False
                return np.array([1.0, 0.0, 0.0])

            def get_quat(self, *, relative):
                assert relative is False
                return np.array([1.0, 0.0, 0.0, 0.0])

            def set_pos(self, value, **kwargs):
                assert kwargs["relative"] is False
                self.position = value

            def set_quat(self, value, **kwargs):
                assert kwargs["relative"] is False
                self.quat = value

        rigid = Rigid()
        transform_entity(rigid, (0.1, 0.2), 90, np.zeros(3))
        np.testing.assert_allclose(rigid.position, [0.1, 1.2, 0.0], atol=1e-12)

        class Particles:
            def get_particles_pos(self):
                return np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])

            def set_particles_pos(self, value):
                self.position = value

            def set_particles_vel(self, value):
                self.velocity = value

        particles = Particles()
        transform_entity(particles, (0.1, 0.2), 90, np.zeros(3))
        np.testing.assert_allclose(
            particles.position, [[0.1, 1.2, 0.0], [0.1, -0.8, 0.0]], atol=1e-12
        )
        np.testing.assert_array_equal(particles.velocity, np.zeros((2, 3)))

    def test_rollout_stops_on_success_and_logs_state(self):
        from unittest.mock import patch
        from firm_sim.evaluation.worker import execute

        class Environment:
            def __init__(self, *args):
                self.steps = 0

            def state(self):
                return {"step": self.steps}

            def observation(self):
                return {}

            def step(self, action):
                self.steps += 1

        class Policy:
            def __init__(self, options):
                pass

            def reset(self, observation):
                pass

            def act(self, observation):
                return None

        class Judge:
            def __init__(self, options):
                pass

            def reset(self, env):
                pass

            def evaluate(self, env):
                return env.steps == 3, False

        condition = next(generate_conditions(["tape_manipulation"]))
        config = {
            "policy": "test:policy",
            "judge": "test:judge",
            "camera": {},
            "max_steps": 10,
        }
        with (
            patch("firm_sim.evaluation.worker.version", return_value="1.2.3"),
            patch(
                "firm_sim.evaluation.worker.load_factory", side_effect=[Policy, Judge]
            ),
            patch("firm_sim.evaluation.environment.GenesisEnvironment", Environment),
        ):
            result = execute(condition, config)
        self.assertTrue(result["success"])
        self.assertEqual(result["steps"], 3)
        self.assertEqual(result["initial_state"], {"step": 0})
        self.assertEqual(result["final_state"], {"step": 3})


class RunnerTest(unittest.TestCase):
    def test_complete_schedule_and_resume(self):
        import contextlib
        import io
        from pathlib import Path
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        from firm_sim.evaluation import cli

        conditions = list(generate_conditions(["tape_manipulation"]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            schedule = root / "schedule.json"
            config = root / "config.json"
            output = root / "results"
            schedule.write_text(json.dumps([c.to_dict() for c in conditions]))
            config.write_text(
                json.dumps(
                    {
                        "method": "test",
                        "policy": "test:policy",
                        "judge": "test:judge",
                        "camera": {},
                        "max_steps": 2,
                    }
                )
            )

            def worker(command, **kwargs):
                request = json.loads(Path(command[3]).read_text())
                condition = Condition(**request["condition"])
                Path(command[4]).write_text(
                    json.dumps(
                        {
                            "condition_id": condition.id,
                            "status": "completed",
                            "success": False,
                        }
                    )
                )
                return SimpleNamespace(returncode=0)

            arguments = [
                "evaluation",
                "run",
                "--schedule",
                str(schedule),
                "--config",
                str(config),
                "--output",
                str(output),
            ]
            with (
                patch("sys.argv", arguments),
                patch.object(cli.platform, "platform", return_value="test-platform"),
                contextlib.redirect_stdout(io.StringIO()),
                patch.object(cli.subprocess, "run", side_effect=worker) as run,
            ):
                cli.main()
                self.assertEqual(run.call_count, 300)
                run.reset_mock()
                cli.main()
                run.assert_not_called()
            self.assertFalse((output / "request.json").exists())
            self.assertEqual(
                json.loads((output / "summary.json").read_text())["tasks"][
                    "tape_manipulation"
                ]["pr_auc"],
                0,
            )


if __name__ == "__main__":
    unittest.main()
