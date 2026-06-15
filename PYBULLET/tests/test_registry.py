import unittest

from firm_sim.registry import make_task, task_names


class RegistryTest(unittest.TestCase):
    def test_all_firm_families_registered(self):
        self.assertEqual(
            task_names(),
            [
                "box_folding",
                "cable_manipulation",
                "instruction_manual_insertion",
                "sponge_pad_placement",
                "tape_manipulation",
            ],
        )

    def test_make_task_returns_placeholder_tasks(self):
        task = make_task("cable_manipulation")
        self.assertTrue(task.placeholder)
        self.assertEqual(task.name, "cable_manipulation")


if __name__ == "__main__":
    unittest.main()
