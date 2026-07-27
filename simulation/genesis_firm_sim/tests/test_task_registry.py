import unittest

from firm_sim.tasks import get_task_spec, task_names


class TaskRegistryTest(unittest.TestCase):
    def test_expected_task_scenes_exist(self):
        self.assertEqual(
            task_names(),
            [
                "box_folding",
                "cable_manipulation",
                "instruction_manual",
                "sponge_pad",
                "tape_manipulation",
            ],
        )

    def test_instruction_manual_metadata(self):
        spec = get_task_spec("instruction_manual")
        self.assertEqual(spec.object_class, "five-layer hinged paper proxy")
        self.assertEqual(spec.scene_name, "instruction_manual")

    def test_tape_metadata(self):
        spec = get_task_spec("tape_manipulation")
        self.assertEqual(spec.object_class, "rigid annulus proxy")
        self.assertEqual(spec.scene_name, "tape_manipulation")

    def test_cable_metadata(self):
        spec = get_task_spec("cable_manipulation")
        self.assertEqual(spec.object_class, "bundled cable + rigid mouse proxy")
        self.assertEqual(spec.scene_name, "cable_manipulation")

    def test_box_folding_metadata(self):
        spec = get_task_spec("box_folding")
        self.assertEqual(spec.object_class, "cardboard box proxy")
        self.assertEqual(spec.scene_name, "box_folding")


if __name__ == "__main__":
    unittest.main()
