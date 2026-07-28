import unittest

from firm_sim.physical_parameters import FIRM_PHYSICS
from firm_sim.scenes.workspace import (
    WorkspaceSpec,
    _cable_mouse_spawn_position,
    _generated_box_folding_mjcf_path,
    _generated_cable_connected_mouse_mesh_path,
    _generated_instruction_manual_mjcf_path,
    _generated_tape_annulus_mesh_path,
    table_surface_z,
)


class ScenePhysicsTest(unittest.TestCase):
    def test_manual_is_five_full_size_bound_sheets(self):
        spec = WorkspaceSpec()
        xml = _generated_instruction_manual_mjcf_path(spec, FIRM_PHYSICS).read_text()
        self.assertNotIn('type="hinge"', xml)
        self.assertNotIn('left_sheet_', xml)
        self.assertNotIn('right_sheet_', xml)
        self.assertEqual(xml.count('name="booklet_sheet_'), 5)
        self.assertEqual(xml.count('size="0.100000 0.075000 0.000050"'), 5)

    def test_box_rest_angle_follows_scene_role(self):
        spec = WorkspaceSpec()
        closed_path = _generated_box_folding_mjcf_path(spec, FIRM_PHYSICS, "box")
        outward_path = _generated_box_folding_mjcf_path(spec, FIRM_PHYSICS, "manual")
        closed = closed_path.read_text()
        outward = outward_path.read_text()
        self.assertEqual(closed_path.name, "box_hinged_closed.xml")
        self.assertEqual(outward_path.name, "box_fixed_open.xml")
        self.assertIn('stiffness="0.200000"', closed)
        self.assertIn('damping="0.007000"', closed)
        self.assertIn('springref="0.000000"', closed)
        self.assertNotIn('name="lid_hinge"', outward)
        self.assertIn('euler="2.800000 0 0"', outward)

    def test_only_box_folding_starts_closed(self):
        spec = WorkspaceSpec()
        for task_object in ("manual", "sponge", "tape", "cable"):
            xml = _generated_box_folding_mjcf_path(
                spec, FIRM_PHYSICS, task_object
            ).read_text()
            self.assertNotIn('name="lid_hinge"', xml)
            self.assertIn('euler="2.800000 0 0"', xml)

    def test_task_objects_share_visible_spawn_region_beside_box(self):
        spec = WorkspaceSpec()
        spawn_xy = spec.sponge_center[:2]
        self.assertEqual(spec.manual_center[:2], spawn_xy)
        self.assertEqual(spec.tape_center[:2], spawn_xy)
        self.assertEqual(spawn_xy, (0.55, 0.18))
        self.assertEqual(spec.sponge_center[0], spec.box_fold_proxy_center[0])
        support_front_edge = (
            spec.box_fold_target_center[1] + spec.box_fold_target_size[1] / 2.0
        )
        sponge_collision_back = (
            spec.sponge_center[1]
            - spec.sponge_size[1] / 2.0
            - FIRM_PHYSICS.pbd_particle_size / 2.0
        )
        manual_back = spec.manual_center[1] - spec.manual_size[1] / 2.0
        self.assertGreater(sponge_collision_back, support_front_edge)
        self.assertGreater(manual_back, support_front_edge)
        particle_bottom = spec.sponge_center[2] - FIRM_PHYSICS.pbd_particle_size / 2.0
        self.assertAlmostEqual(particle_bottom, table_surface_z(spec), places=7)

    def test_geometry_matches_documented_objects(self):
        spec = WorkspaceSpec()
        self.assertEqual(spec.manual_size, (0.20, 0.15, 0.0005))
        self.assertEqual(spec.box_fold_base_size, (0.245, 0.18, 0.008))
        self.assertEqual(spec.sponge_size, (0.196, 0.144, 0.001))
        self.assertEqual(spec.tape_inner_radius, 0.0381)

    def test_manual_fits_inside_box_interior(self):
        spec = WorkspaceSpec()
        interior_x = spec.box_fold_base_size[0] - 2.0 * spec.box_fold_wall_thickness
        interior_y = spec.box_fold_base_size[1] - 2.0 * spec.box_fold_wall_thickness
        self.assertLess(spec.manual_size[0], interior_x)
        self.assertLess(spec.manual_size[1], interior_y)
        self.assertAlmostEqual(spec.manual_center[2] - spec.manual_size[2] / 2.0, 0.71)

    def test_tape_mesh_is_a_hollow_annulus(self):
        spec = WorkspaceSpec()
        mesh = _generated_tape_annulus_mesh_path(spec).read_text()
        vertices = [
            tuple(float(value) for value in line.split()[1:4])
            for line in mesh.splitlines()
            if line.startswith("v ")
        ]
        radii = {(round((x * x + y * y) ** 0.5, 4)) for x, y, _ in vertices}
        self.assertEqual(radii, {0.0381, 0.0475})
        self.assertNotIn((0.0, 0.0, 0.0), vertices)

    def test_mouse_local_mesh_starts_on_table(self):
        spec = WorkspaceSpec()
        mesh = _generated_cable_connected_mouse_mesh_path(spec).read_text()
        vertices = [
            tuple(float(value) for value in line.split()[1:4])
            for line in mesh.splitlines()
            if line.startswith("v ")
        ]
        spawn_z = _cable_mouse_spawn_position(spec)[2]
        world_bottom = spawn_z + min(vertex[2] for vertex in vertices)
        self.assertAlmostEqual(world_bottom, table_surface_z(spec), places=6)

if __name__ == "__main__":
    unittest.main()
