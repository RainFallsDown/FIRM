import math
import unittest

from firm_sim.physical_parameters import FIRM_PHYSICS


class PhysicalParametersTest(unittest.TestCase):
    def test_sponge_areal_density_matches_target_mass(self):
        area = 0.196 * 0.144
        mass = area * FIRM_PHYSICS.sponge_areal_density
        self.assertAlmostEqual(mass, 0.00130, delta=2e-5)

    def test_tape_annulus_density_matches_target_mass(self):
        volume = math.pi * (0.0475**2 - 0.0381**2) * 0.01
        mass = volume * FIRM_PHYSICS.tape_density
        self.assertAlmostEqual(mass, 0.032, delta=3e-4)

    def test_manual_density_matches_five_sheet_mass(self):
        volume = 0.20 * 0.15 * 0.0005
        mass = volume * FIRM_PHYSICS.manual_density
        self.assertAlmostEqual(mass, 0.012, delta=1e-6)

    def test_solver_configuration_is_explicit(self):
        self.assertEqual(FIRM_PHYSICS.rigid_constraint_solver, "Newton")
        self.assertEqual(FIRM_PHYSICS.rigid_dt, 0.01)
        self.assertEqual(FIRM_PHYSICS.rigid_substeps, 2)
        self.assertEqual(FIRM_PHYSICS.deformable_dt, 0.004)
        self.assertEqual(FIRM_PHYSICS.deformable_substeps, 10)
        self.assertEqual(FIRM_PHYSICS.rigid_solver_iterations, 100)
        self.assertTrue(FIRM_PHYSICS.rigid_enable_collision)
        self.assertFalse(FIRM_PHYSICS.rigid_enable_self_collision)

    def test_cable_profile_keeps_physical_targets_explicit(self):
        self.assertAlmostEqual(FIRM_PHYSICS.cable_target_line_density, 0.037)
        self.assertAlmostEqual(FIRM_PHYSICS.cable_target_bending_stiffness, 0.0012)
        self.assertAlmostEqual(FIRM_PHYSICS.cable_target_precurvature, 16.7)


if __name__ == "__main__":
    unittest.main()
