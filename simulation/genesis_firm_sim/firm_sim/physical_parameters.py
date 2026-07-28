"""Physical parameters for the released FIRM-Sim scenes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalParameters:
    """Genesis-facing parameters plus their physical target values."""

    rigid_dt: float
    rigid_substeps: int
    deformable_dt: float
    deformable_substeps: int
    rigid_constraint_solver: str
    rigid_solver_iterations: int
    rigid_enable_collision: bool
    rigid_enable_self_collision: bool
    pbd_particle_size: float
    pbd_stretch_solver_iterations: int
    pbd_bending_solver_iterations: int
    table_friction: float
    fixture_friction: float
    manual_density: float
    manual_friction: float
    manual_hinge_stiffness: float
    manual_hinge_damping: float
    sponge_areal_density: float
    sponge_static_friction: float
    sponge_kinetic_friction: float
    sponge_stretch_compliance: float
    sponge_bending_compliance: float
    sponge_air_resistance: float
    tape_density: float
    tape_friction: float
    box_equivalent_density: float
    box_friction: float
    box_hinge_stiffness: float
    box_hinge_damping: float
    cable_areal_density: float
    cable_static_friction: float
    cable_kinetic_friction: float
    cable_stretch_compliance: float
    cable_bending_compliance: float
    cable_air_resistance: float
    cable_end_density: float
    cable_end_friction: float
    cable_target_line_density: float
    cable_target_bending_stiffness: float
    cable_target_precurvature: float
    tape_target_rolling_resistance: float


FIRM_PHYSICS = PhysicalParameters(
    rigid_dt=0.01,
    rigid_substeps=2,
    deformable_dt=0.004,
    deformable_substeps=10,
    rigid_constraint_solver="Newton",
    rigid_solver_iterations=100,
    rigid_enable_collision=True,
    rigid_enable_self_collision=False,
    pbd_particle_size=0.005,
    pbd_stretch_solver_iterations=8,
    pbd_bending_solver_iterations=3,
    table_friction=0.4,
    fixture_friction=0.4,
    # Five 0.1 mm, 200 x 150 mm sheets weigh 12 g at this density.
    manual_density=800.0,
    manual_friction=0.35,
    # The five full-size sheets form one bound booklet with no central hinge.
    manual_hinge_stiffness=0.0,
    manual_hinge_damping=0.0,
    # PBD cloth rho is kg/m^2. This gives the 196 x 144 mm pad a 1.30 g mass.
    sponge_areal_density=0.046,
    sponge_static_friction=0.8,
    sponge_kinetic_friction=0.6,
    sponge_stretch_compliance=5e-8,
    sponge_bending_compliance=8e-5,
    sponge_air_resistance=2e-3,
    # The 95/76.2 mm annulus weighs about 32 g at this effective density.
    tape_density=1266.0,
    tape_friction=0.4,
    # Proxy volume times 61 kg/m^3 matches about 350 g/m^2 physical paperboard.
    box_equivalent_density=61.0,
    box_friction=0.35,
    box_hinge_stiffness=0.20,
    box_hinge_damping=7e-3,
    # Rest cloth area is 9e-4 m^2; this produces about 24.6 g over 0.665 m.
    cable_areal_density=27.4,
    cable_static_friction=0.4,
    cable_kinetic_friction=0.3,
    cable_stretch_compliance=1e-8,
    cable_bending_compliance=5e-6,
    cable_air_resistance=4e-3,
    cable_end_density=500.0,
    cable_end_friction=0.4,
    cable_target_line_density=0.037,
    cable_target_bending_stiffness=0.0012,
    cable_target_precurvature=16.7,
    tape_target_rolling_resistance=0.020,
)
