"""Reusable Genesis workspace scaffold and task-specific scene builders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile

import genesis as gs
import numpy as np

from firm_sim.physical_parameters import FIRM_PHYSICS, PhysicalParameters
from firm_sim.perturbations import PerturbationSample
from firm_sim.runtime import init_genesis


@dataclass(frozen=True)
class WorkspaceSpec:
    table_size: tuple[float, float, float] = (1.00, 0.54, 0.71)
    table_center: tuple[float, float, float] = (0.55, 0.0, 0.355)
    tabletop_thickness: float = 0.04
    fixture_size: tuple[float, float, float] = (0.245, 0.18, 0.003)
    fixture_wall_thickness: float = 0.003
    fixture_wall_height: float = 0.05
    fixture_center: tuple[float, float, float] = (0.55, 0.0, 0.7115)
    target_size: tuple[float, float, float] = (0.20, 0.15, 0.0015)
    target_center: tuple[float, float, float] = (0.55, 0.0, 0.71375)
    manual_size: tuple[float, float, float] = (0.20, 0.15, 0.0005)
    manual_center: tuple[float, float, float] = (0.55, 0.18, 0.71025)
    manual_fold_angle: float = 0.0
    sponge_size: tuple[float, float, float] = (0.196, 0.144, 0.001)
    sponge_scale: tuple[float, float, float] = (0.392, 0.288, 1.0)
    sponge_center: tuple[float, float, float] = (0.55, 0.18, 0.7125)
    tape_radius: float = 0.0475
    tape_inner_radius: float = 0.0381
    tape_height: float = 0.01
    tape_center: tuple[float, float, float] = (0.55, 0.18, 0.715)
    cable_connected_line_scale: tuple[float, float, float] = (0.30, 0.012, 1.0)
    cable_connected_line_center: tuple[float, float, float] = (0.75, -0.05, 0.713)
    cable_mouse_body_size: tuple[float, float, float] = (0.05, 0.085, 0.028)
    cable_mouse_body_center: tuple[float, float, float] = (0.81, -0.095, 0.724)
    cable_mouse_top_size: tuple[float, float, float] = (0.04, 0.06, 0.02)
    cable_mouse_top_center: tuple[float, float, float] = (0.81, -0.093, 0.742)
    cable_mouse_nose_size: tuple[float, float, float] = (0.022, 0.02, 0.016)
    cable_mouse_nose_center: tuple[float, float, float] = (0.81, -0.13, 0.726)
    box_fold_base_size: tuple[float, float, float] = (0.245, 0.18, 0.008)
    box_fold_wall_height: float = 0.05
    box_fold_wall_thickness: float = 0.004
    box_fold_lid_thickness: float = 0.006
    box_fold_lid_closed_angle: float = 0.0
    box_fold_lid_open_angle_outward: float = 2.80
    box_fold_proxy_center: tuple[float, float, float] = (0.55, 0.0, 0.714)
    box_fold_target_size: tuple[float, float, float] = (0.255, 0.19, 0.0015)
    box_fold_target_center: tuple[float, float, float] = (0.55, 0.0, 0.7145)
    tianqing_robot_pos: tuple[float, float, float] = (0.55, 0.43, 0.0)
    tianqing_robot_euler: tuple[float, float, float] = (0.0, 0.0, -90.0)

def build_instruction_manual_scene(
    show_viewer: bool = True,
    camera_specs: dict[str, object] | None = None,
    perturbation: PerturbationSample | None = None,
) -> tuple[gs.Scene, dict[str, object]]:
    """Build the instruction-manual insertion scene."""
    return _build_workspace(
        show_viewer=show_viewer,
        task_object="manual",
        camera_specs=camera_specs,
        perturbation=perturbation,
    )


def build_sponge_pad_scene(
    show_viewer: bool = True,
    camera_specs: dict[str, object] | None = None,
    perturbation: PerturbationSample | None = None,
) -> tuple[gs.Scene, dict[str, object]]:
    """Build the sponge-pad placement scene using the shared workspace."""
    return _build_workspace(
        show_viewer=show_viewer,
        task_object="sponge",
        camera_specs=camera_specs,
        perturbation=perturbation,
    )


def build_tape_scene(
    show_viewer: bool = True,
    camera_specs: dict[str, object] | None = None,
    perturbation: PerturbationSample | None = None,
) -> tuple[gs.Scene, dict[str, object]]:
    """Build the tape-manipulation scene using the shared workspace."""
    return _build_workspace(
        show_viewer=show_viewer,
        task_object="tape",
        camera_specs=camera_specs,
        perturbation=perturbation,
    )


def build_cable_scene(
    show_viewer: bool = True,
    camera_specs: dict[str, object] | None = None,
    perturbation: PerturbationSample | None = None,
) -> tuple[gs.Scene, dict[str, object]]:
    """Build the cable-manipulation scene using the shared workspace."""
    return _build_workspace(
        show_viewer=show_viewer,
        task_object="cable",
        camera_specs=camera_specs,
        perturbation=perturbation,
    )


def build_box_folding_scene(
    show_viewer: bool = True,
    camera_specs: dict[str, object] | None = None,
    perturbation: PerturbationSample | None = None,
) -> tuple[gs.Scene, dict[str, object]]:
    """Build the box-folding scene using the shared workspace."""
    return _build_workspace(
        show_viewer=show_viewer,
        task_object="box",
        camera_specs=camera_specs,
        perturbation=perturbation,
    )


def _build_workspace(
    show_viewer: bool,
    task_object: str,
    camera_specs: dict[str, object] | None = None,
    physics_profile: PhysicalParameters = FIRM_PHYSICS,
    perturbation: PerturbationSample | None = None,
) -> tuple[gs.Scene, dict[str, object]]:
    """Build the shared workspace with exactly one task object."""
    init_genesis()

    include_deformable = task_object in {"sponge", "cable"}
    sim_options = gs.options.SimOptions(
        dt=physics_profile.deformable_dt if include_deformable else physics_profile.rigid_dt,
        substeps=(
            physics_profile.deformable_substeps
            if include_deformable
            else physics_profile.rigid_substeps
        ),
    )
    pbd_options = (
        gs.options.PBDOptions(
            particle_size=physics_profile.pbd_particle_size,
            max_stretch_solver_iterations=physics_profile.pbd_stretch_solver_iterations,
            max_bending_solver_iterations=physics_profile.pbd_bending_solver_iterations,
            lower_bound=(0.0, -0.5, 0.0),
            upper_bound=(1.1, 0.5, 1.3),
        )
        if include_deformable
        else None
    )

    scene = gs.Scene(
        sim_options=sim_options,
        rigid_options=gs.options.RigidOptions(
            enable_collision=physics_profile.rigid_enable_collision,
            enable_self_collision=physics_profile.rigid_enable_self_collision,
            constraint_solver=getattr(gs.constraint_solver, physics_profile.rigid_constraint_solver),
            iterations=physics_profile.rigid_solver_iterations,
        ),
        pbd_options=pbd_options,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.6, -1.8, 1.9),
            camera_lookat=(0.55, 0.0, 0.55),
            camera_fov=42,
            refresh_rate=60,
            res=(1600, 1000),
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=show_viewer,
    )

    spec = WorkspaceSpec()
    entities: dict[str, object] = {}

    entities["floor"] = scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(friction=1.0),
        surface=gs.surfaces.Default(color=(0.9, 0.92, 0.95)),
        name="floor",
    )

    _add_table(scene, spec, entities, physics_profile)
    uses_hinged_box = task_object in {"none", "manual", "sponge", "tape", "cable", "box"}
    if uses_hinged_box:
        _add_box_fold_target_zone(scene, spec, entities)
        _add_box_fold_support(scene, spec, entities, physics_profile)
        _add_box_folding_proxy(scene, spec, entities, physics_profile, task_object)
    else:
        _add_target_zone(scene, spec, entities)
        _add_reference_fixture(scene, spec, entities)
    _add_tianqing_robot(scene, spec, entities)
    if task_object == "manual":
        _add_manual_proxy(scene, spec, entities, physics_profile)
    elif task_object == "sponge":
        _add_sponge_proxy(scene, spec, entities, physics_profile)
    elif task_object == "tape":
        _add_tape_proxy(scene, spec, entities, physics_profile)
    elif task_object == "cable":
        _add_cable_connected_proxy(scene, spec, entities, physics_profile)

    if camera_specs is not None:
        entities["snapshot_camera"] = scene.add_camera(**camera_specs)

    scene.build()
    if uses_hinged_box:
        _initialize_box_folding_proxy(spec, entities, task_object)
    if task_object == "manual":
        _initialize_instruction_manual_proxy(spec, entities)
    if task_object == "cable":
        _initialize_cable_connected_proxy(spec, entities)
    if perturbation is not None:
        _apply_scene_perturbation(spec, task_object, entities, perturbation)
    debug_center = spec.box_fold_target_center if uses_hinged_box else spec.target_center
    scene.draw_debug_frame(_make_transform(debug_center), axis_length=0.18)
    return scene, entities


def _add_table(
    scene: gs.Scene,
    spec: WorkspaceSpec,
    entities: dict[str, object],
    physics_profile: PhysicalParameters,
) -> None:
    table_color = (0.66, 0.58, 0.49)
    leg_color = (0.3, 0.31, 0.34)
    leg_square = 0.05
    leg_height = spec.table_size[2] - spec.tabletop_thickness
    top_z = spec.table_center[2] + spec.table_size[2] / 2.0 - spec.tabletop_thickness / 2.0
    leg_z = leg_height / 2.0

    entities["table_top"] = scene.add_entity(
        gs.morphs.Box(
            fixed=True,
            pos=(spec.table_center[0], spec.table_center[1], top_z),
            size=(spec.table_size[0], spec.table_size[1], spec.tabletop_thickness),
        ),
        material=gs.materials.Rigid(friction=physics_profile.table_friction),
        surface=gs.surfaces.Default(color=table_color),
        name="table_top",
    )

    x_offset = spec.table_size[0] / 2.0 - leg_square / 2.0 - 0.06
    y_offset = spec.table_size[1] / 2.0 - leg_square / 2.0 - 0.04
    for idx, (dx, dy) in enumerate(
        [
            (-x_offset, -y_offset),
            (-x_offset, y_offset),
            (x_offset, -y_offset),
            (x_offset, y_offset),
        ],
        start=1,
    ):
        entities[f"table_leg_{idx}"] = scene.add_entity(
            gs.morphs.Box(
                fixed=True,
                pos=(spec.table_center[0] + dx, spec.table_center[1] + dy, leg_z),
                size=(leg_square, leg_square, leg_height),
            ),
            material=gs.materials.Rigid(friction=1.0),
            surface=gs.surfaces.Default(color=leg_color),
            name=f"table_leg_{idx}",
        )


def _add_target_zone(scene: gs.Scene, spec: WorkspaceSpec, entities: dict[str, object]) -> None:
    entities["target_zone"] = scene.add_entity(
        gs.morphs.Box(
            fixed=True,
            pos=spec.target_center,
            size=spec.target_size,
        ),
        material=gs.materials.Rigid(friction=1.0),
        surface=gs.surfaces.Default(color=(0.2, 0.78, 0.32), opacity=0.55),
        name="target_zone",
        )


def _add_box_fold_target_zone(scene: gs.Scene, spec: WorkspaceSpec, entities: dict[str, object]) -> None:
    entities["target_zone"] = scene.add_entity(
        gs.morphs.Box(
            fixed=True,
            pos=spec.box_fold_target_center,
            size=spec.box_fold_target_size,
        ),
        material=gs.materials.Rigid(friction=1.0),
        surface=gs.surfaces.Default(color=(0.28, 0.78, 0.42), opacity=0.45),
        name="target_zone",
    )


def _add_reference_fixture(scene: gs.Scene, spec: WorkspaceSpec, entities: dict[str, object]) -> None:
    fixture_color = (0.78, 0.8, 0.85)
    floor_x, floor_y, floor_z = spec.fixture_size
    cx, cy, cz = spec.fixture_center
    wall_t = spec.fixture_wall_thickness
    wall_h = spec.fixture_wall_height
    parts = {
        "fixture_floor": ((floor_x, floor_y, floor_z), (cx, cy, cz)),
        "fixture_back_wall": (
            (floor_x, wall_t, wall_h),
            (cx, cy - floor_y / 2.0 + wall_t / 2.0, cz + floor_z / 2.0 + wall_h / 2.0),
        ),
        "fixture_left_wall": (
            (wall_t, floor_y, wall_h),
            (cx - floor_x / 2.0 + wall_t / 2.0, cy, cz + floor_z / 2.0 + wall_h / 2.0),
        ),
        "fixture_right_wall": (
            (wall_t, floor_y, wall_h),
            (cx + floor_x / 2.0 - wall_t / 2.0, cy, cz + floor_z / 2.0 + wall_h / 2.0),
        ),
    }

    for name, (size, pos) in parts.items():
        entities[name] = scene.add_entity(
            gs.morphs.Box(fixed=True, pos=pos, size=size),
            material=gs.materials.Rigid(friction=1.1),
            surface=gs.surfaces.Default(color=fixture_color),
            name=name,
        )


def _add_box_fold_support(
    scene: gs.Scene,
    spec: WorkspaceSpec,
    entities: dict[str, object],
    physics_profile: PhysicalParameters,
) -> None:
    support_size = (spec.box_fold_target_size[0], spec.box_fold_target_size[1], 0.006)
    support_pos = (
        spec.box_fold_target_center[0],
        spec.box_fold_target_center[1],
        table_surface_z(spec) + support_size[2] / 2.0,
    )
    entities["box_fold_support"] = scene.add_entity(
        gs.morphs.Box(fixed=True, pos=support_pos, size=support_size),
        material=gs.materials.Rigid(friction=physics_profile.fixture_friction),
        surface=gs.surfaces.Default(color=(0.7, 0.73, 0.78)),
        name="box_fold_support",
    )


def _add_manual_proxy(
    scene: gs.Scene,
    spec: WorkspaceSpec,
    entities: dict[str, object],
    physics_profile: PhysicalParameters,
) -> None:
    entities["manual_booklet"] = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=str(_generated_instruction_manual_mjcf_path(spec, physics_profile)),
            pos=spec.manual_center,
            scale=1.0,
            requires_jac_and_IK=False,
        ),
        material=gs.materials.Rigid(
            rho=physics_profile.manual_density,
            friction=physics_profile.manual_friction,
        ),
        surface=gs.surfaces.Default(
            color=(0.95, 0.94, 0.87, 1.0),
            vis_mode="visual",
        ),
        name="manual_booklet",
    )


def _add_sponge_proxy(
    scene: gs.Scene,
    spec: WorkspaceSpec,
    entities: dict[str, object],
    physics_profile: PhysicalParameters,
) -> None:
    entities["sponge_pad_proxy"] = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(_genesis_cloth_mesh_path()),
            scale=spec.sponge_scale,
            pos=spec.sponge_center,
            euler=(0.0, 0.0, 0.0),
        ),
        material=gs.materials.PBD.Cloth(
            rho=physics_profile.sponge_areal_density,
            static_friction=physics_profile.sponge_static_friction,
            kinetic_friction=physics_profile.sponge_kinetic_friction,
            stretch_compliance=physics_profile.sponge_stretch_compliance,
            bending_compliance=physics_profile.sponge_bending_compliance,
            air_resistance=physics_profile.sponge_air_resistance,
        ),
        surface=gs.surfaces.Default(
            color=(0.95, 0.76, 0.18, 1.0),
            vis_mode="visual",
            double_sided=True,
        ),
        name="sponge_pad_proxy",
    )


def _add_tape_proxy(
    scene: gs.Scene,
    spec: WorkspaceSpec,
    entities: dict[str, object],
    physics_profile: PhysicalParameters,
) -> None:
    morph = gs.morphs.Mesh(
        file=str(_generated_tape_annulus_mesh_path(spec)),
        pos=spec.tape_center,
        scale=1.0,
        fixed=False,
        convexify=False,
        decimate=False,
        watertighten=0,
        recompute_inertia=True,
    )
    entities["tape_proxy"] = scene.add_entity(
        morph,
        material=gs.materials.Rigid(
            rho=physics_profile.tape_density,
            friction=physics_profile.tape_friction,
        ),
        surface=gs.surfaces.Default(
            color=(0.79, 0.68, 0.28, 1.0),
            vis_mode="visual",
        ),
        name="tape_proxy",
    )


def _add_box_folding_proxy(
    scene: gs.Scene,
    spec: WorkspaceSpec,
    entities: dict[str, object],
    physics_profile: PhysicalParameters,
    task_object: str,
) -> None:
    entities["box_folding_proxy"] = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=str(_generated_box_folding_mjcf_path(spec, physics_profile, task_object)),
            pos=spec.box_fold_proxy_center,
            scale=1.0,
            requires_jac_and_IK=False,
        ),
        material=gs.materials.Rigid(
            rho=physics_profile.box_equivalent_density,
            friction=physics_profile.box_friction,
        ),
        surface=gs.surfaces.Default(
            color=(0.72, 0.49, 0.24, 1.0),
            vis_mode="visual",
        ),
        name="box_folding_proxy",
    )


def _add_tianqing_robot(scene: gs.Scene, spec: WorkspaceSpec, entities: dict[str, object]) -> None:
    entities["tianqing_robot"] = scene.add_entity(
        morph=gs.morphs.URDF(
            file=str(_tianqing_urdf_path()),
            pos=spec.tianqing_robot_pos,
            euler=spec.tianqing_robot_euler,
            fixed=True,
            requires_jac_and_IK=True,
        ),
        material=gs.materials.Rigid(friction=1.2),
        surface=gs.surfaces.Default(
            color=(0.84, 0.84, 0.86, 1.0),
            vis_mode="visual",
        ),
        name="tianqing_robot",
    )


def _add_cable_connected_proxy(
    scene: gs.Scene,
    spec: WorkspaceSpec,
    entities: dict[str, object],
    physics_profile: PhysicalParameters,
) -> None:
    cable_color = (0.15, 0.15, 0.17, 1.0)
    entities["cable_connected_line_proxy"] = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(_genesis_cloth_mesh_path()),
            scale=spec.cable_connected_line_scale,
            pos=spec.cable_connected_line_center,
            euler=(0.0, 0.0, 0.0),
        ),
        material=gs.materials.PBD.Cloth(
            rho=physics_profile.cable_areal_density,
            static_friction=physics_profile.cable_static_friction,
            kinetic_friction=physics_profile.cable_kinetic_friction,
            stretch_compliance=physics_profile.cable_stretch_compliance,
            bending_compliance=physics_profile.cable_bending_compliance,
            air_resistance=physics_profile.cable_air_resistance,
        ),
        surface=gs.surfaces.Default(
            color=cable_color,
            vis_mode="visual",
            double_sided=True,
        ),
        name="cable_connected_line_proxy",
    )

    entities["cable_connected_mouse"] = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(_generated_cable_connected_mouse_mesh_path(spec)),
            pos=_cable_mouse_spawn_position(spec),
            scale=1.0,
            fixed=False,
            convexify=True,
            recompute_inertia=True,
        ),
        material=gs.materials.Rigid(
            rho=physics_profile.cable_end_density,
            friction=physics_profile.cable_end_friction,
        ),
        surface=gs.surfaces.Default(
            color=cable_color,
            vis_mode="visual",
        ),
        name="cable_connected_mouse",
    )


def _build_cable_connected_centerline(spec: WorkspaceSpec, n_sections: int) -> np.ndarray:
    anchor = np.array(
        [
            spec.cable_mouse_nose_center[0] - 0.002,
            spec.cable_mouse_nose_center[1] - spec.cable_mouse_nose_size[1] / 2.0 - 0.004,
        ],
        dtype=np.float32,
    )
    points = np.array(
        [
            anchor,
            (0.792, -0.158),
            (0.752, -0.191),
            (0.701, -0.191),
            (0.667, -0.163),
            (0.673, -0.116),
            (0.716, -0.091),
            (0.769, -0.091),
            (0.804, -0.118),
            (0.804, -0.158),
            (0.777, -0.179),
            (0.735, -0.179),
            (0.708, -0.158),
            (0.708, -0.126),
            (0.735, -0.108),
            (0.771, -0.109),
            (0.792, -0.126),
            (0.792, -0.151),
        ],
        dtype=np.float32,
    )

    deltas = np.diff(points, axis=0)
    seg_lens = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(seg_lens)))
    samples = np.linspace(0.0, cumulative[-1], n_sections)
    xs = np.interp(samples, cumulative, points[:, 0])
    ys = np.interp(samples, cumulative, points[:, 1])
    zs = np.full_like(xs, table_surface_z(spec) + 0.0045)
    return np.stack([xs, ys, zs], axis=1)


def _generated_box_folding_mjcf_path(
    spec: WorkspaceSpec,
    physics_profile: PhysicalParameters = FIRM_PHYSICS,
    task_object: str = "box",
) -> Path:
    output_dir = Path(__file__).resolve().parent / "generated_assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    lid_rest_angle = (
        spec.box_fold_lid_closed_angle
        if task_object == "box"
        else spec.box_fold_lid_open_angle_outward
    )
    box_mode = "hinged_closed" if task_object == "box" else "fixed_open"
    xml_path = output_dir / f"box_{box_mode}.xml"

    base_x, base_y, base_t = spec.box_fold_base_size
    wall_h = spec.box_fold_wall_height
    wall_t = spec.box_fold_wall_thickness
    lid_t = spec.box_fold_lid_thickness
    lid_depth = (base_y - wall_t) * 1.25
    lid_depth_half = lid_depth / 2.0

    if task_object == "box":
        lid_body_pose = ""
        lid_joint_xml = f"""        <joint name="lid_hinge" type="hinge" axis="1 0 0" limited="true" range="0 2.9"
               stiffness="{physics_profile.box_hinge_stiffness:.6f}"
               damping="{physics_profile.box_hinge_damping:.6f}"
               springref="{lid_rest_angle:.6f}"/>"""
    else:
        lid_body_pose = f' euler="{lid_rest_angle:.6f} 0 0"'
        lid_joint_xml = ""

    xml = f"""<mujoco model="box_{box_mode}">
  <compiler angle="radian"/>
  <worldbody>
    <body name="box_root">
      <geom name="bottom" type="box" pos="0 0 0" size="{base_x / 2:.6f} {base_y / 2:.6f} {base_t / 2:.6f}"/>
      <geom name="left_wall" type="box" pos="{-base_x / 2 + wall_t / 2:.6f} 0 {wall_h / 2:.6f}" size="{wall_t / 2:.6f} {base_y / 2:.6f} {wall_h / 2:.6f}"/>
      <geom name="right_wall" type="box" pos="{base_x / 2 - wall_t / 2:.6f} 0 {wall_h / 2:.6f}" size="{wall_t / 2:.6f} {base_y / 2:.6f} {wall_h / 2:.6f}"/>
      <geom name="front_wall" type="box" pos="0 {base_y / 2 - wall_t / 2:.6f} {wall_h / 2:.6f}" size="{base_x / 2:.6f} {wall_t / 2:.6f} {wall_h / 2:.6f}"/>
      <geom name="back_wall" type="box" pos="0 {-base_y / 2 + wall_t / 2:.6f} {wall_h / 2:.6f}" size="{base_x / 2:.6f} {wall_t / 2:.6f} {wall_h / 2:.6f}"/>
      <body name="lid" pos="0 {-base_y / 2 + wall_t / 2:.6f} {wall_h + lid_t / 2:.6f}"{lid_body_pose}>
{lid_joint_xml}
        <geom name="lid_panel" type="box" pos="0 {lid_depth_half:.6f} 0" size="{base_x / 2:.6f} {lid_depth_half:.6f} {lid_t / 2:.6f}"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
    xml_path.write_text(xml, encoding="utf-8")
    return xml_path


def _tianqing_urdf_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    zip_path = root / "tianqing_urdf.zip"
    extract_dir = root / "assets" / "tianqing_urdf"
    urdf_path = extract_dir / "a2p_v3.urdf"
    if urdf_path.exists():
        return urdf_path
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    return urdf_path


def _generated_instruction_manual_mjcf_path(
    spec: WorkspaceSpec,
    physics_profile: PhysicalParameters = FIRM_PHYSICS,
) -> Path:
    output_dir = Path(__file__).resolve().parent / "generated_assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = output_dir / "instruction_manual_booklet.xml"

    manual_x, manual_y, manual_t = spec.manual_size
    sheet_count = 5
    sheet_thickness = manual_t / sheet_count
    sheet_half_t = sheet_thickness / 2.0
    sheet_geoms = []
    for index in range(sheet_count):
        z = (index - (sheet_count - 1) / 2.0) * sheet_thickness
        sheet_number = index + 1
        sheet_geoms.append(
            f'      <geom name="booklet_sheet_{sheet_number}" type="box" '
            f'pos="0 0 {z:.6f}" '
            f'size="{manual_x / 2.0:.6f} {manual_y / 2.0:.6f} {sheet_half_t:.6f}"/>'
        )

    xml = f"""<mujoco model="instruction_manual_booklet">
  <compiler angle="radian"/>
  <worldbody>
    <body name="manual_booklet">
{"\n".join(sheet_geoms)}
    </body>
  </worldbody>
</mujoco>
"""
    xml_path.write_text(xml, encoding="utf-8")
    return xml_path


def _generated_tape_annulus_mesh_path(spec: WorkspaceSpec) -> Path:
    output_dir = Path(__file__).resolve().parent / "generated_assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = output_dir / "tape_annulus.obj"

    segments = 64
    half_height = spec.tape_height / 2.0
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    for index in range(segments):
        angle = 2.0 * np.pi * index / segments
        cos_angle = float(np.cos(angle))
        sin_angle = float(np.sin(angle))
        vertices.extend(
            [
                (spec.tape_radius * cos_angle, spec.tape_radius * sin_angle, -half_height),
                (spec.tape_radius * cos_angle, spec.tape_radius * sin_angle, half_height),
                (
                    spec.tape_inner_radius * cos_angle,
                    spec.tape_inner_radius * sin_angle,
                    -half_height,
                ),
                (
                    spec.tape_inner_radius * cos_angle,
                    spec.tape_inner_radius * sin_angle,
                    half_height,
                ),
            ]
        )

    for index in range(segments):
        current = index * 4 + 1
        following = ((index + 1) % segments) * 4 + 1
        outer_bottom, outer_top, inner_bottom, inner_top = range(current, current + 4)
        next_outer_bottom, next_outer_top, next_inner_bottom, next_inner_top = range(
            following, following + 4
        )
        faces.extend(
            [
                (outer_bottom, next_outer_bottom, next_outer_top),
                (outer_bottom, next_outer_top, outer_top),
                (inner_bottom, inner_top, next_inner_top),
                (inner_bottom, next_inner_top, next_inner_bottom),
                (outer_top, next_outer_top, next_inner_top),
                (outer_top, next_inner_top, inner_top),
                (outer_bottom, inner_bottom, next_inner_bottom),
                (outer_bottom, next_inner_bottom, next_outer_bottom),
            ]
        )

    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    lines.extend(f"f {a} {b} {c}" for a, b, c in faces)
    mesh_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return mesh_path


def _initialize_box_folding_proxy(
    spec: WorkspaceSpec,
    entities: dict[str, object],
    task_object: str,
) -> None:
    if task_object != "box":
        return
    box = entities["box_folding_proxy"]
    box.set_dofs_position(
        np.array([spec.box_fold_lid_closed_angle], dtype=np.float32),
        zero_velocity=True,
    )


def _initialize_instruction_manual_proxy(spec: WorkspaceSpec, entities: dict[str, object]) -> None:
    return


def _generated_cable_connected_mouse_mesh_path(spec: WorkspaceSpec) -> Path:
    output_dir = Path(__file__).resolve().parent / "generated_assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = output_dir / "cable_connected_mouse.obj"

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    origin = np.asarray(spec.cable_mouse_body_center, dtype=np.float32)
    for center, size in (
        (spec.cable_mouse_body_center, spec.cable_mouse_body_size),
        (spec.cable_mouse_top_center, spec.cable_mouse_top_size),
        (spec.cable_mouse_nose_center, spec.cable_mouse_nose_size),
    ):
        local_center = tuple(np.asarray(center, dtype=np.float32) - origin)
        _append_box_mesh(vertices, faces, local_center, size)

    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    lines.extend(f"f {a} {b} {c}" for a, b, c in faces)
    mesh_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return mesh_path


def _initialize_cable_connected_proxy(
    spec: WorkspaceSpec,
    entities: dict[str, object],
) -> None:
    line = entities["cable_connected_line_proxy"]
    mouse = entities["cable_connected_mouse"]
    particles = line.get_particles_pos().detach().cpu().numpy()
    n_particles = particles.shape[0]
    n_sections = n_particles // 3
    width = 0.0048
    z = table_surface_z(spec) + 0.006

    centerline = _build_cable_connected_centerline(spec, n_sections)
    spawn_offset = np.asarray(_cable_spawn_offset(spec), dtype=np.float32)
    centerline += spawn_offset
    tangent = np.gradient(centerline[:, :2], axis=0)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)

    reordered = np.zeros_like(particles)
    for section_idx in range(n_sections):
        center = centerline[section_idx]
        offset = normal[section_idx] * width
        base = section_idx * 3
        reordered[base + 0] = np.array([center[0] - offset[0], center[1] - offset[1], z], dtype=np.float32)
        reordered[base + 1] = np.array([center[0], center[1], z], dtype=np.float32)
        reordered[base + 2] = np.array([center[0] + offset[0], center[1] + offset[1], z], dtype=np.float32)

    line.set_particles_pos(reordered)
    line.fix_particles_to_link(mouse.base_link_idx, particles_idx_local=np.arange(0, 6))


def _apply_scene_perturbation(
    spec: WorkspaceSpec,
    task_object: str,
    entities: dict[str, object],
    sample: PerturbationSample,
) -> None:
    """Apply physical perturbations after all nominal poses are initialized."""
    fixture_xy = sample.fixture_translation_xy_m
    if fixture_xy != (0.0, 0.0):
        for name in ("box_fold_support", "target_zone"):
            if name in entities:
                _translate_rigid_entity(entities[name], fixture_xy)
        if task_object != "box" and "box_folding_proxy" in entities:
            _translate_rigid_entity(entities["box_folding_proxy"], fixture_xy)

    object_xy = sample.object_translation_xy_m
    yaw_deg = sample.object_yaw_deg
    if task_object == "sponge":
        _transform_particles(entities["sponge_pad_proxy"], object_xy, yaw_deg)
    elif task_object == "cable" and "cable_connected_line_proxy" in entities:
        mouse = entities["cable_connected_mouse"]
        pivot = _as_numpy(mouse.get_pos())[:2]
        _transform_particles(
            entities["cable_connected_line_proxy"],
            object_xy,
            yaw_deg,
            pivot_xy=pivot,
        )
        _transform_rigid_entity(mouse, object_xy, yaw_deg)
    else:
        entity_name = {
            "manual": "manual_booklet",
            "tape": "tape_proxy",
            "box": "box_folding_proxy",
        }.get(task_object)
        if entity_name is not None and entity_name in entities:
            _transform_rigid_entity(entities[entity_name], object_xy, yaw_deg)


def _translate_rigid_entity(entity: object, xy: tuple[float, float]) -> None:
    position = _as_numpy(entity.get_pos()).astype(np.float32)
    position[:2] += np.asarray(xy, dtype=np.float32)
    entity.set_pos(position, relative=False, zero_velocity=True)


def _transform_rigid_entity(
    entity: object,
    xy: tuple[float, float],
    yaw_deg: float,
) -> None:
    _translate_rigid_entity(entity, xy)
    if yaw_deg == 0.0:
        return
    current = _as_numpy(entity.get_quat()).astype(np.float32)
    yaw = np.deg2rad(yaw_deg)
    yaw_quat = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float32)
    entity.set_quat(_multiply_quaternions(yaw_quat, current), relative=False, zero_velocity=True)


def _transform_particles(
    entity: object,
    xy: tuple[float, float],
    yaw_deg: float,
    pivot_xy: np.ndarray | None = None,
) -> None:
    particles = _as_numpy(entity.get_particles_pos()).astype(np.float32)
    pivot = particles[:, :2].mean(axis=0) if pivot_xy is None else np.asarray(pivot_xy, dtype=np.float32)
    if yaw_deg != 0.0:
        angle = np.deg2rad(yaw_deg)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float32,
        )
        particles[:, :2] = (particles[:, :2] - pivot) @ rotation.T + pivot
    particles[:, :2] += np.asarray(xy, dtype=np.float32)
    entity.set_particles_pos(particles)


def _multiply_quaternions(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float32,
    )


def _as_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _append_box_mesh(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    center: tuple[float, float, float],
    size: tuple[float, float, float],
) -> None:
    cx, cy, cz = center
    sx, sy, sz = size
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    start = len(vertices) + 1
    vertices.extend(
        [
            (cx - hx, cy - hy, cz - hz),
            (cx + hx, cy - hy, cz - hz),
            (cx + hx, cy + hy, cz - hz),
            (cx - hx, cy + hy, cz - hz),
            (cx - hx, cy - hy, cz + hz),
            (cx + hx, cy - hy, cz + hz),
            (cx + hx, cy + hy, cz + hz),
            (cx - hx, cy + hy, cz + hz),
        ]
    )
    faces.extend(
        [
            (start, start + 1, start + 2),
            (start, start + 2, start + 3),
            (start + 4, start + 7, start + 6),
            (start + 4, start + 6, start + 5),
            (start, start + 4, start + 5),
            (start, start + 5, start + 1),
            (start + 1, start + 5, start + 6),
            (start + 1, start + 6, start + 2),
            (start + 2, start + 6, start + 7),
            (start + 2, start + 7, start + 3),
            (start + 3, start + 7, start + 4),
            (start + 3, start + 4, start),
        ]
    )


def _genesis_cloth_mesh_path() -> Path:
    return Path(gs.__file__).resolve().parent / "assets" / "meshes" / "cloth.obj"


def table_surface_z(spec: WorkspaceSpec) -> float:
    return spec.table_center[2] + spec.table_size[2] / 2.0


def _cable_spawn_offset(spec: WorkspaceSpec) -> tuple[float, float, float]:
    layout_center_x = 0.7641
    layout_center_y = -0.1428
    return (
        spec.sponge_center[0] - layout_center_x,
        spec.sponge_center[1] - layout_center_y,
        0.0,
    )


def _cable_mouse_spawn_position(spec: WorkspaceSpec) -> tuple[float, float, float]:
    offset = _cable_spawn_offset(spec)
    return tuple(
        center + delta
        for center, delta in zip(spec.cable_mouse_body_center, offset, strict=True)
    )


def _make_transform(position: tuple[float, float, float]):
    return [
        [1.0, 0.0, 0.0, position[0]],
        [0.0, 1.0, 0.0, position[1]],
        [0.0, 0.0, 1.0, position[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
