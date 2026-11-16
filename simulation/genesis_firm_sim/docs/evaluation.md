# Perturbation robustness evaluation

## Protocol

Each method–task pair is evaluated on three separate families. Every family uses strengths δ = {0, 0.25, 0.50, 0.75, 1.00}, with 20 rollouts per strength. Amplitudes below are multiplied by δ.

| Family | Quantity | Maximum | Distribution |
|---|---|---:|---|
| Object | Planar translation | 40 mm | Uniform in a disk |
| Object | Yaw | 20° | Uniform in ± amplitude |
| Fixture | Planar translation | 20 mm | Uniform in a disk |
| Fixture | Yaw | 10° | Uniform in ± amplitude |
| Scene/perception | Camera translation | 10 mm | Uniform in a ball |
| Scene/perception | Camera rotation | 5° | Rotation vector uniform in a ball |
| Scene/perception | RGB noise | σ = 10/255 | Independent Gaussian per pixel/channel |
| Scene/perception | Depth noise | 5 mm | Independent uniform in ± amplitude |
| Scene/perception | Distractors | 4 | Seeded subset of four peripheral positions |

Object and fixture perturbations rotate around their respective initial object root and target center. PBD objects rotate around their particle centroid. Box folding perturbs the box as the object and the support/target as the fixture. Other tasks perturb the receiving box, support, and target together as the fixture. The camera rotation changes orientation in the world frame; translation is applied independently. Camera error is fixed throughout a rollout, while image noise changes at every simulation step. RGB is clipped to its valid range. Invalid depth is preserved. Distractor counts are 0, 1, 2, 3, and 4 for the five strengths. All effects vanish at δ=0.

Translation and angular values specify sampling bounds rather than a displacement forced to the bound. Random directions and normalized amplitudes are paired across strengths for a given task, family, and trial. Sample identifiers are hashes of the full condition. All methods use the same schedule; method names never enter the sample seed.

## Policy interface

The configuration names a policy factory as `module:attribute`. The factory receives `policy_options` and returns an object with:

```python
def reset(self, observation): ...
def act(self, observation): ...
```

Observations contain `rgb`, `depth_m`, and `joint_position`. RGB is H×W×3, depth is in metres, and joint positions use the Genesis robot DOF order. An action is a finite position target for every robot DOF; `None` advances physics without changing targets. Policy factories may load model weights specified in their options. Keep weights and external adapter code fixed and record their revisions in the evaluation configuration. All methods must share camera, termination, and success settings.

## Success and termination

The judge factory receives `judge_options`, then implements `reset(env)` and `evaluate(env) -> (success, terminated)`. Judges inspect simulator state through the environment; observation noise does not alter ground truth. A successful or terminated rollout stops immediately. Reaching `max_steps` without success is a completed failure. Exceptions and wall-clock timeouts are recorded as errors and excluded from aggregation until successfully rerun.

`GeometricJudge` measures the object's root position (particle centroid for PBD) in the target frame. `lower_m` and `upper_m` bound this position; `hold_steps` requires consecutive qualifying simulation steps. An optional `hinge` object adds `entity`, `dof`, `lower_rad`, and `upper_rad` constraints, for example the box lid angle. These geometric criteria describe the proxy scenes. Task-specific insertion, folding, deformation, or contact criteria can be supplied by a judge using the same interface. The supplied stationary baseline and placement bounds are an installation example; published task results require the corresponding declared task criteria and trained policies.

## Metric

For each family and strength, SR is the successful rollout count divided by 20. The family AUC is the trapezoidal integral of SR over δ, divided by the strength interval (1.0). PR-AUC is the arithmetic mean of object, fixture, and scene/perception AUC. No task-dependent family weights are applied.

For success rates [0.90, 0.80, 0.70, 0.50, 0.30], the family AUC is 0.65. PR-AUC lies in [0, 1] and reflects average success across the perturbation interval. Report nominal SR and the curves alongside PR-AUC to distinguish baseline ability from degradation. This metric measures simulator-side robustness under the stated models and perturbations.

## Reproducibility

The output directory retains the schedule, configuration, package source and robot archive digests, runtime versions, per-condition logs, resolved initial/final states, and Genesis version. Fresh worker processes prevent scene and solver state from leaking between rollouts. Exact numerical reproducibility across hardware backends is not guaranteed; compare methods on the same software and hardware configuration. Use a separate output directory for each method/configuration. Do not run simultaneous writers against one output directory.
