---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Dynamical decoupling for Linear schedules

The Linear backend provides three dynamical-decoupling (DD) passes over an
{py:class}`~mqt.ionshuttler.linear.ActionSchedule`:

- shuttling-aware dynamical decoupling (SADD), with pulse-only and
  transport-enabled variants;
- an idealized Hahn reference that ignores hardware-control constraints; and
- periodic global DD with optional pulse-position refinement.

Every pass returns a {py:class}`~mqt.ionshuttler.linear.dd.DDPassResult`. Its
`schedule` is the transformed action schedule, while its `report` contains
method-specific decisions and diagnostics. The input schedule is immutable.

## Schedule and report ownership

An action schedule describes ordered hardware-level actions, their stable
identifiers, the architecture, and the initial machine state. It does not label
why a gate was introduced. In particular, a local DD rotation and an equal
algorithmic rotation have the same action representation.

Local-pulse identity belongs to
{py:class}`~mqt.ionshuttler.linear.dd.LocalDDSequence`. Each sequence records
parallel `pulse_timesteps` and `action_ids` tuples. Consumers that distinguish
local pulses from algorithmic gates should pass the reported IDs explicitly:

```python
local_pulse_action_ids = frozenset(
    action_id for sequence in output.report.sequences for action_id in sequence.action_ids
)
```

Global pulses use the distinct
{py:class}`~mqt.ionshuttler.linear.actions.GlobalPulse` action and therefore do
not require a parallel identity record.

## Idealized Hahn reference

The idealized reference inserts a Hahn sequence into every sufficiently long
ion-local idle window. It is useful as a comparator, but its output is not a
claim that the inserted controls are executable on the modeled device.

```{code-cell} ipython3
from mqt.ionshuttler.linear import ActionSchedule, Architecture
from mqt.ionshuttler.linear.actions import AdvanceTime
from mqt.ionshuttler.linear.dd import apply_idealized_hahn
from mqt.ionshuttler.linear.state import create_initial_state

architecture = Architecture(num_sites=1, processing_zones={"pz": [0]})
schedule = ActionSchedule.from_actions(
    [AdvanceTime() for _ in range(4)],
    architecture,
    create_initial_state(1, architecture),
)
output = apply_idealized_hahn(schedule)

output.report.sequences[0].pulse_timesteps
```

## Shuttling-aware dynamical decoupling

{py:func}`~mqt.ionshuttler.linear.dd.run_sadd` solves bounded control windows
with the optional OR-Tools dependency. `PULSE_ONLY` may insert local pulses but
does not alter transport. `FULL` may additionally move participating ions to and
from processing-zone control sites.

```python
from mqt.ionshuttler.linear.dd import SADDConfig, SADDMethod, run_sadd

output = run_sadd(
    schedule,
    SADDMethod.FULL,
    SADDConfig(max_accepted_windows=1),
)
```

The default `SADDConfig` contains the paper-narrative optimization parameters.
An opportunity is accepted only when its reconstructed schedule is replay-valid
and its phase objective improves by more than `improvement_tolerance`. Solver
runtime, model size, participant selection, pulse positions, trajectories, and
transport decisions are available in the ordered opportunity records.

## Periodic global DD

{py:func}`~mqt.ionshuttler.linear.dd.apply_periodic_global_dd` places global
odd-$\pi$ rotations at a configured spacing. With a nonzero `shift_range`, it
uses coordinate descent to refine pulse boundaries according to the selected
residual-phase objective.

```python
from mqt.ionshuttler.linear.dd import GlobalDDConfig, apply_periodic_global_dd

output = apply_periodic_global_dd(schedule, GlobalDDConfig(spacing=4))
```

## Comparison utilities

The DD package includes critical-segment reconstruction, Pauli-frame replay, and
reusable schedule metrics. Local-pulse-aware functions accept the explicit set
of report-owned action identifiers:

```python
from mqt.ionshuttler.linear.dd import compute_critical_segments

trace = compute_critical_segments(
    output.schedule,
    local_pulse_action_ids=local_pulse_action_ids,
)
trace.j_phi
```

The phase metrics are schedule-ranking and comparison proxies. They are not a
complete noisy-circuit simulation; the simulation stack is a separate layer.

## See also

- {doc}`linear_compiler` — compile circuits into action schedules
- {doc}`linear_hardware_model` — understand sites, processing zones, and timing
- {doc}`api/mqt/ionshuttler/linear/dd/index` — complete DD Python API reference
