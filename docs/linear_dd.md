---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Dynamical decoupling for Linear schedules

Dynamical decoupling (DD) adds control pulses to a compiled schedule to reduce
the phase accumulated by idle ions. The Linear backend provides four methods:

| Method                    | Intended use                                       | Hardware constraints                                   |
| ------------------------- | -------------------------------------------------- | ------------------------------------------------------ |
| Shuttling-aware DD (SADD) | Producing a schedule intended for execution        | Respects control locations and may add transport       |
| Nearest Hahn reference    | Measuring what unmodified transport already allows | Respects control locations and adds no transport       |
| Idealized Hahn reference  | Estimating the benefit of unrestricted control     | Deliberately ignores where local pulses can be applied |
| Periodic global DD        | Applying one pulse to all ions at regular spacing  | Uses schedule-wide global pulses                       |

Each method leaves the input {py:class}`~mqt.ionshuttler.linear.ActionSchedule`
unchanged and returns a {py:class}`~mqt.ionshuttler.linear.dd.DDPassResult`. The
result contains the transformed `schedule` and a method-specific `report`.

## Installation

The Hahn references and the global method use the standard installation. SADD
also requires OR-Tools:

```console
pip install "mqt.ionshuttler[dd]"
```

OR-Tools is loaded only when SADD reaches an optimization opportunity.

## Compile a circuit

The examples compile a depth-2 four-qubit circuit for a six-site architecture
with one two-site processing zone. Executing the circuit requires both transport
and idle time, giving the DD passes a nontrivial schedule to work with.

```{code-cell} ipython3
from collections import Counter
from dataclasses import replace

from qiskit import QuantumCircuit

from mqt.ionshuttler.linear import Architecture, LinearCompiler
from mqt.ionshuttler.linear.actions import DEFAULT_ACTION_TYPES, GlobalPulse
from mqt.ionshuttler.linear.dd import compute_critical_segments
from mqt.ionshuttler.linear.field_profile import FieldProfile

compilation_architecture = Architecture(
    num_sites=6,
    processing_zones={"pz": [2, 3]},
    field_profile=FieldProfile(
        6,
        (
            (0, 4.0),
            (1, 3.0),
            (2, 1.0),
            (3, 1.0),
            (4, 3.0),
            (5, 4.0),
        ),
    ),
)

circuit = QuantumCircuit(4)
circuit.rx(0.25, 0)
circuit.ry(0.5, 1)
circuit.rzz(0.75, 0, 1)
circuit.rzz(0.5, 2, 3)
circuit.rx(0.25, 2)

compilation = LinearCompiler(compilation_architecture).compile(circuit)
schedule = compilation.schedule
dd_architecture = replace(
    compilation.architecture,
    supported_action_types=(*DEFAULT_ACTION_TYPES, GlobalPulse),
)
{
    "circuit_depth": circuit.depth(),
    "status": compilation.status.value,
    "timesteps": schedule.num_timesteps,
    "actions": dict(Counter(type(action).__name__ for action in schedule.path)),
}
```

## Phase metric

Dephasing is modeled as quasistatic, site-dependent longitudinal $Z$-phase
accumulation. A refocusing pulse reverses the sign of subsequent accumulation,
allowing positive and negative contributions to cancel.

The phase is evaluated at *critical points*: logical gates that rotate the
accumulated phase out of the longitudinal axis and therefore terminate a
phase-coherent segment. Residual phase at these points is considered detrimental
to execution fidelity and is used as a cost and comparison metric by the DD
methods. The implementation exposes this as `phase_cost`; it corresponds to
$J_\phi$, the sum of squared residual phases over all critical segments:

```{code-cell} ipython3
baseline = compute_critical_segments(schedule, dd_architecture)
{"critical_segments": len(baseline.segments), "phase_cost": round(baseline.phase_cost, 3)}
```

Smaller values indicate less residual phase under this model. The value is a
schedule-comparison metric, not a complete noisy-circuit fidelity estimate.

## Shuttling-aware dynamical decoupling

`SADDMethod.PULSE_ONLY` adds local pulses without changing transport.
`SADDMethod.FULL` may also move ions to and from control sites. Both variants
use the same SADD backend (publication pending).

```{code-cell} ipython3
from mqt.ionshuttler.linear.dd import SADDConfig, SADDMethod, run_sadd

sadd = run_sadd(
    schedule,
    dd_architecture,
    SADDMethod.FULL,
    SADDConfig(max_accepted_windows=1, num_search_workers=1),
)
opportunity = next(record for record in sadd.report.opportunities if record.accepted)
{
    "window": opportunity.window,
    "status": opportunity.status,
    "eligible_ions": opportunity.eligible_ions,
    "participating_ions": opportunity.participating_ions,
    "pulse_timesteps": dict(opportunity.pulse_timesteps or {}),
    "transport_delta": dict(opportunity.transport_delta),
    "phase_cost_before": round(opportunity.phase_cost_before, 3),
    "phase_cost_after": round(opportunity.phase_cost_after or 0.0, 3),
}
```

The default {py:class}`~mqt.ionshuttler.linear.dd.SADDConfig` uses the current
paper configuration: windows of 2–16 timesteps, at most five ions per window, a
10-second timeout, phase-based ion ordering, chronological windows, eight solver
workers, and shuttle/swap/local-pulse durations of 1/3/1 timesteps.

The opportunity records expose the solver status, whether a proposal was
accepted, its objective values, and the inserted pulse and transport actions.
`transport_delta` gives the signed change in transport actions by type relative
to the schedule entering that opportunity. Runtime and model-size fields are
diagnostic and may vary between runs.

## Idealized Hahn reference

The idealized Hahn pass inserts local refocusing sequences without enforcing
where or alongside which operations the pulses can be applied. It provides a
comparison point rather than an executable hardware schedule.

By default it places a single pulse at each idle window's midpoint. The Pauli
frame that pulse introduces is not undone by a second physical pulse; it
persists and is discharged virtually by a consumer that corrects the terminal
frame. Set `include_terminating_pulse=True` for the closing-pulse variant, which
returns the frame to the identity at the cost of a second pulse per window.

```{code-cell} ipython3
from mqt.ionshuttler.linear.dd import IdealizedHahnConfig, apply_idealized_hahn

hahn = apply_idealized_hahn(
    schedule,
    dd_architecture,
    IdealizedHahnConfig(min_idle_timesteps=11),
)
hahn_pulse_ids = frozenset(
    action_id
    for sequence in hahn.report.sequences
    for action_id in sequence.action_ids
)
{
    "sequences": len(hahn.report.sequences),
    "pulse_timesteps": hahn.report.sequences[0].pulse_timesteps,
}
```

The minimum idle-window length restricts this short example to its two longest
idle intervals. The default processes every idle interval long enough to hold
the selected pulse sequence.

## Nearest Hahn reference

Nearest Hahn places one X pulse as close to each idle window's midpoint as the
compiled trajectory already permits. It uses only processing-zone access the ion
already has, inserts no transport, and never extends the makespan, so unlike the
idealized reference every schedule it returns is executable on the given
architecture. Its frame also persists and is corrected virtually.

```{code-cell} ipython3
from mqt.ionshuttler.linear.dd import NearestHahnConfig, run_nearest_hahn

nearest = run_nearest_hahn(schedule, dd_architecture, NearestHahnConfig())
nearest_pulse_ids = frozenset(
    action_id
    for sequence in nearest.report.sequences
    for action_id in sequence.action_ids
)
{
    "placed": nearest.report.placed,
    "eligible_windows": len(nearest.report.opportunities),
    "statuses": sorted({record.status for record in nearest.report.opportunities}),
}
```

Every eligible window appears in `opportunities`, including the ones that
received no pulse. A skipped window records why: the ion never reached a
processing zone, the ion was busy, or the zone was occupied. A placed pulse
records its displacement from the ideal midpoint, so a comparison can separate
windows the hardware served exactly from windows it served only approximately.

```{code-cell} ipython3
[
    {
        "ion": record.ion,
        "window": record.window,
        "status": record.status,
        "selected_timestep": record.selected_timestep,
        "signed_displacement": record.signed_displacement,
        "skip_reason": record.skip_reason,
    }
    for record in nearest.report.opportunities
]
```

## Periodic global DD

Periodic global DD inserts X pulses that act on all ions. `spacing` sets their
nominal separation. A nonzero `shift_range` allows small position adjustments
that minimize the same critical-segment $J_\phi$ metric used by SADD.

```{code-cell} ipython3
from mqt.ionshuttler.linear.dd import GlobalDDConfig, apply_periodic_global_dd

global_dd = apply_periodic_global_dd(schedule, dd_architecture, GlobalDDConfig(spacing=10))
{
    "pulse_timesteps": global_dd.report.pulse_timesteps,
    "phase_cost": round(global_dd.report.phase_cost, 3),
}
```

The ten-timestep spacing inserts one global pulse in this 15-timestep schedule.
Shorter spacing would assume a substantially higher global control rate.

The base circuit was compiled before `GlobalPulse` was added to the architecture
catalog. The DD pass accepts the extended architecture because the original
schedule remains valid on it. It would reject an architecture without
`GlobalPulse` or one that cannot execute the existing schedule.

## Comparing results

Local DD pulses use ordinary single-qubit actions. Pass their report-owned
action IDs to the phase analysis so they are not interpreted as logical gates:

```{code-cell} ipython3
sadd_pulse_ids = frozenset(
    action_id
    for sequence in sadd.report.sequences
    for action_id in sequence.action_ids
)

{
    "without_dd": round(compute_critical_segments(schedule, dd_architecture).phase_cost, 3),
    "sadd": round(
        compute_critical_segments(
            sadd.schedule,
            dd_architecture,
            local_pulse_action_ids=sadd_pulse_ids,
        ).phase_cost,
        3,
    ),
    "nearest_hahn": round(
        compute_critical_segments(
            nearest.schedule,
            dd_architecture,
            local_pulse_action_ids=nearest_pulse_ids,
        ).phase_cost,
        3,
    ),
    "idealized_hahn": round(
        compute_critical_segments(
            hahn.schedule,
            dd_architecture,
            local_pulse_action_ids=hahn_pulse_ids,
        ).phase_cost,
        3,
    ),
    "global_dd": round(global_dd.report.phase_cost, 3),
}
```

Use the same field profile and metric settings for all schedules in a
comparison. These values do not compare methods under equal control constraints.
SADD and Nearest Hahn respect the hardware's control access, so their numbers
are achievable. Idealized Hahn assumes unconstrained local pulses and global DD
assumes schedule-wide pulses; both reflect substantially relaxed control
assumptions and are reference points, not competing implementations.

## Limitations

- Equal-quality SADD solutions may place pulses differently; compare their
  validity and objective values rather than one exact placement.
- The idealized Hahn result is intentionally not hardware constrained.
- Both Hahn references leave a persistent Pauli frame by default. A consumer
  that does not correct the terminal frame will misread their results.
- These phase metrics are comparison proxies. Full noisy-circuit simulation is
  provided by a separate simulation layer.

## See also

- {doc}`linear_compiler` — compile circuits into action schedules
- {doc}`linear_hardware_model` — define sites, processing zones, timing, and
  field profiles
- {doc}`api/mqt/ionshuttler/linear/dd/index` — consult the complete DD Python
  API
