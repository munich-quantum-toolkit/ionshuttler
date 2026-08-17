# Trapped-Ion Hardware Model

The Linear compiler models a one-dimensional quantum charge-coupled device
(QCCD): ions are stored at discrete trap sites, moved through the device, and
brought into designated processing zones for quantum operations, also referred
to as *laser interaction zones* (LIZ). This follows the central QCCD idea of
separating storage from interaction regions and using controlled ion transport
to assemble the qubits needed for each operation.
{cite:p}`kielpinski2002architecture,schoenberger2024shuttling`.

The model lives at the scheduling level (see also {doc}`linear_design`). It
describes where ions may be, which operations may occur together, and how long
those operations occupy hardware resources. It does not attempt to reproduce
electrode waveforms, motional-mode dynamics, or laser pulses. Those belong to
(absent) lower control and physics layers.

## A chain of discrete sites

The device is represented as an ordered set of sites:

<figure>
  <img src="_static/linear_hardware_model.png" width="100%" alt="Schematic of
  ions above a linear segmented trap. Colored site markers identify two
  processing-zone regions, arrows indicate shuttling and swapping, and bar
  heights sketch a position-dependent field profile.">
  <figcaption><b>Figure 1:</b> Scheduling-level view of ions ("Q"), processing
  zones, transport operations, and a site-dependent field profile.</figcaption>
</figure>

Each ion occupies one site, and two ions cannot occupy the same site. Empty
sites provide the space needed to rearrange the chain. A shuttle moves an ion to
an adjacent empty site. An adjacent ion swap represents a hardware-supported
reordering operation whose detailed physical realization is left to the device.

This discrete description is intentionally more abstract than a segmented-trap
layout. A site can be understood as a stable location relevant to scheduling,
not necessarily as one electrode segment or one fixed physical distance.

## Processing zones

Processing zones identify contiguous groups of sites with gate-control access.
Sites outside these zones act as memory or transport locations. A device may
have one processing zone or several independently available zones.

In the current model:

- a physical one-qubit gate requires its ion to be in a processing zone;
- a two-qubit gate requires both ions to occupy distinct sites in the same
  processing zone; and
- a processing zone is reserved while a physical gate is running.

Two ions in the same processing zone do not additionally need to occupy
neighboring sites. This treats the zone as the relevant interaction region and
allows its internal control mechanism to mediate a gate across any pair of its
sites. Architectures that require stricter locality would need a more specific
zone or gate-validity model.

Separate processing zones may operate concurrently when their ions and other
resources do not conflict. This captures the main scheduling benefit of a
multi-zone device without prescribing how laser beams, optical channels, or
shared control electronics are allocated below that level.

## Time and concurrent operations

Time is divided into integer compiler timesteps. Every physical operation has a
configured duration, and participating ions or processing zones remain busy
until that duration has elapsed. Operations may start at the same timestep when
they do not compete for the same modeled resources.

The timestep is an architecture-defined scheduling unit rather than a fixed
physical duration. A hardware description might choose one timestep to represent
a microsecond, a transport clock period, or another convenient unit, provided
all configured durations use the same scale.

This model captures resource occupancy and makespan, but it does not by itself
model continuously varying transport speed, pulse overlap, calibration drift, or
uncertainty in operation time.

## Physical and virtual rotations

The default hardware catalog provides `Rx`, `Ry`, `Rz`, and `Rzz`, a common
trapped-ion gate set. `Rxx` and `Ryy` gates are also implemented and can be
enabled explicitly for different hardware models.

`Rx` and `Ry` rotations are physical by default: they occupy their ion and its
processing zone for the configured gate duration. `Rz` is virtual by default,
reflecting the common control convention of implementing a z rotation as a
change of phase reference rather than as an additional driven pulse.

A virtual rotation remains part of the logical schedule and is fully tracked as
a circuit operation, but it takes zero compiler time and does not reserve a
processing zone. Virtuality is configurable rather than inferred from the gate
name or duration, allowing a hardware model to describe physical z rotations or
virtual rotations about another axis when appropriate.

The distinction is useful because a zero-duration physical operation and a
virtual frame update are not the same statement about the hardware. The former
still requires the ion to be in an eligible control region; the latter does not
consume a physical gate resource.

## Site-dependent field profile

An architecture may attach a scalar field value to every site. This can
represent a spatially varying sensitivity or frequency-shift profile used by
later dephasing and control analyses. The profile travels with the architecture
and compiled result, but it does not influence the base schedule search.

Field values retain the scale supplied by the user. They are not automatically
RMS-normalized or otherwise calibrated by the hardware model. Normalization, a
noise-strength convention, and the physical interpretation of the scalar values
belong to the downstream analysis using the profile.

## See also

- {doc}`linear_compiler` — compile Qiskit and QASM circuits for this model
- {doc}`linear_design` — software boundaries and extension points
- {doc}`references` — publications underlying the QCCD and shuttling model
