# Hardware Models

Trapped-ion quantum charge-coupled devices (QCCDs) store ions in segmented
linear traps. Electrode voltages confine and move the ions; junctions connect
trap segments; and dedicated processing zones provide the control needed for
quantum gates.

Controlling this hardware spans several layers, from routing ions across a
device to executing local transport and control operations. IonShuttler uses
multiple abstraction levels to address these tasks at the scale and level of
detail they require. Each model retains the hardware properties relevant to its
compilation task and abstracts properties that belong to other layers or would
only complicate that task.

IonShuttler currently provides two hardware models at different scales. The Grid
model treats complete segments and their junctions as a routing graph. The
Linear model resolves individual sites and local transport within a trap
segment. A concrete device architecture supplies the layout, processing zones,
operations, and timing values for the chosen model.

## Grid model: a network of segments

<figure>
  <img src="_static/qccd_device.png" width="62%" alt="A QCCD architecture formed from connected trap segments and processing zones.">
  <img src="_static/graph.png" width="32%" alt="Graph representation of the QCCD architecture.">
  <figcaption><b>Segment-level model.</b> A QCCD layout becomes a graph of trap
  segments, junctions, and processing zones.</figcaption>
</figure>

The Grid model treats connected trap segments as the main storage and transport
locations. Junctions define which segments ions can move between, and processing
zones attach gate resources to this network. The model tracks motion through the
network without resolving an ion's relative position within a segment.
Performing gates requires the participating ion(s) to be (co-)located on a
segment carrying a processing zone.

This coarser level retains the topology that governs routing while reducing the
state needed to describe a large device. It is a good fit when network-scale
movement matters more than motion within one segment. A grid is one possible
architecture at this level; the abstraction itself is a graph of segments and
junctions.

The {doc}`Grid compiler guide <grid_compilers>` introduces the corresponding
compilation methods.

## Linear model: sites within a segment

<figure>
  <img src="_static/linear_hardware_model.png" width="100%" alt="Ions on a linear array with processing zones, transport arrows, and a field profile.">
  <figcaption><b>Site-level model.</b> Ions occupy discrete sites and move into
  processing zones for gates.</figcaption>
</figure>

The Linear model resolves a trap segment as an ordered sequence of sites. Each
site holds at most one ion, so empty sites provide room to rearrange an ion
chain. A shuttle moves an ion to an adjacent empty site, while a physical swap
exchanges adjacent ions.

A processing zone contains one or more contiguous sites. A physical one-qubit
gate requires its ion in a processing zone. A physical multi-qubit gate requires
all participating ions in the same zone. Operations reserve their ions and
hardware resources for an architecture-defined number of timesteps.

This detailed level exposes motion and concurrency within a segment. It is a
good fit when local ion order and transport affect the result. The
{doc}`Linear compiler guide <linear_compiler>` shows how to define a concrete
architecture with the site-level model. The site-level model omits crystal
splitting and merging. Their overhead is included in the surrounding transport
operations.

## Shared scheduling assumptions

Both models use the same broad QCCD picture: ions move through memory and
transport regions to processing zones, physical operations take time, and
operations may overlap only when they do not compete for modeled resources. In
both models, time is discretized into timesteps and all operations take an
integer number of timesteps to complete. Each architecture supplies the concrete
layout and timing values.

## See also

- {doc}`design` — compiler and architecture responsibilities
- {doc}`references` — publications behind the QCCD and shuttling methods
