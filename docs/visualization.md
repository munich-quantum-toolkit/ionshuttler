# Visualization

{py:func}`mqt.ionshuttler.visualize` shows a compilation result as an
interactive view. A {doc}`Grid <grid_compilers>` result gives a
{py:class}`~mqt.ionshuttler.visualization.GridView`, and a
{doc}`Linear <linear_compiler>` result gives a
{py:class}`~mqt.ionshuttler.visualization.LinearView`. Both views use the same
player, display settings, colors, and export functions. This page describes
these shared parts first and then the drawing of each backend.

## Show a result

In a notebook, the view shows itself as the cell output. Elsewhere, save it as
an HTML file and open it in a browser. The file contains all data and needs no
network access.

```python
from mqt.ionshuttler import visualize

view = visualize(result)
view.save("schedule.html")
```

## Player controls

The view draws the schedule on a canvas. Use **Play**, **Previous layer**,
**Next layer**, and the time slider to move through the schedule. Simultaneous
actions move together. Running gates highlight their ions and processing zone.
The header shows the time and the actions of the running layer.

The tool bar sets the playback speed, the theme, the ion colors, and the labels.
A side-bar button opens details that follow the playback: **Compiler planning**
for a Grid result with a planning trace and **Schedule details** for a Linear
result.

**Edit layout** changes the drawing, also during playback:

- The **Size** slider scales ions and hardware markers.
- Click an ion or a processing zone to change its color.
- In a Grid view, drag a junction to move it. Segments and ions follow at once.
- **Copy settings** copies the changed settings as Python keyword arguments for
  the visualizer.
- **Reset** restores the original layout.

## Result viewer

The result viewer is one browser page for all results. It runs in your Python
process and listens only on this computer. Python replays each result; the page
draws it.

```python
from mqt.ionshuttler import visualize
from mqt.ionshuttler.visualization import open_viewer

visualize(result).open()  # show one result
open_viewer()  # open the viewer without a result
```

Use **Open result…** in the page, or drop a file on it, to show a result saved
with `result.save("result.json")`. The **View** menu switches between all
results of the session. In a notebook, `open` returns at once and the viewer
runs as long as the kernel. In a plain script, `open` waits until you press
Ctrl+C, so that the viewer keeps running. Under WSL, the viewer opens in the
Windows browser. The returned address also opens the viewer by hand.

## Display settings

{py:class}`~mqt.ionshuttler.visualization.GridVisualizer` and
{py:class}`~mqt.ionshuttler.visualization.LinearVisualizer` hold the display
settings. The browser controls start from these values. `visualize(result)` uses
the default settings.

```python
from mqt.ionshuttler.visualization import GridVisualizer, LinearVisualizer

grid_view = GridVisualizer(theme="light", timesteps_per_second=8).visualize(grid_result)
linear_view = LinearVisualizer(theme="light", show_ion_labels=False).visualize(linear_result)
```

Both visualizers accept these settings:

| Setting                                | Meaning                                                                    |
| -------------------------------------- | -------------------------------------------------------------------------- |
| `theme`                                | `"dark"` (the default), `"light"`, or `"auto"`, which follows the browser. |
| `show_ion_labels`                      | Write each ion ID inside its circle.                                       |
| `show_processing_zone_labels`          | Write processing-zone IDs.                                                 |
| `timesteps_per_second`                 | Playback and video speed in schedule timesteps per second.                 |
| `width`, `height`                      | View and video size in pixels.                                             |
| `marker_size`                          | Size of ions and hardware markers relative to the automatic size.          |
| `ion_colors`, `processing_zone_colors` | Ion and processing-zone colors, described in the next section.             |
| `frames_per_second`                    | Video frame rate.                                                          |
| `video_start_time`, `video_end_time`   | First and last timestep of exported videos.                                |

Figures and videos use the same theme as the view. `"auto"` draws them light.
Pass `theme="light"` for print. The sections [Grid results](#grid-results) and
[Linear results](#linear-results) describe the settings of each backend.

## Ion and processing-zone colors

Each ion is drawn as a circle with a colored ring. By default, every ion has its
own ring color. Each processing zone also has its own color. A running gate
highlights its ions and its processing zone in that color. `ion_colors="single"`
gives all ions one ring color. A mapping sets the colors of chosen ions, for
example to show the role of each ion. A color can change at a schedule time, and
a label adds a legend entry:

```python
from mqt.ionshuttler.visualization import GridVisualizer, IonColor

data = IonColor(border="tab:blue", label="data")
idle = IonColor(border="#94a3b8", label="ancilla")
active = IonColor(border="crimson", fill="#fee2e2", label="active ancilla")
visualizer = GridVisualizer(
    ion_colors={0: data, 1: data, 2: [(0, idle), (120, active)]},
    processing_zone_colors={"pz": "teal"},
)
```

`LinearVisualizer` accepts the same arguments. Colors accept any Matplotlib
color. Ions without an entry have a gray ring. The **Ion colors** switch in the
view changes between distinct, single, and custom colors. Colors changed in
**Edit layout** are part of **Copy settings**.

## Video export

Open **Export video** in the view and drag the two handles to select the first
and last timestep. The export uses the current theme, labels, speed, and frame
rate. The video shows `timesteps_per_second` timesteps per second, so the frame
rate changes only smoothness. The browser draws each frame at a fixed schedule
time and does not wait for playback. It writes a WebM file and needs a browser
with WebCodecs video encoding, such as a current Chromium-based browser.

Scripts can export a video without a browser. This uses Matplotlib with the same
layout, colors, and labels:

```python
view = GridVisualizer(theme="dark", video_start_time=315, video_end_time=420).visualize(result)
view.export_video("interesting.gif", frames_per_second=30)
view.export_video("interesting.mp4", start_time=315, end_time=360)
```

A `.gif` file needs no further software. The `.mp4`, `.m4v`, `.mov`, `.mkv`, and
`.webm` formats need the FFmpeg program.

## Static figures

`plot` draws one schedule time as a Matplotlib figure, for example for a paper.
A fractional time shows ions in motion.

```python
visualizer.plot(result, 120).savefig("t120.png", dpi=200)
```

## Comparisons

`compare` plays several results on one clock. The clock uses absolute schedule
time. A result that ends earlier keeps showing its final state. A Grid
comparison shows the results side by side; a Linear comparison stacks them one
above the other.

```python
view = visualizer.compare({"breadth-first": first, "greedy": second})
view.save("comparison.html")
```

## Grid results

The Grid view draws segments as lines between junctions. Ions that cross a
junction travel from their segment through the junction into the destination
segment. `show_hardware_ids=True` writes segment and junction IDs.

### Layout

The rectangular and square generators use junction IDs of the form
`j:row:column`. The visualizer uses those IDs to recover the generated lattice.
For other architectures, including hexagonal grids, it first attempts a planar
layout. A non-planar graph uses a deterministic force-directed layout instead.
To control the drawing, pass explicit coordinates. The mapping must contain each
junction exactly once. Keys can be junction IDs or `Junction` values.

```python
visualizer = GridVisualizer(junction_coordinates={"memory-processor": (0.0, 0.0)})
```

Coordinates belong to the visualization, not `GridArchitecture`. Changing a
drawing therefore does not change compilation, replay, or serialized hardware.
Junctions dragged in **Edit layout** are part of **Copy settings**.

Results on different architectures can use their own coordinates in a
comparison. For a comparison, **Copy settings** copies one mapping per panel.

```python
view = visualizer.compare(
    {"loop": loop_result, "square": square_result},
    junction_coordinates={"loop": loop_coordinates},
)
```

### Compiler planning

A result compiled with `GridCompilerConfig(record_planning_trace=True)` stores
what the compiler intended to do at each scheduling decision. For such a result,
the view has a **Compiler planning** button. It opens a side bar next to the
drawing. The side bar and the marks on the drawing show the latest decision at
or before the current time:

- The side bar lists the selected and lookahead gates with their ions and target
  processing zone, the prioritized ions with the reason why each one moves or
  stays, the retained ions, the selected moves, and the ready gates. Each ion
  appears in its ring color, and each processing zone in its zone color.
- Target segments glow. Prioritized ions have a dashed ring, their priority
  rank, and a dotted line to their target segment, all in the ion's ring color.
- A square marks each ion that must stay in its processing zone.
- Solid arrows show the selected moves. Dashed arrows show moves that the
  compiler proposed and rejected.
- At the end of a failed compilation, the side bar states why the compiler
  stopped.

`GridVisualizer(show_compiler_planning=True)` starts the view with the side bar
open. Results without a trace have no button. Matplotlib figures and videos do
not draw the marks.

## Linear results

The Linear view plays the schedule on the ion chain. The trap sites lie on one
line, and processing zones color their part of the line. Shuttled ions slide to
their new site. Swapped ions pass each other, one on each side of the line.
Global gates light up all their ions. `show_site_numbers=False` hides the site
numbers.

For an architecture with a field profile, the **Field profile** switch, or
`show_field_profile=True`, draws the profile as a faint curve behind the chain.

### Layout

A chain lies on a straight line. A ring (`topology="ring"`) lies on an ellipse
that fills the panel, with site 0 at the top and the other sites in clockwise
order. Ions that move between the last site and site 0 use the edge that joins
them, and a processing zone across this edge stays in one piece. Swapping ions
pass on the inside and the outside of the ring, and gate names appear outside
it.

To control the drawing, pass explicit coordinates: a mapping from site number to
`(x, y)` or a sequence of positions in site order, with `y` pointing up. The
coordinates must name every site exactly once. Straight lines join neighboring
sites. The drawing keeps the shape of the coordinates and scales it to fit the
panel.

```python
visualizer = LinearVisualizer(site_coordinates=[(0, 0), (1, 1), (2, 0), (3, 1)])
```

As for Grid results, coordinates belong to the visualization, not
`LinearArchitecture`. In a comparison, `compare(..., site_coordinates=...)`
gives chosen panels their own coordinates, keyed by panel title. Rings and
explicit coordinates do not draw the field profile.

### Schedule details

**Schedule details** opens a side bar that follows the playback. It lists the
running gates with their ions and processing zones, the next gates of the
schedule, the moving ions, each ion's site, the outcome of the compilation, and
the run statistics that the compiler reports, such as the search score. When the
compiler assigned gates to processing zones before scheduling
(`pre_partition=True`), the side bar names each gate that ran in another zone
and counts the gates that kept their planned zone.
`LinearVisualizer(show_schedule_details=True)` starts the view with the side bar
open.

### Ion trajectories

{py:meth}`~mqt.ionshuttler.visualization.LinearVisualizer.plot_trajectories`
draws a different figure: time on the horizontal axis and sites on the vertical
axis. Each ion follows a colored trajectory, transport appears as movement
between sites, processing zones form horizontal bands, and markers identify gate
applications. Long schedules wrap onto consecutive 25-timestep rows with major
ticks every five timesteps. A partial final row is proportionally narrower so
that every row uses the same timestep scale. For a ring, dotted lines mark the
boundary between the last site and site zero. A transport across this boundary
leaves and re-enters the plot instead of crossing every intermediate site. The
figure uses Matplotlib's own style, not the display settings.

```python
from mqt.ionshuttler.visualization import LinearVisualizer

LinearVisualizer.plot_trajectories(result).savefig("trajectories.png", dpi=150)
```

## See also

- {doc}`grid_compilers` — compile for Grid architectures
- {doc}`linear_compiler` — compile for Linear architectures
- {doc}`design` — how the shared player and the backend drawings fit together
- {doc}`api/mqt/ionshuttler/visualization/index` — complete visualization Python
  API reference
