// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// Draw Grid results for the schedule player.
//
// Python replays the schedule and sends each ion's movements as places: a
// segment and a fraction of the way along it. This script turns places into
// coordinates from the current junction positions, so moving a junction moves
// every segment and ion attached to it. It never decides whether an action is
// legal. The Matplotlib renderer in `_matplotlib.py` uses the same layout rules.

"use strict";

const GridDrawing = (() => {
  const ION_RADIUS_PER_SPACING = 0.38;
  const MAX_ION_RADIUS = 0.06;
  const MARGIN_PER_ION_RADIUS = 2.5;
  // Colors and text sizes that every drawing shares. Node.js tests load this
  // file without the page, so they read the shared rules from their file.
  const {
    AMBIENT_GLOW,
    FONT,
    ZONE_PALETTE,
    distinctColor,
    ionColors,
    ionLabelFontSize,
    legendEntries,
    paintBackground,
    paintGlow,
    paintLineGlow,
    ringStroke,
    textColor,
    truncate,
  } =
    typeof PlayerStyle === "object"
      ? PlayerStyle
      : require("../_player/style.js");

  function zoneColor(view, panel, zone) {
    return view.zoneEdits[zone] ?? panel.zoneColors[zone];
  }

  // ---------------------------------------------------------------------------
  // Drawing data and time queries
  // ---------------------------------------------------------------------------

  function place(segment, fraction) {
    return { segment, fraction };
  }

  function preparePanel(raw) {
    const ions = raw.ions.map(([ion, segment, fraction, movements]) => ({
      ion,
      initial: place(segment, fraction),
      movements: movements.map((item) => ({
        start: item[0],
        end: item[1],
        target: place(item[2], item[3]),
        via: item.length > 4 ? item[4] : null,
      })),
      starts: movements.map((item) => item[0]),
    }));
    const layers = raw.layers.map(([start, end, description, gates]) => ({
      start,
      end,
      description,
      gates: gates.map(([gateIons, zone, gateStart, gateEnd]) => ({
        ions: gateIons,
        zone,
        start: gateStart,
        end: gateEnd,
      })),
    }));
    const gates = layers.flatMap((layer) => layer.gates);
    const end = ([junction, x, y]) => ({ junction, offset: [x, y] });
    const junctions = raw.junctions.map(([, x, y]) => [x, y]);
    return {
      title: raw.title,
      start: raw.start,
      end: raw.end,
      planning:
        raw.planning === undefined ? null : preparePlanning(raw.planning),
      junctionIds: raw.junctions.map(([id]) => id),
      junctions,
      originalJunctions: junctions.map((point) => [...point]),
      segments: raw.segments.map(([id, first, second, capacity, zones]) => ({
        id,
        start: end(first),
        end: end(second),
        capacity,
        zones,
      })),
      zones: raw.processing_zones.map(([id, segment, fraction]) => ({
        id,
        place: place(segment, fraction),
      })),
      ions,
      ionIndex: new Map(ions.map((track, index) => [track.ion, index])),
      layers,
      layerStarts: layers.map((layer) => layer.start),
      gates,
      gateStarts: gates.map((gate) => gate.start),
      longestGate: gates.reduce(
        (longest, gate) => Math.max(longest, gate.end - gate.start),
        0,
      ),
      frozen: null,
      layout: null,
    };
  }

  // The compiler planning trace in the layout of `PlanningTrace.to_dict` in
  // `_scene.py`. Segments are segment indices.
  function preparePlanning(raw) {
    const crossings = (items) =>
      items.map(([ions, source, destination]) => ({
        ions,
        source,
        destination,
      }));
    const snapshots = raw.snapshots.map((item) => ({
      iteration: item[0],
      time: item[1],
      ready: item[2],
      selected: item[3],
      lookahead: item[4],
      started: item[5],
      targets: new Map(item[6]),
      requests: item[7].map(([ion, target, status, moves]) => ({
        ion,
        target,
        status,
        moves: crossings(moves),
      })),
      retained: item[8],
      moves: crossings(item[9]),
      idle: item[10],
    }));
    return {
      stopReason: raw.stop_reason,
      gates: new Map(raw.gates.map(([id, name, ions]) => [id, { name, ions }])),
      snapshots,
      times: snapshots.map((snapshot) => snapshot.time),
    };
  }

  function prepare(data) {
    const zoneOverrides = data.zone_colors ?? {};
    const panels = data.panels.map(preparePanel);
    for (const panel of panels) {
      panel.zoneColors = Object.fromEntries(
        panel.zones.map((zone, index) => [
          zone.id,
          zoneOverrides[zone.id] ?? distinctColor(index, ZONE_PALETTE),
        ]),
      );
    }
    return {
      colors: data.colors,
      panels,
      ionChanges: data.ion_colors ?? {},
      ionEdits: {},
      zoneEdits: {},
    };
  }

  function timeBounds(view) {
    return [
      Math.min(...view.panels.map((panel) => panel.start)),
      Math.max(...view.panels.map((panel) => panel.end)),
    ];
  }

  function stepTimes(view) {
    return [...new Set(view.panels.flatMap((panel) => panel.layerStarts))].sort(
      (a, b) => a - b,
    );
  }

  function bisectRight(values, value) {
    let low = 0;
    let high = values.length;
    while (low < high) {
      const middle = (low + high) >> 1;
      if (values[middle] <= value) low = middle + 1;
      else high = middle;
    }
    return low;
  }

  function isRunning(start, end, time) {
    return (start <= time && time < end) || start === time;
  }

  function interpolate(start, end, progress) {
    return [
      start[0] + progress * (end[0] - start[0]),
      start[1] + progress * (end[1] - start[1]),
    ];
  }

  function distance(first, second) {
    return Math.hypot(second[0] - first[0], second[1] - first[1]);
  }

  // Same rules as `geometry` in `_scene.py`.
  function geometry(panel) {
    const locate = (end) => {
      if (end.junction === null) return end.offset;
      const [x, y] = panel.junctions[end.junction];
      return [x + end.offset[0], y + end.offset[1]];
    };
    const segments = panel.segments.map((segment) => [
      locate(segment.start),
      locate(segment.end),
    ]);
    const points = [...panel.junctions, ...segments.flat()];
    const xs = points.map((point) => point[0]);
    const ys = points.map((point) => point[1]);
    const [minX, maxX, minY, maxY] = [
      Math.min(...xs),
      Math.max(...xs),
      Math.min(...ys),
      Math.max(...ys),
    ];
    const size = Math.max(maxX - minX, maxY - minY) || 1;
    const spacings = panel.segments
      .map(
        (segment, index) =>
          distance(...segments[index]) / (segment.capacity + 1),
      )
      .filter((spacing, index) => distance(...segments[index]) > 0);
    const ionRadius = spacings.length
      ? Math.min(
          ION_RADIUS_PER_SPACING * Math.min(...spacings),
          MAX_ION_RADIUS * size,
        )
      : size / 40;
    const margin = MARGIN_PER_ION_RADIUS * ionRadius;
    return {
      junctions: panel.junctions,
      segments,
      bounds: [minX - margin, minY - margin, maxX + margin, maxY + margin],
      ionRadius,
    };
  }

  function position(shape, where) {
    const [start, end] = shape.segments[where.segment];
    return interpolate(start, end, where.fraction);
  }

  function alongPath(source, via, target, progress) {
    if (via === null) return interpolate(source, target, progress);
    const first = distance(source, via);
    const second = distance(via, target);
    const total = first + second;
    if (total === 0) return target;
    const travelled = progress * total;
    if (travelled <= first)
      return first > 0 ? interpolate(source, via, travelled / first) : via;
    return interpolate(via, target, (travelled - first) / second);
  }

  function trackPosition(track, time, shape) {
    const index = bisectRight(track.starts, time) - 1;
    if (index < 0) return position(shape, track.initial);
    const movement = track.movements[index];
    const target = position(shape, movement.target);
    if (time >= movement.end) return target;
    const source = position(
      shape,
      index === 0 ? track.initial : track.movements[index - 1].target,
    );
    const via = movement.via === null ? null : shape.junctions[movement.via];
    const progress = (time - movement.start) / (movement.end - movement.start);
    return alongPath(source, via, target, progress);
  }

  function runningGates(panel, time) {
    const running = [];
    let index = bisectRight(panel.gateStarts, time);
    const earliestStart = time - panel.longestGate;
    while (index > 0 && panel.gates[index - 1].start >= earliestStart) {
      index -= 1;
      const gate = panel.gates[index];
      if (isRunning(gate.start, gate.end, time)) running.push(gate);
    }
    return running.reverse();
  }

  function currentLayer(panel, time) {
    const index = bisectRight(panel.layerStarts, time) - 1;
    if (index < 0) return null;
    const layer = panel.layers[index];
    return isRunning(layer.start, layer.end, time) ? layer : null;
  }

  // The latest planning snapshot at or before a schedule time. Several
  // decisions can share one timestep; the last of them is shown. Same rule as
  // `PlanningTrace.snapshot_at` in `_scene.py`.
  function planningSnapshot(panel, time) {
    if (panel.planning === null) return null;
    const index = bisectRight(panel.planning.times, time) - 1;
    return index < 0 ? null : panel.planning.snapshots[index];
  }

  // ---------------------------------------------------------------------------
  // Layout
  // ---------------------------------------------------------------------------

  // Same rules as `panel_layout` in `_scene.py`.
  function panelLayout(shape, left, top, width, height, markerSize = 1) {
    const fontSize = Math.max(
      11,
      Math.min(17, Math.round(Math.min(width, height) * 0.026)),
    );
    const headerHeight = Math.round(fontSize * 3.6);
    const padding = Math.round(Math.min(width, height) * 0.03);
    const availableWidth = Math.max(width - 2 * padding, 1);
    const availableHeight = Math.max(height - headerHeight - 2 * padding, 1);
    const [minX, minY, maxX, maxY] = shape.bounds;
    const extentX = Math.max(maxX - minX, 1e-9);
    const extentY = Math.max(maxY - minY, 1e-9);
    const scale = Math.min(availableWidth / extentX, availableHeight / extentY);
    const origin = [
      left + padding + (availableWidth - extentX * scale) / 2,
      top + headerHeight + padding + (availableHeight - extentY * scale) / 2,
    ];
    return {
      left,
      top,
      width,
      height,
      scale,
      fontSize,
      headerHeight,
      padding,
      // At least 2 px, so that every ion stays visible.
      ionRadius: Math.max(
        Math.min(shape.ionRadius * scale, 0.037 * Math.min(width, height)) *
          markerSize,
        2,
      ),
      point: ([x, y]) => [
        origin[0] + (x - minX) * scale,
        origin[1] + (maxY - y) * scale,
      ],
      coordinate: ([x, y]) => [
        minX + (x - origin[0]) / scale,
        maxY - (y - origin[1]) / scale,
      ],
    };
  }

  function labelNormal(start, end) {
    const dx = end[0] - start[0];
    const dy = end[1] - start[1];
    const length = Math.hypot(dx, dy);
    let normal = length === 0 ? [0, 1] : [-dy / length, dx / length];
    if (normal[1] < 0 || (normal[1] === 0 && normal[0] > 0))
      normal = [-normal[0], -normal[1]];
    return normal;
  }

  function headerLines(panel, time) {
    const shownTime = Math.min(Math.max(time, panel.start), panel.end);
    const layer = currentLayer(panel, shownTime);
    return [
      panel.title,
      `t ${shownTime.toFixed(1)} / ${panel.end}`,
      layer === null ? "" : layer.description,
    ];
  }

  // ---------------------------------------------------------------------------
  // Drawing
  // ---------------------------------------------------------------------------

  function drawAtTime(context, view, time, settings, width, height) {
    const colors = view.colors[settings.theme];
    context.save();
    paintBackground(context, 0, 0, width, height, colors);
    const panelWidth = width / view.panels.length;
    view.panels.forEach((panel, index) => {
      const shape = geometry(panel);
      if (panel.frozen !== null) Object.assign(shape, panel.frozen);
      const layout = panelLayout(
        shape,
        index * panelWidth,
        0,
        panelWidth,
        height,
        settings.marker_size ?? 1,
      );
      if (index > 0) {
        context.fillStyle = colors.chip;
        context.fillRect(
          Math.round(index * panelWidth),
          layout.padding,
          1,
          height - 2 * layout.padding,
        );
      }
      panel.layout = layout;
      drawPanel(context, view, panel, shape, layout, time, settings, colors);
    });
    context.restore();
  }

  function drawPanel(
    context,
    view,
    panel,
    shape,
    layout,
    time,
    settings,
    colors,
  ) {
    const shownTime = Math.min(Math.max(time, panel.start), panel.end);
    const gates = runningGates(panel, shownTime);
    // Gates that take no time, such as virtual rz gates, get no glow.
    const gateGlows = new Map();
    for (const gate of gates) {
      if (gate.end === gate.start) continue;
      const glow =
        gate.zone === null ? colors.gate : zoneColor(view, panel, gate.zone);
      for (const ion of gate.ions) gateGlows.set(ion, glow);
    }
    const activeZones = new Set(
      gates.map((gate) => gate.zone).filter((zone) => zone !== null),
    );
    const radius = layout.ionRadius;
    const segments = shape.segments.map(([start, end]) => [
      layout.point(start),
      layout.point(end),
    ]);

    context.lineCap = "round";
    panel.segments.forEach((segment, index) => {
      const color =
        segment.zones.length > 0
          ? zoneColor(view, panel, segment.zones[0])
          : colors.segment;
      const active = segment.zones.some((zone) => activeZones.has(zone));
      const [[x0, y0], [x1, y1]] = segments[index];
      const width = segmentWidth(segment, radius);
      // Processing zones glow faintly in their own color, and fully while
      // they run a gate.
      if (segment.zones.length > 0) {
        const strength = active ? 1 : AMBIENT_GLOW;
        paintLineGlow(context, x0, y0, x1, y1, width, color, strength);
      }
      context.strokeStyle = color;
      context.lineWidth = width;
      line(context, x0, y0, x1, y1);
    });
    const showPlanning =
      settings.show_compiler_planning === true && panel.planning !== null;
    const snapshot = showPlanning ? planningSnapshot(panel, shownTime) : null;
    if (snapshot !== null)
      drawPlanningTargets(context, panel, snapshot, segments, radius, colors);

    const zoneMarker = 0.75 * radius;
    for (const zone of panel.zones) {
      const [x, y] = layout.point(position(shape, zone.place));
      context.fillStyle = zoneColor(view, panel, zone.id);
      context.beginPath();
      context.moveTo(x, y - zoneMarker);
      context.lineTo(x + zoneMarker, y);
      context.lineTo(x, y + zoneMarker);
      context.lineTo(x - zoneMarker, y);
      context.closePath();
      context.fill();
    }
    if (settings.show_processing_zone_labels) {
      const labelSize = zoneLabelSize(layout.fontSize);
      for (const zone of panel.zones) {
        const [start, end] = segments[zone.place.segment];
        const normal = labelNormal(...shape.segments[zone.place.segment]);
        context.fillStyle = zoneColor(view, panel, zone.id);
        context.font = `600 ${labelSize}px ${FONT}`;
        parallelText(
          context,
          zone.id,
          layout.point(position(shape, zone.place)),
          start,
          end,
          [normal[0], -normal[1]],
          zoneLabelGap(radius, labelSize),
        );
      }
    }

    const junctionSide = 0.95 * radius;
    context.fillStyle = colors.junction;
    for (const point of shape.junctions) {
      const [x, y] = layout.point(point);
      roundedSquare(context, x, y, junctionSide);
      context.fill();
    }
    if (settings.show_hardware_ids || settings.editing) {
      drawHardwareIds(
        context,
        panel,
        shape,
        segments,
        layout,
        colors,
        settings.show_hardware_ids,
      );
    }
    if (settings.editing)
      drawHandles(context, panel, shape, layout, colors, settings.highlight);

    const points = panel.ions.map((track) =>
      layout.point(trackPosition(track, shownTime, shape)),
    );
    panel.ionPoints = points;
    const styles = panel.ions.map((track) =>
      ionColors(
        view,
        track.ion,
        shownTime,
        settings.ion_colors,
        colors.ion,
        colors.ion_fill,
      ),
    );
    // Every ion glows faintly in its ring color. A running gate makes its
    // ions glow fully in the color of its processing zone.
    panel.ions.forEach((track, index) => {
      const [x, y] = points[index];
      const glow = gateGlows.get(track.ion);
      if (glow === undefined) {
        const halo = styles[index].border;
        paintGlow(
          context,
          x,
          y,
          0.8 * radius,
          1.6 * radius,
          halo,
          AMBIENT_GLOW,
        );
      } else paintGlow(context, x, y, 0.8 * radius, 1.9 * radius, glow);
    });
    const ringWidth = Math.max(1.5, 0.28 * radius);
    panel.ions.forEach((track, index) => {
      const [x, y] = points[index];
      const middle = radius - ringWidth / 2;
      const border = styles[index].border;
      context.fillStyle = styles[index].fill;
      context.strokeStyle = ringStroke(
        context,
        x,
        y,
        middle,
        ringWidth,
        border,
      );
      context.lineWidth = ringWidth;
      circle(context, x, y, middle);
      context.fill();
      context.stroke();
      const highlight = settings.highlight;
      if (
        highlight !== null &&
        highlight.panel === panel &&
        highlight.ion === track.ion
      ) {
        context.strokeStyle = colors.text;
        context.lineWidth = 1.5;
        circle(context, x, y, radius + 3);
        context.stroke();
      }
    });
    if (settings.show_ion_labels) {
      context.textAlign = "center";
      context.textBaseline = "middle";
      panel.ions.forEach((track, index) => {
        const label = `q${track.ion}`;
        const labelSize = ionLabelFontSize(radius, label);
        if (labelSize < 6) return;
        context.fillStyle = styles[index].text;
        context.font = `600 ${labelSize}px ${FONT}`;
        context.fillText(
          label,
          points[index][0],
          points[index][1] + labelSize * 0.04,
        );
      });
    }
    if (snapshot !== null)
      drawPlanningMarks(
        context,
        panel,
        snapshot,
        points,
        styles,
        segments,
        layout,
        colors,
      );
    if (settings.ion_colors === "custom")
      drawLegend(context, view, layout, colors);
    drawHeader(context, panel, layout, time, colors);
  }

  // Same rule as `zone_label_size` in `_scene.py`.
  function zoneLabelSize(fontSize) {
    return Math.max(11, 0.95 * fontSize);
  }

  // Distance from a processing zone to its label, clear of the gate glow.
  // Same rule as `_zone_label_gap` in `_matplotlib.py`.
  function zoneLabelGap(radius, labelSize) {
    return 1.75 * radius + 0.5 * labelSize + 2;
  }

  // Same rules as `_segment_width` in `_matplotlib.py`.
  function segmentWidth(segment, radius) {
    return segment.zones.length === 0
      ? Math.max(1.5, 0.45 * radius)
      : Math.max(2, 0.7 * radius);
  }

  function roundedSquare(context, x, y, side) {
    const half = side / 2;
    const corner = side / 4;
    context.beginPath();
    context.moveTo(x - half + corner, y - half);
    context.arcTo(x + half, y - half, x + half, y + half, corner);
    context.arcTo(x + half, y + half, x - half, y + half, corner);
    context.arcTo(x - half, y + half, x - half, y - half, corner);
    context.arcTo(x - half, y - half, x + half, y - half, corner);
    context.closePath();
  }

  function drawLegend(context, view, layout, colors) {
    const size = Math.max(9, 0.8 * layout.fontSize);
    const entries = legendEntries(view, colors.ion_fill).reverse();
    context.font = `400 ${size}px ${FONT}`;
    context.textAlign = "left";
    context.textBaseline = "middle";
    entries.forEach((entry, index) => {
      const x = layout.left + layout.padding + size / 2;
      const y =
        layout.top +
        layout.height -
        layout.padding -
        index * 1.6 * size -
        size / 2;
      context.fillStyle = entry.fill;
      context.strokeStyle = entry.border;
      context.lineWidth = 1.5;
      circle(context, x, y, size / 2 - 1);
      context.fill();
      context.stroke();
      context.fillStyle = colors.text;
      context.fillText(entry.label, x + size, y);
    });
  }

  // Write text along a segment, moved away from it along `normal` (pixels, y
  // down). The text stays readable from left to right.
  function parallelText(context, text, anchor, start, end, normal, gap) {
    let angle = Math.atan2(end[1] - start[1], end[0] - start[0]);
    if (angle > Math.PI / 2) angle -= Math.PI;
    else if (angle <= -Math.PI / 2) angle += Math.PI;
    context.save();
    context.translate(anchor[0] + normal[0] * gap, anchor[1] + normal[1] * gap);
    context.rotate(angle);
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(text, 0, 0);
    context.restore();
  }

  function drawHeader(context, panel, layout, time, colors) {
    const [title, timeText, description] = headerLines(panel, time);
    const fontSize = layout.fontSize;
    const left = layout.left + layout.padding;
    const right = layout.left + layout.width - layout.padding;
    const top = layout.top + layout.padding * 0.6;
    context.textBaseline = "top";
    context.font = `400 ${0.9 * fontSize}px ${FONT}`;
    const chipWidth = context.measureText(timeText).width + fontSize;
    context.fillStyle = colors.muted_text;
    context.textAlign = "right";
    context.fillText(timeText, right, top);
    context.textAlign = "left";
    if (title) {
      context.font = `600 ${fontSize}px ${FONT}`;
      context.fillText(
        truncate(title, right - left - chipWidth - fontSize, fontSize),
        left,
        top,
      );
    }
    // Without a title, the action line moves up beside the time chip.
    const descriptionTop = title
      ? top + 1.55 * fontSize
      : top + 0.12 * fontSize;
    const descriptionWidth = title
      ? right - left
      : right - left - chipWidth - fontSize;
    context.font = `400 ${0.85 * fontSize}px ${FONT}`;
    context.fillStyle = colors.muted_text;
    context.fillText(
      truncate(description, descriptionWidth, 0.85 * fontSize),
      left,
      descriptionTop,
    );
  }

  // Segment IDs run along their segment, on the side away from the
  // processing-zone labels. Junction IDs go above and left of their junction.
  function drawHardwareIds(
    context,
    panel,
    shape,
    segments,
    layout,
    colors,
    showSegments,
  ) {
    const radius = layout.ionRadius;
    // IDs shrink with the ions, so that dense layouts stay readable.
    // Same rule as `hardware_id_font_size` in `_scene.py`.
    const fontSize = Math.max(
      7,
      Math.min(0.72 * layout.fontSize, 1.1 * radius),
    );
    context.fillStyle = colors.muted_text;
    context.font = `400 ${fontSize}px ${FONT}`;
    if (showSegments) {
      panel.segments.forEach((segment, index) => {
        const [start, end] = segments[index];
        const normal = labelNormal(...shape.segments[index]);
        const middle = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2];
        parallelText(
          context,
          segment.id,
          middle,
          start,
          end,
          [-normal[0], normal[1]],
          0.6 * radius + 0.6 * fontSize,
        );
      });
    }
    context.textAlign = "right";
    context.textBaseline = "bottom";
    shape.junctions.forEach((point, index) => {
      const [x, y] = layout.point(point);
      context.fillText(
        panel.junctionIds[index],
        x - 0.7 * radius,
        y - 0.7 * radius,
      );
    });
  }

  function drawHandles(context, panel, shape, layout, colors, highlight) {
    const size = Math.max(4, 0.8 * layout.ionRadius);
    shape.junctions.forEach((point, index) => {
      const [x, y] = layout.point(point);
      const active =
        highlight !== null &&
        highlight.kind === "junction" &&
        highlight.panel === panel &&
        highlight.junction === index;
      context.lineWidth = active ? 3 : 2;
      context.strokeStyle = colors.ion;
      context.fillStyle = colors.background;
      context.globalAlpha = active ? 1 : 0.85;
      circle(context, x, y, active ? size * 1.25 : size);
      context.fill();
      context.stroke();
      context.globalAlpha = 1;
    });
  }

  function line(context, x0, y0, x1, y1) {
    context.beginPath();
    context.moveTo(x0, y0);
    context.lineTo(x1, y1);
    context.stroke();
  }

  function circle(context, x, y, radius) {
    context.beginPath();
    context.arc(x, y, radius, 0, 2 * Math.PI);
  }

  // ---------------------------------------------------------------------------
  // Compiler planning overlay
  // ---------------------------------------------------------------------------

  // Texts for the values of `MoveStatus`, `IdleReason`, and `GridStopReason`
  // in the Grid compiler.
  const MOVE_STATUS_TEXT = {
    selected: "moves",
    in_gate: "in gate",
    at_target: "at target",
    no_route: "no route",
    no_candidate: "no legal move",
    already_selected: "already in layer",
    conflict: "conflict",
    displaces_higher_priority: "would displace higher priority",
  };
  const IDLE_TEXT = {
    waiting_for_resource: "waiting for a busy resource",
    no_legal_action: "nothing legal, nothing to wait for",
    no_route: "route search found no route",
  };
  const STOP_TEXT = {
    no_progress: "no legal progress",
    iteration_limit: "iteration limit reached",
  };

  function segmentMiddle(segments, index) {
    const [[x0, y0], [x1, y1]] = segments[index];
    return [(x0 + x1) / 2, (y0 + y1) / 2];
  }

  // Under the ions: the target segments of prioritized ions and selected gates.
  function drawPlanningTargets(
    context,
    panel,
    snapshot,
    segments,
    radius,
    colors,
  ) {
    const targets = new Set(snapshot.requests.map((request) => request.target));
    for (const id of snapshot.selected) {
      const target = snapshot.targets.get(id);
      if (target !== null && target !== undefined) targets.add(target);
    }
    context.save();
    context.strokeStyle = colors.planning;
    context.globalAlpha = 0.35;
    for (const index of targets) {
      const [[x0, y0], [x1, y1]] = segments[index];
      context.lineWidth = 2.6 * segmentWidth(panel.segments[index], radius);
      line(context, x0, y0, x1, y1);
    }
    context.restore();
  }

  // Over the ions: selected gates, prioritized ions, retained ions, and moves.
  function drawPlanningMarks(
    context,
    panel,
    snapshot,
    points,
    styles,
    segments,
    layout,
    colors,
  ) {
    const radius = layout.ionRadius;
    const point = (ion) => points[panel.ionIndex.get(ion)];
    const ring = (ion) => styles[panel.ionIndex.get(ion)].border;
    context.save();
    context.lineWidth = 1.5;
    // A dashed line joins the ions of each selected gate that does not start yet.
    context.strokeStyle = colors.gate;
    context.setLineDash([4, 4]);
    for (const id of snapshot.selected) {
      const gate = panel.planning.gates.get(id);
      if (
        gate === undefined ||
        gate.ions.length < 2 ||
        snapshot.started.includes(id)
      )
        continue;
      context.beginPath();
      gate.ions.forEach((ion, index) => {
        const [x, y] = point(ion);
        if (index === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      });
      context.stroke();
    }
    // Prioritized ions: a dashed ring, the priority rank, and a dotted line to
    // the target segment, all in the ion's ring color. The side bar shows the
    // ion in the same color.
    const rankSize = Math.max(8, 0.62 * layout.fontSize);
    context.font = `600 ${rankSize}px ${FONT}`;
    context.textAlign = "left";
    context.textBaseline = "bottom";
    snapshot.requests.forEach((request, rank) => {
      const [x, y] = point(request.ion);
      const [targetX, targetY] = segmentMiddle(segments, request.target);
      context.strokeStyle = ring(request.ion);
      context.fillStyle = ring(request.ion);
      context.lineWidth = 1.5;
      context.setLineDash([3, 3]);
      circle(context, x, y, radius + 4);
      context.stroke();
      context.globalAlpha = 0.75;
      context.lineWidth = 2;
      context.setLineDash([2, 4]);
      line(context, x, y, targetX, targetY);
      context.globalAlpha = 1;
      context.fillText(String(rank + 1), x + 0.8 * radius, y - 0.8 * radius);
    });
    context.lineWidth = 1.5;
    // Ions retained in a processing zone: a square around the ion.
    context.setLineDash([]);
    context.strokeStyle = colors.text;
    for (const ion of snapshot.retained) {
      const [x, y] = point(ion);
      roundedSquare(context, x, y, 2 * radius + 10);
      context.stroke();
    }
    // Moves: proposals that the compiler rejected are dashed and muted;
    // selected moves are solid.
    const head = Math.max(6, 0.6 * radius);
    context.setLineDash([4, 3]);
    context.strokeStyle = colors.muted_text;
    context.fillStyle = colors.muted_text;
    for (const request of snapshot.requests) {
      if (request.status === "selected") continue;
      for (const move of request.moves) arrow(context, segments, move, head);
    }
    context.setLineDash([]);
    context.lineWidth = 2.5;
    context.strokeStyle = colors.planning;
    context.fillStyle = colors.planning;
    for (const move of snapshot.moves) arrow(context, segments, move, head);
    context.restore();
  }

  // An arrow from the middle of the source segment to the middle of the
  // destination segment.
  function arrow(context, segments, move, head) {
    const [x0, y0] = segmentMiddle(segments, move.source);
    const [x1, y1] = segmentMiddle(segments, move.destination);
    const angle = Math.atan2(y1 - y0, x1 - x0);
    line(context, x0, y0, x1, y1);
    context.beginPath();
    context.moveTo(x1, y1);
    context.lineTo(
      x1 - head * Math.cos(angle - 0.45),
      y1 - head * Math.sin(angle - 0.45),
    );
    context.lineTo(
      x1 - head * Math.cos(angle + 0.45),
      y1 - head * Math.sin(angle + 0.45),
    );
    context.closePath();
    context.fill();
  }

  // The side bar of the player: one section per panel with a planning trace,
  // or `null` when no panel has one.
  function details(view, time, settings) {
    const panels = view.panels.filter((panel) => panel.planning !== null);
    if (panels.length === 0) return null;
    return panels.map((panel) => {
      const shownTime = Math.min(Math.max(time, panel.start), panel.end);
      const snapshot = planningSnapshot(panel, shownTime);
      return {
        title: view.panels.length > 1 ? panel.title : "",
        rows:
          snapshot === null
            ? [["Decision", ["no decision yet"]]]
            : planningRows(view, panel, snapshot, shownTime, settings),
      };
    });
  }

  // The rows of the side bar for one snapshot, as `[label, lines]`. Ions are
  // pills framed in their ring color, as on the drawing. Target segments are
  // pills filled with the color of their processing zone.
  function planningRows(view, panel, snapshot, time, settings) {
    const planning = panel.planning;
    const colors = view.colors[settings.theme];
    const ion = (value) => ({
      text: `q${value}`,
      border: ionColors(
        view,
        value,
        time,
        settings.ion_colors,
        colors.ion,
        colors.ion_fill,
      ).border,
    });
    const target = (index) => {
      if (index === null || index === undefined) return "no target";
      const segment = panel.segments[index];
      if (segment.zones.length === 0) return { text: segment.id };
      const fill = zoneColor(view, panel, segment.zones[0]);
      return {
        text: segment.zones.join(", "),
        fill,
        border: fill,
        ink: textColor(fill),
      };
    };
    const gateLine = (id) => {
      const gate = planning.gates.get(id);
      if (gate === undefined) return `g${id}`;
      const state = snapshot.started.includes(id)
        ? ["starts"]
        : snapshot.selected.includes(id)
          ? ["selected"]
          : [];
      return [
        `g${id} ${gate.name}`,
        ...gate.ions.map(ion),
        "→",
        target(snapshot.targets.get(id)),
        ...state,
      ];
    };
    // The gates that the compiler considers: the selected and lookahead gates.
    const considered = [
      ...new Set([...snapshot.selected, ...snapshot.lookahead]),
    ];
    const rows = [
      ["Decision", [`iteration ${snapshot.iteration} · t ${snapshot.time}`]],
    ];
    if (time >= panel.end && planning.stopReason in STOP_TEXT)
      rows.push(["Stopped", [STOP_TEXT[planning.stopReason]]]);
    if (snapshot.idle !== null)
      rows.push(["No action", [IDLE_TEXT[snapshot.idle] ?? snapshot.idle]]);
    rows.push(
      ["Gates", considered.map(gateLine)],
      [
        "Priority",
        snapshot.requests.map((request, rank) => [
          `${rank + 1}.`,
          ion(request.ion),
          "→",
          target(request.target),
          MOVE_STATUS_TEXT[request.status] ?? request.status,
        ]),
      ],
      [
        "Retained",
        snapshot.retained.length === 0 ? [] : [snapshot.retained.map(ion)],
      ],
      [
        "Moves",
        snapshot.moves.map((move) => [
          ...move.ions.map(ion),
          `${panel.segments[move.source].id} → ${panel.segments[move.destination].id}`,
        ]),
      ],
      [
        "Ready",
        snapshot.ready.length === 0
          ? []
          : [snapshot.ready.map((id) => `g${id}`).join(" ")],
      ],
    );
    return rows;
  }

  // ---------------------------------------------------------------------------
  // Layout editing
  // ---------------------------------------------------------------------------

  // Return what lies under a canvas point in the last drawn frame: an ion, a
  // junction, or a processing zone, in that order.
  function pick(view, x, y) {
    for (const panel of view.panels) {
      const layout = panel.layout;
      if (layout === null) continue;
      if (x < layout.left || x > layout.left + layout.width) continue;
      const closest = (points) => {
        let best = null;
        points.forEach((point, index) => {
          const gap = Math.hypot(point[0] - x, point[1] - y);
          if (best === null || gap < best.gap) best = { index, gap };
        });
        return best;
      };
      const ion = closest(panel.ionPoints ?? []);
      if (ion !== null && ion.gap <= Math.max(8, layout.ionRadius)) {
        return { kind: "ion", panel, ion: panel.ions[ion.index].ion };
      }
      const junction = closest(
        panel.junctions.map((point) => layout.point(point)),
      );
      if (
        junction !== null &&
        junction.gap <= Math.max(12, 1.2 * layout.ionRadius)
      ) {
        return { kind: "junction", panel, junction: junction.index };
      }
      const shape = geometry(panel);
      const zone = closest(
        panel.zones.map((item) => layout.point(position(shape, item.place))),
      );
      if (zone !== null && zone.gap <= Math.max(10, layout.ionRadius)) {
        return { kind: "zone", panel, zone: panel.zones[zone.index].id };
      }
    }
    return null;
  }

  // The colors that the view can change for a picked ion or processing zone.
  function paintParts(view, handle, settings, time) {
    if (handle.kind === "ion") {
      const theme = view.colors[settings.theme];
      const style = ionColors(
        view,
        handle.ion,
        time,
        settings.ion_colors,
        theme.ion,
        theme.ion_fill,
      );
      return {
        title: `Ion q${handle.ion}`,
        parts: [
          { part: "border", label: "Ring", value: style.border },
          { part: "fill", label: "Fill", value: style.fill },
        ],
      };
    }
    if (handle.kind === "zone") {
      return {
        title: `Processing zone ${handle.zone}`,
        parts: [
          {
            part: "color",
            label: "Color",
            value: zoneColor(view, handle.panel, handle.zone),
          },
        ],
      };
    }
    return null;
  }

  // Set one color of an ion or processing zone, or clear the changes with `part === null`.
  function paint(view, handle, part, value) {
    if (handle.kind === "ion") {
      if (part === null) delete view.ionEdits[handle.ion];
      else
        view.ionEdits[handle.ion] = {
          ...view.ionEdits[handle.ion],
          [part]: value,
        };
    } else if (handle.kind === "zone") {
      if (part === null) delete view.zoneEdits[handle.zone];
      else view.zoneEdits[handle.zone] = value;
    }
  }

  // Keep the scale and position fixed while a junction moves.
  function beginDrag(view, handle) {
    const shape = geometry(handle.panel);
    handle.panel.frozen = { bounds: shape.bounds, ionRadius: shape.ionRadius };
  }

  function dragTo(view, handle, x, y) {
    const [sx, sy] = handle.panel.layout.coordinate([x, y]);
    handle.panel.junctions[handle.junction] = [
      Math.round(sx * 1e4) / 1e4,
      Math.round(sy * 1e4) / 1e4,
    ];
  }

  function endDrag(view, handle) {
    handle.panel.frozen = null;
  }

  // Undo every layout and color change made in the view.
  function resetLayout(view) {
    for (const panel of view.panels) {
      panel.junctions = panel.originalJunctions.map((point) => [...point]);
      panel.frozen = null;
    }
    view.ionEdits = {};
    view.zoneEdits = {};
  }

  // Python keyword arguments that reproduce the current layout, the marker
  // size, and the colors changed in the view. Comparisons get one coordinate
  // mapping per panel.
  function settingsText(view, settings = {}) {
    const number = (value) => String(Math.round(value * 1000) / 1000);
    const mapping = (panel, indent) =>
      [
        "{",
        ...panel.junctionIds.map(
          (id, index) =>
            `${indent}    ${JSON.stringify(id)}: (${number(panel.junctions[index][0])}, ${number(panel.junctions[index][1])}),`,
        ),
        `${indent}}`,
      ].join("\n");
    const lines =
      view.panels.length === 1
        ? [`junction_coordinates=${mapping(view.panels[0], "")},`]
        : [
            "junction_coordinates={",
            ...view.panels.map(
              (panel) =>
                `    ${JSON.stringify(panel.title)}: ${mapping(panel, "    ")},`,
            ),
            "},",
          ];
    const markerSize = settings.marker_size ?? 1;
    if (markerSize !== 1) lines.push(`marker_size=${number(markerSize)},`);
    const ions = Object.keys(view.ionEdits)
      .map(Number)
      .sort((a, b) => a - b);
    if (ions.length > 0) {
      lines.push("ion_colors={");
      for (const ion of ions) {
        const edit = view.ionEdits[ion];
        const parts = ["border", "fill"]
          .filter((part) => edit[part] !== undefined)
          .map((part) => `${part}=${JSON.stringify(edit[part])}`);
        lines.push(`    ${ion}: IonColor(${parts.join(", ")}),`);
      }
      lines.push("},");
    }
    const zones = Object.keys(view.zoneEdits).sort();
    if (zones.length > 0) {
      lines.push("processing_zone_colors={");
      for (const zone of zones)
        lines.push(
          `    ${JSON.stringify(zone)}: ${JSON.stringify(view.zoneEdits[zone])},`,
        );
      lines.push("},");
    }
    return lines.join("\n");
  }

  return {
    beginDrag,
    currentLayer,
    details,
    editHint: "Drag junctions. Click an ion or processing zone to color it.",
    dragTo,
    drawAtTime,
    endDrag,
    geometry,
    headerLines,
    ionColors,
    paint,
    paintParts,
    panelLayout,
    pick,
    planningSnapshot,
    prepare,
    resetLayout,
    settingsText,
    runningGates,
    stepTimes,
    timeBounds,
    trackPosition,
  };
})();

if (typeof module === "object" && module.exports) module.exports = GridDrawing;
