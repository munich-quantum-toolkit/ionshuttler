// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// Draw Linear results for the schedule player.
//
// Python replays the schedule and sends each ion's movements as target sites.
// This script turns sites into pixels: a chain lies on a straight line, a ring
// on an ellipse that fills the panel, or the sites lie at the coordinates that
// the user supplied. Swapping ions pass each other on both sides of the chain.
// It never decides whether an action is legal. The Matplotlib renderer in
// `_matplotlib.py` uses the same layout rules as `_scene.py`.

"use strict";

const LinearDrawing = (() => {
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
  // Same values as `SWAP_LIFT`, `SITE_TICK`, and `_CURVE_STEPS` in `_scene.py`.
  const SWAP_LIFT = 1.15;
  const SITE_TICK = 0.6;
  const CURVE_STEPS = 8;
  const UPCOMING_GATES = 6;
  const STATUS_TEXT = { SUCCESS: "completed", FAILED: "failed" };

  function zoneColor(view, panel, zone) {
    return view.zoneEdits[zone] ?? panel.zoneColors[zone];
  }

  // ---------------------------------------------------------------------------
  // Drawing data and time queries
  // ---------------------------------------------------------------------------

  function preparePanel(raw) {
    const ions = raw.ions.map(([ion, initial, movements]) => ({
      ion,
      initial,
      movements: movements.map(([start, end, target, side]) => ({
        start,
        end,
        target,
        side,
      })),
      starts: movements.map((item) => item[0]),
    }));
    const layers = raw.layers.map(([start, end, description, gates]) => ({
      start,
      end,
      description,
      gates: gates.map(
        ([id, name, gateIons, zone, planned, gateStart, gateEnd, global]) => ({
          id,
          name,
          ions: gateIons,
          zone,
          planned,
          start: gateStart,
          end: gateEnd,
          global,
        }),
      ),
    }));
    const gates = layers.flatMap((layer) => layer.gates);
    const siteZones = Array.from({ length: raw.sites }, () => null);
    for (const [zone, sites] of raw.processing_zones) {
      for (const site of sites) siteZones[site] = zone;
    }
    return {
      title: raw.title,
      start: raw.start,
      end: raw.end,
      sites: raw.sites,
      ring: raw.ring,
      coordinates: raw.coordinates,
      field: raw.field,
      zones: raw.processing_zones.map(([id, sites]) => ({ id, sites })),
      siteZones,
      ions,
      layers,
      layerStarts: layers.map((layer) => layer.start),
      gates,
      gateStarts: gates.map((gate) => gate.start),
      longestGate: gates.reduce(
        (longest, gate) => Math.max(longest, gate.end - gate.start),
        0,
      ),
      summary: raw.summary,
      layout: null,
      ionPoints: [],
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

  function modulo(value, divisor) {
    return ((value % divisor) + divisor) % divisor;
  }

  // Same rules as `track_position` in `_scene.py`: `[site, lift]`. A ring
  // passes its number of sites as `ringSize`.
  function trackPosition(track, time, ringSize = null) {
    const index = bisectRight(track.starts, time) - 1;
    if (index < 0) return [track.initial, 0];
    const movement = track.movements[index];
    if (time >= movement.end) return [movement.target, 0];
    const source =
      index === 0 ? track.initial : track.movements[index - 1].target;
    const progress = (time - movement.start) / (movement.end - movement.start);
    const lift = movement.side * Math.sin(Math.PI * progress);
    let step = movement.target - source;
    if (ringSize === null) return [source + step * progress, lift];
    if (Math.abs(step) === ringSize - 1) step = step > 0 ? -1 : 1;
    return [modulo(source + step * progress, ringSize), lift];
  }

  function panelPosition(panel, track, time) {
    return trackPosition(track, time, panel.ring ? panel.sites : null);
  }

  function runningGates(panel, time) {
    let index = bisectRight(panel.gateStarts, time);
    const earliestStart = time - panel.longestGate;
    const running = [];
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

  // ---------------------------------------------------------------------------
  // Layout; same rules as `_scene.py`
  // ---------------------------------------------------------------------------

  function panelLayout(panel, left, top, width, height, markerSize = 1) {
    const fontSize = Math.max(
      11,
      Math.min(17, Math.round(Math.min(width, height) * 0.026)),
    );
    const headerHeight = Math.round(fontSize * 3.6);
    const padding = Math.round(Math.min(width, height) * 0.03);
    const availableWidth = Math.max(width - 2 * padding, 1);
    const availableHeight = Math.max(height - headerHeight - 2 * padding, 1);
    const areaLeft = left + padding;
    const areaTop = top + headerHeight + padding;
    const count = panel.sites;
    let ellipse = null;
    let sites;
    let spacing;
    if (panel.coordinates === null && !panel.ring) {
      spacing = availableWidth / count;
      const chainY = areaTop + 0.45 * availableHeight;
      sites = Array.from({ length: count }, (_, site) => [
        areaLeft + (site + 0.5) * spacing,
        chainY,
      ]);
    } else {
      const largestRadius = Math.min(
        0.09 * availableHeight,
        0.05 * Math.min(width, height),
      );
      // Room for swapping ions and gate names, which use the gate-name font size.
      const margin =
        (SWAP_LIFT + 1) * largestRadius + Math.max(9, 0.72 * fontSize) + 4;
      const innerWidth = Math.max(availableWidth - 2 * margin, 1);
      const innerHeight = Math.max(availableHeight - 2 * margin, 1);
      const centerX = areaLeft + availableWidth / 2;
      const centerY = areaTop + availableHeight / 2;
      if (panel.coordinates === null) {
        ellipse = [centerX, centerY, innerWidth / 2, innerHeight / 2];
        sites = Array.from({ length: count }, (_, site) => {
          const angle = ringAngle(site, count);
          return [
            centerX + (innerWidth / 2) * Math.cos(angle),
            centerY + (innerHeight / 2) * Math.sin(angle),
          ];
        });
      } else {
        sites = fit(
          panel.coordinates,
          centerX,
          centerY,
          innerWidth,
          innerHeight,
          panel.ring,
        );
      }
      const neighbors = sites
        .slice(0, panel.ring ? count : count - 1)
        .map((site, index) => {
          const next = sites[(index + 1) % count];
          return Math.hypot(next[0] - site[0], next[1] - site[1]);
        });
      spacing = neighbors.length > 0 ? Math.min(...neighbors) : innerWidth;
    }
    const ionRadius = Math.max(
      Math.min(
        0.3 * spacing,
        0.09 * availableHeight,
        0.05 * Math.min(width, height),
      ) * markerSize,
      2,
    );
    const layout = {
      left,
      top,
      width,
      height,
      sites,
      ring: panel.ring,
      ellipse,
      spacing,
      // The field profile fills the panel behind the chain.
      fieldTop: areaTop + 0.04 * availableHeight,
      fieldBottom: areaTop + 0.86 * availableHeight,
      ionRadius,
      fontSize,
      headerHeight,
      padding,
      pieceNormals: pieceNormals(sites, panel.ring),
    };
    layout.onChain = (site) => onChain(layout, site);
    layout.normal = (site) => normalAt(layout, site);
    layout.beside = (site, distance) => {
      const [x, y] = onChain(layout, site);
      if (distance === 0) return [x, y];
      const [normalX, normalY] = normalAt(layout, site);
      return [x + normalX * distance, y + normalY * distance];
    };
    layout.point = (site, lift = 0) =>
      layout.beside(site, lift * SWAP_LIFT * ionRadius);
    return layout;
  }

  // Same rules as `PanelLayout` in `_scene.py`.
  function ringAngle(site, count) {
    return -Math.PI / 2 + (2 * Math.PI * site) / count;
  }

  function unit(x, y) {
    const length = Math.hypot(x, y);
    return [x / length, y / length];
  }

  // The box also holds the ends of a chain, half a piece beyond the first
  // and the last site.
  function fit(coordinates, centerX, centerY, width, height, ring) {
    const shown = [...coordinates];
    const count = coordinates.length;
    if (!ring && count > 1) {
      for (const [end, inner] of [
        [coordinates[0], coordinates[1]],
        [coordinates[count - 1], coordinates[count - 2]],
      ])
        shown.push([
          1.5 * end[0] - 0.5 * inner[0],
          1.5 * end[1] - 0.5 * inner[1],
        ]);
    }
    const xs = shown.map(([x]) => x);
    const ys = shown.map(([, y]) => y);
    const [minX, maxX] = [Math.min(...xs), Math.max(...xs)];
    const [minY, maxY] = [Math.min(...ys), Math.max(...ys)];
    const scale = Math.min(
      width / Math.max(maxX - minX, 1e-9),
      height / Math.max(maxY - minY, 1e-9),
    );
    const middleX = (minX + maxX) / 2;
    const middleY = (minY + maxY) / 2;
    return coordinates.map(([x, y]) => [
      centerX + (x - middleX) * scale,
      centerY - (y - middleY) * scale,
    ]);
  }

  function piece(layout, site) {
    const count = layout.sites.length;
    let index;
    if (layout.ring) {
      site = modulo(site, count);
      index = Math.min(Math.floor(site), count - 1);
    } else {
      index = Math.min(Math.max(Math.floor(site), 0), count - 2);
      if (site === count - 1) return [count - 1, 0];
    }
    return [index, site - index];
  }

  // The shoelace area is positive for a ring that runs clockwise on the
  // screen; a ring turns its normals outward.
  function pieceNormals(sites, ring) {
    const count = sites.length;
    const pieces = sites
      .slice(0, ring ? count : count - 1)
      .map((site, index) => [site, sites[(index + 1) % count]]);
    const area = pieces.reduce(
      (sum, [[x0, y0], [x1, y1]]) => sum + x0 * y1 - x1 * y0,
      0,
    );
    const orientation = ring && area < 0 ? -1 : 1;
    const normals = pieces.map(([[x0, y0], [x1, y1]]) => {
      const length = Math.hypot(x1 - x0, y1 - y0);
      return length === 0
        ? [0, -1]
        : [
            (orientation * (y1 - y0)) / length,
            (orientation * (x0 - x1)) / length,
          ];
    });
    if (!ring && normals.length > 0) normals.push(normals[normals.length - 1]);
    return normals;
  }

  function onChain(layout, site) {
    const count = layout.sites.length;
    if (layout.ellipse !== null) {
      const [centerX, centerY, radiusX, radiusY] = layout.ellipse;
      const angle = ringAngle(site, count);
      return [
        centerX + radiusX * Math.cos(angle),
        centerY + radiusY * Math.sin(angle),
      ];
    }
    if (count === 1) {
      const [x, y] = layout.sites[0];
      return [x + site * layout.spacing, y];
    }
    const [index, fraction] = piece(layout, site);
    const [x0, y0] = layout.sites[index];
    const [x1, y1] = layout.sites[(index + 1) % count];
    return [x0 + (x1 - x0) * fraction, y0 + (y1 - y0) * fraction];
  }

  function normalAt(layout, site) {
    const count = layout.sites.length;
    if (layout.ellipse !== null) {
      const [, , radiusX, radiusY] = layout.ellipse;
      const angle = ringAngle(site, count);
      return unit(Math.cos(angle) / radiusX, Math.sin(angle) / radiusY);
    }
    if (count === 1) return [0, -1];
    const [index, fraction] = piece(layout, site);
    const normals = layout.pieceNormals;
    if (fraction !== 0) return normals[index];
    if (layout.ring || index > 0) {
      const [beforeX, beforeY] = normals[modulo(index - 1, normals.length)];
      const [afterX, afterY] = normals[index];
      if (Math.hypot(beforeX + afterX, beforeY + afterY) > 1e-9)
        return unit(beforeX + afterX, beforeY + afterY);
    }
    return normals[index];
  }

  // Same rule as `rail_path` in `_scene.py`.
  function railPath(layout, first, last) {
    const steps = layout.ellipse === null ? 1 : CURVE_STEPS;
    const marks = [first];
    for (let site = Math.floor(first) + 1; site < Math.ceil(last); site += 1)
      marks.push(site);
    marks.push(last);
    const points = [layout.onChain(first)];
    for (let index = 0; index < marks.length - 1; index += 1) {
      const [start, end] = [marks[index], marks[index + 1]];
      for (let step = 1; step <= steps; step += 1)
        points.push(layout.onChain(start + ((end - start) * step) / steps));
    }
    return points;
  }

  // Same rule as `site_span` in `_scene.py`: `[first, last]`.
  function siteSpan(sites, panel) {
    if (!panel.ring) return [Math.min(...sites), Math.max(...sites)];
    const ordered = sites
      .map((site) => modulo(site, panel.sites))
      .sort((a, b) => a - b);
    const gaps = ordered.slice(1).map((site, index) => site - ordered[index]);
    gaps.push(ordered[0] + panel.sites - ordered[ordered.length - 1]);
    let widest = 0;
    gaps.forEach((gap, index) => {
      if (gap > gaps[widest]) widest = index;
    });
    if (widest === gaps.length - 1)
      return [ordered[0], ordered[ordered.length - 1]];
    return [ordered[widest + 1], ordered[widest] + panel.sites];
  }

  function fieldPoints(panel, layout) {
    if (panel.field === null) return [];
    const low = Math.min(...panel.field);
    const span = Math.max(...panel.field) - low;
    return panel.field.map((value, site) => [
      layout.point(site)[0],
      layout.fieldBottom -
        ((value - low) / span) * (layout.fieldBottom - layout.fieldTop),
    ]);
  }

  // Same rule as `rail_runs` in `_scene.py`: `[first site, last site, zone]`
  // for each stretch of neighboring sites in the same processing zone, or in
  // none. On a ring, a stretch across site 0 starts at a negative site.
  function railRuns(panel) {
    const runs = [];
    panel.siteZones.forEach((zone, site) => {
      const last = runs[runs.length - 1];
      if (last !== undefined && last[2] === zone) last[1] = site;
      else runs.push([site, site, zone]);
    });
    if (
      panel.ring &&
      runs.length > 1 &&
      runs[0][2] === runs[runs.length - 1][2]
    ) {
      const [first] = runs.pop();
      runs[0][0] = first - panel.sites;
    }
    return runs;
  }

  // Same rule as `site_tick_width` in `_scene.py`.
  function siteTickWidth(radius) {
    return Math.max(1, 0.12 * radius);
  }

  // Same rule as `rail_width` in `_scene.py`.
  function railWidth(radius, inZone) {
    return inZone ? Math.max(2, 0.7 * radius) : Math.max(1.5, 0.45 * radius);
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

  // Comparison panels stack from top to bottom, so every chain keeps the
  // full width.
  function drawAtTime(context, view, time, settings, width, height) {
    const colors = view.colors[settings.theme];
    context.save();
    paintBackground(context, 0, 0, width, height, colors);
    const panelHeight = height / view.panels.length;
    view.panels.forEach((panel, index) => {
      const layout = panelLayout(
        panel,
        0,
        index * panelHeight,
        width,
        panelHeight,
        settings.marker_size ?? 1,
      );
      if (index > 0) {
        context.fillStyle = colors.chip;
        context.fillRect(
          layout.padding,
          Math.round(index * panelHeight),
          width - 2 * layout.padding,
          1,
        );
      }
      panel.layout = layout;
      drawPanel(context, view, panel, layout, time, settings, colors);
    });
    context.restore();
  }

  function drawPanel(context, view, panel, layout, time, settings, colors) {
    const shownTime = Math.min(Math.max(time, panel.start), panel.end);
    const gates = runningGates(panel, shownTime);
    const radius = layout.ionRadius;
    const gateColor = (gate) =>
      gate.zone === null ? colors.gate : zoneColor(view, panel, gate.zone);
    const activeZones = new Set(
      gates.map((gate) => gate.zone).filter((zone) => zone !== null),
    );
    const positions = panel.ions.map((track) =>
      panelPosition(panel, track, shownTime),
    );
    const points = positions.map(([site, lift]) => layout.point(site, lift));
    const pointOf = new Map(
      panel.ions.map((track, index) => [track.ion, points[index]]),
    );
    const siteOf = new Map(
      panel.ions.map((track, index) => [track.ion, positions[index][0]]),
    );
    panel.ionPoints = points;

    if (settings.show_field_profile && panel.field !== null)
      drawField(context, panel, layout, colors);

    // The chain: wider and colored in processing zones.
    const runs = railRuns(panel);
    context.lineJoin = "round";
    for (const inZone of [false, true]) {
      for (const [first, last, zone] of runs) {
        if ((zone !== null) !== inZone) continue;
        const path = railPath(layout, first - 0.5, last + 0.5);
        const width = railWidth(radius, inZone);
        const color = inZone ? zoneColor(view, panel, zone) : colors.segment;
        // Processing zones glow faintly in their own color, and fully while
        // they run a gate.
        if (inZone) {
          const strength = activeZones.has(zone) ? 1 : AMBIENT_GLOW;
          paintLineGlow(context, path, width, color, strength);
        }
        context.lineCap = "round";
        context.strokeStyle = color;
        context.lineWidth = width;
        polyline(context, path);
      }
    }
    // Each site is a tick across the chain, like the marks of a ruler.
    context.strokeStyle = colors.junction;
    context.lineWidth = siteTickWidth(radius);
    context.lineCap = "butt";
    for (let site = 0; site < panel.sites; site += 1) {
      polyline(context, [
        layout.beside(site, SITE_TICK * radius),
        layout.beside(site, -SITE_TICK * radius),
      ]);
    }
    context.lineCap = "round";
    drawSiteLabels(context, view, panel, layout, settings, colors);

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
    // Every ion outside a running gate glows faintly in its ring color.
    const gateIons = new Set(
      gates
        .filter((gate) => gate.end > gate.start)
        .flatMap((gate) => gate.ions),
    );
    panel.ions.forEach((track, index) => {
      if (gateIons.has(track.ion)) return;
      const [x, y] = points[index];
      const halo = styles[index].border;
      paintGlow(context, x, y, 0.8 * radius, 1.6 * radius, halo, AMBIENT_GLOW);
    });

    // Gates under the ions: a band behind global gates, a bar between the
    // ions of two-ion gates, and a glow around every gate ion.
    for (const gate of gates) {
      if (gate.end === gate.start) continue;
      const gatePoints = gate.ions
        .map((ion) => pointOf.get(ion))
        .filter((point) => point !== undefined);
      if (gatePoints.length === 0) continue;
      context.fillStyle = gateColor(gate);
      context.strokeStyle = gateColor(gate);
      // Bands and bars follow the chain between the gate ions.
      const [first, last] = siteSpan(
        gate.ions
          .filter((ion) => siteOf.has(ion))
          .map((ion) => siteOf.get(ion)),
        panel,
      );
      if (gate.global) {
        context.globalAlpha = 0.14;
        context.lineWidth = 2.8 * radius;
        polyline(context, railPath(layout, first, last));
      } else if (gatePoints.length === 2) {
        context.globalAlpha = 0.35;
        context.lineWidth = 0.9 * radius;
        polyline(context, railPath(layout, first, last));
      }
      context.globalAlpha = 1;
      for (const [x, y] of gatePoints)
        paintGlow(context, x, y, 0.8 * radius, 1.9 * radius, gateColor(gate));
    }

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
    drawGateNames(context, panel, gates, siteOf, layout, gateColor);
    if (settings.ion_colors === "custom")
      drawLegend(context, view, layout, colors);
    drawHeader(context, panel, layout, time, colors);
  }

  // The field profile behind the chain: a smooth curve through the field of
  // every site, filled down to the bottom of the backdrop.
  function drawField(context, panel, layout, colors) {
    const points = fieldPoints(panel, layout);
    if (points.length < 2) return;
    const trace = () => {
      context.beginPath();
      context.moveTo(...points[0]);
      for (let index = 0; index < points.length - 1; index += 1) {
        const [x, y] = points[index];
        const [nextX, nextY] = points[index + 1];
        context.quadraticCurveTo(x, y, (x + nextX) / 2, (y + nextY) / 2);
      }
      context.lineTo(...points[points.length - 1]);
    };
    trace();
    context.lineTo(points[points.length - 1][0], layout.fieldBottom);
    context.lineTo(points[0][0], layout.fieldBottom);
    context.closePath();
    // A faint backdrop, so the chain in front of it stays clear.
    context.fillStyle = colors.muted_text;
    context.globalAlpha = 0.07;
    context.fill();
    context.globalAlpha = 0.5;
    trace();
    context.strokeStyle = colors.muted_text;
    context.lineWidth = 1.5;
    context.lineJoin = "round";
    context.stroke();
    context.globalAlpha = 1;
    context.fillStyle = colors.muted_text;
    context.font = `400 ${Math.max(9, 0.72 * layout.fontSize)}px ${FONT}`;
    context.textAlign = "left";
    context.textBaseline = "top";
    context.fillText("field", layout.left + layout.padding, layout.fieldTop);
  }

  // Site numbers on the side opposite the normal, then processing-zone names
  // beyond them.
  function drawSiteLabels(context, view, panel, layout, settings, colors) {
    const radius = layout.ionRadius;
    const size = siteLabelSize(layout);
    const numbersDistance = (SWAP_LIFT + 1) * radius + 0.4 * size + 2;
    context.textAlign = "center";
    context.textBaseline = "middle";
    if (settings.show_site_numbers) {
      context.fillStyle = colors.muted_text;
      context.font = `400 ${size}px ${FONT}`;
      const step = siteLabelStep(panel, layout, size);
      for (let site = 0; site < panel.sites; site += step) {
        context.fillText(
          String(site),
          ...layout.beside(site, -(numbersDistance + 0.5 * size)),
        );
      }
    }
    if (settings.show_processing_zone_labels) {
      const zoneSize = zoneLabelSize(layout);
      context.font = `600 ${zoneSize}px ${FONT}`;
      const zonesDistance =
        numbersDistance +
        (settings.show_site_numbers ? 1.5 * size : 0) +
        0.5 * zoneSize;
      for (const zone of panel.zones) {
        if (zone.sites.length === 0) continue;
        const [first, last] = siteSpan(zone.sites, panel);
        context.fillStyle = zoneColor(view, panel, zone.id);
        context.fillText(
          zone.id,
          ...layout.beside((first + last) / 2, -zonesDistance),
        );
      }
    }
  }

  // Same rule as `zone_label_size` in `_matplotlib.py`.
  function zoneLabelSize(layout) {
    return Math.max(11, 0.95 * layout.fontSize);
  }

  // Same rules as `site_label_size` and `site_label_step` in `_matplotlib.py`.
  // Long or crowded chains number every second, third, or later site.
  function siteLabelSize(layout) {
    return Math.max(9, 0.72 * layout.fontSize);
  }

  function siteLabelStep(panel, layout, size) {
    const widest = String(panel.sites - 1).length * 0.6 * size + 6;
    return Math.max(1, Math.ceil(widest / layout.spacing));
  }

  // Gate names beside the running gates, on the side of the normal.
  function drawGateNames(context, panel, gates, siteOf, layout, gateColor) {
    const size = Math.max(9, 0.72 * layout.fontSize);
    const distance = (SWAP_LIFT + 1) * layout.ionRadius + 0.5 * size + 2;
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.font = `600 ${size}px ${FONT}`;
    for (const gate of gates) {
      const sites = gate.ions
        .filter((ion) => siteOf.has(ion))
        .map((ion) => siteOf.get(ion));
      if (sites.length === 0) continue;
      const [first, last] = siteSpan(sites, panel);
      context.fillStyle = gateColor(gate);
      context.fillText(
        gate.name,
        ...layout.beside((first + last) / 2, distance),
      );
    }
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

  function drawHeader(context, panel, layout, time, colors) {
    const [title, timeText, description] = headerLines(panel, time);
    const fontSize = layout.fontSize;
    const left = layout.left + layout.padding;
    const right = layout.left + layout.width - layout.padding;
    const top = layout.top + layout.padding * 0.6;
    context.textBaseline = "top";
    context.font = `400 ${0.9 * fontSize}px ${FONT}`;
    const timeWidth = context.measureText(timeText).width + fontSize;
    context.fillStyle = colors.muted_text;
    context.textAlign = "right";
    context.fillText(timeText, right, top);
    context.textAlign = "left";
    if (title) {
      context.fillStyle = colors.text;
      context.font = `600 ${fontSize}px ${FONT}`;
      context.fillText(
        truncate(title, right - left - timeWidth - fontSize, fontSize),
        left,
        top,
      );
    }
    // Without a title, the action line moves up beside the time.
    const descriptionTop = title
      ? top + 1.55 * fontSize
      : top + 0.12 * fontSize;
    const descriptionWidth = title
      ? right - left
      : right - left - timeWidth - fontSize;
    context.font = `400 ${0.85 * fontSize}px ${FONT}`;
    context.fillStyle = colors.muted_text;
    context.fillText(
      truncate(description, descriptionWidth, 0.85 * fontSize),
      left,
      descriptionTop,
    );
  }

  function polyline(context, points) {
    context.beginPath();
    context.moveTo(...points[0]);
    for (const point of points.slice(1)) context.lineTo(...point);
    context.stroke();
  }

  function circle(context, x, y, radius) {
    context.beginPath();
    context.arc(x, y, radius, 0, 2 * Math.PI);
  }

  // ---------------------------------------------------------------------------
  // Schedule details in the side bar
  // ---------------------------------------------------------------------------

  // The side bar lists the running and upcoming gates of every panel, the
  // moving ions, each ion's site, and the compilation summary. The schedule
  // is final, so upcoming gates are the gates that the compiler fixed.
  function details(view, time, settings) {
    return view.panels.map((panel) => {
      const shownTime = Math.min(Math.max(time, panel.start), panel.end);
      return {
        title: view.panels.length > 1 ? panel.title : "",
        rows: scheduleRows(view, panel, shownTime, settings),
      };
    });
  }

  // The rows of the side bar for one panel, as `[label, lines]`. Ions are
  // pills framed in their ring color, as on the drawing. Processing zones are
  // pills filled with their color.
  function scheduleRows(view, panel, time, settings) {
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
    const zone = (id) => {
      const fill = zoneColor(view, panel, id) ?? colors.gate;
      return { text: id, fill, border: fill, ink: textColor(fill) };
    };
    const gateLine = (gate) => [
      gate.id === null ? gate.name : `g${gate.id} ${gate.name}`,
      ...(gate.global ? [`${gate.ions.length} ions`] : gate.ions.map(ion)),
      ...(gate.zone === null ? [] : ["→", zone(gate.zone)]),
      ...(gate.planned !== null && gate.planned !== gate.zone
        ? [`planned ${gate.planned}`]
        : []),
    ];
    const running = runningGates(panel, time);
    const next = panel.gates.slice(
      bisectRight(panel.gateStarts, time),
      bisectRight(panel.gateStarts, time) + UPCOMING_GATES,
    );
    const gateOfIon = new Map(
      running.flatMap((gate) => gate.ions.map((value) => [value, gate])),
    );
    const moving = [];
    const sites = panel.ions.map((track) => {
      const [site] = panelPosition(panel, track, time);
      const index = bisectRight(track.starts, time) - 1;
      const movement = index < 0 ? null : track.movements[index];
      const moves = movement !== null && time < movement.end;
      if (moves) {
        const source =
          index === 0 ? track.initial : track.movements[index - 1].target;
        moving.push([
          ion(track.ion),
          `${source} → ${movement.target}`,
          movement.side === 0 ? "shuttle" : "swap",
        ]);
      }
      const nearest = modulo(Math.round(site), panel.sites);
      const siteZone = panel.siteZones[nearest];
      const gate = gateOfIon.get(track.ion);
      return [
        ion(track.ion),
        moves ? `site ${site.toFixed(1)}` : `site ${nearest}`,
        ...(siteZone === null || moves ? [] : [zone(siteZone)]),
        moves ? "moving" : gate === undefined ? "idle" : gate.name,
      ];
    });
    const done = panel.gates.filter(
      (gate) => gate.end < time || (gate.end === time && gate.end > gate.start),
    ).length;
    const summary = panel.summary;
    const run = [
      STATUS_TEXT[summary.status] ?? summary.status.toLowerCase(),
      `${done} of ${summary.gates} gates done`,
      `${summary.swaps} swaps · ${summary.shuttles} shuttles`,
      // The diagnostics of each compiler method name their own statistics.
      ...summary.statistics.map(([label, value]) => `${label}: ${value}`),
    ];
    if (summary.compile_seconds !== null)
      run.push(`compiled in ${summary.compile_seconds.toFixed(2)} s`);
    const rows = [
      [
        "Running",
        running.map((gate) => [...gateLine(gate), `until t ${gate.end}`]),
      ],
      ["Next", next.map((gate) => [`t ${gate.start}`, ...gateLine(gate)])],
      ["Moving", moving],
      ["Ions", sites],
      ["Run", run],
    ];
    const planned = panel.gates.filter((gate) => gate.planned !== null);
    if (planned.length > 0) {
      const kept = planned.filter((gate) => gate.planned === gate.zone).length;
      rows.push([
        "Planned zones",
        [`${kept} of ${planned.length} gates ran in their planned zone`],
      ]);
    }
    return rows;
  }

  // ---------------------------------------------------------------------------
  // Color editing
  // ---------------------------------------------------------------------------

  const editHint = "Click an ion or processing zone to color it.";

  // Return what lies under a canvas point in the last drawn frame: an ion,
  // then a processing zone on the chain.
  function pick(view, x, y) {
    for (const panel of view.panels) {
      const layout = panel.layout;
      if (layout === null) continue;
      if (y < layout.top || y > layout.top + layout.height) continue;
      let best = null;
      panel.ionPoints.forEach((point, index) => {
        const gap = Math.hypot(point[0] - x, point[1] - y);
        if (best === null || gap < best.gap) best = { index, gap };
      });
      if (best !== null && best.gap <= Math.max(8, layout.ionRadius)) {
        return { kind: "ion", panel, ion: panel.ions[best.index].ion };
      }
      // The nearest of many points along the chain; each site owns the
      // chain halfway to its neighbors.
      let nearest = null;
      const last = panel.ring ? panel.sites : panel.sites - 0.5;
      for (let site = -0.5; site <= last; site += 0.05) {
        const [chainX, chainY] = layout.onChain(site);
        const gap = Math.hypot(chainX - x, chainY - y);
        if (nearest === null || gap < nearest.gap) nearest = { site, gap };
      }
      const rounded = Math.round(nearest.site);
      const site = panel.ring
        ? modulo(rounded, panel.sites)
        : Math.min(Math.max(rounded, 0), panel.sites - 1);
      const zone = panel.siteZones[site];
      if (
        zone !== null &&
        nearest.gap <= Math.max(10, 1.2 * layout.ionRadius)
      ) {
        return { kind: "zone", panel, zone };
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

  // Undo every color change made in the view.
  function resetLayout(view) {
    view.ionEdits = {};
    view.zoneEdits = {};
  }

  // Python keyword arguments that reproduce the marker size and the colors
  // changed in the view.
  function settingsText(view, settings = {}) {
    const lines = [];
    const markerSize = settings.marker_size ?? 1;
    if (markerSize !== 1)
      lines.push(`marker_size=${Math.round(markerSize * 1000) / 1000},`);
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
    return lines.length > 0
      ? lines.join("\n")
      : "# The view uses the original settings.";
  }

  return {
    currentLayer,
    details,
    drawAtTime,
    editHint,
    fieldPoints,
    headerLines,
    paint,
    paintParts,
    panelLayout,
    pick,
    prepare,
    railPath,
    resetLayout,
    runningGates,
    settingsText,
    stepTimes,
    timeBounds,
    trackPosition,
  };
})();

if (typeof module === "object" && module.exports)
  module.exports = LinearDrawing;
