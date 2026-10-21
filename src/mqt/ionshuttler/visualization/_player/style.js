// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// Ion colors and text sizes that every drawing shares; same rules as
// `style.py`. A drawing keeps the colors chosen in Python in `view.ionChanges`
// and the colors changed in the view in `view.ionEdits`.

"use strict";

const PlayerStyle = (() => {
  const FONT =
    "Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif";
  const ION_PALETTE = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#db2777",
    "#65a30d",
    "#ca8a04",
    "#4f46e5",
    "#0d9488",
    "#78716c",
  ];
  const ZONE_PALETTE = [
    "#f59e0b",
    "#14b8a6",
    "#ec4899",
    "#8b5cf6",
    "#84cc16",
    "#06b6d4",
    "#f97316",
    "#ef4444",
  ];
  const NEUTRAL_BORDER = "#64748b";
  const WHITE = "#ffffff";
  const DARK_TEXT = "#0f172a";

  function hsl(hue, saturation, lightness) {
    const chroma = (1 - Math.abs(2 * lightness - 1)) * saturation;
    const second = chroma * (1 - Math.abs(((hue / 60) % 2) - 1));
    const sector = Math.floor(hue / 60);
    const [red, green, blue] = [
      [chroma, second, 0],
      [second, chroma, 0],
      [0, chroma, second],
      [0, second, chroma],
      [second, 0, chroma],
    ][sector] ?? [chroma, 0, second];
    const shift = lightness - chroma / 2;
    return `#${[red, green, blue]
      .map((value) =>
        Math.floor((value + shift) * 255 + 0.5)
          .toString(16)
          .padStart(2, "0"),
      )
      .join("")}`;
  }

  function distinctColor(index, palette) {
    return index < palette.length
      ? palette[index]
      : hsl((index * 137.508) % 360, 0.65, 0.45);
  }

  function textColor(fill) {
    const linear = (channel) =>
      channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
    const [red, green, blue] = [1, 3, 5].map((index) =>
      linear(parseInt(fill.slice(index, index + 2), 16) / 255),
    );
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue > 0.35
      ? DARK_TEXT
      : WHITE;
  }

  // Ring, fill, and text color of an ion. Colors chosen in the view win. Ions
  // without a chosen fill use `plainFill`, which depends on the theme.
  function ionColors(view, ion, time, mode, single, plainFill = WHITE) {
    let border = NEUTRAL_BORDER;
    let fill = plainFill;
    if (mode === "distinct") border = distinctColor(ion, ION_PALETTE);
    else if (mode === "single") border = single;
    else {
      const timeline = view.ionChanges[ion] ?? [];
      let index = 0;
      while (index < timeline.length && timeline[index][0] <= time) index += 1;
      if (index > 0) {
        const [, chosenBorder, chosenFill] = timeline[index - 1];
        border = chosenBorder ?? border;
        fill = chosenFill ?? fill;
      }
    }
    const edit = view.ionEdits[ion];
    if (edit !== undefined) {
      border = edit.border ?? border;
      fill = edit.fill ?? fill;
    }
    return { border, fill, text: textColor(fill) };
  }

  function legendEntries(view, plainFill) {
    const entries = new Map();
    for (const ion of Object.keys(view.ionChanges)
      .map(Number)
      .sort((a, b) => a - b)) {
      for (const [, border, fill, label] of view.ionChanges[ion]) {
        if (label !== null && !entries.has(label))
          entries.set(label, [border ?? NEUTRAL_BORDER, fill ?? plainFill]);
      }
    }
    return [...entries].map(([label, [border, fill]]) => ({
      label,
      border,
      fill,
    }));
  }

  function ionLabelFontSize(ionRadius, label) {
    return Math.min(
      0.95 * ionRadius,
      (2.5 * ionRadius) / Math.max(label.length, 2),
    );
  }

  function truncate(text, width, fontSize) {
    const limit = Math.max(Math.floor(width / (0.55 * fontSize)), 1);
    return text.length <= limit ? text : `${text.slice(0, limit - 1)}…`;
  }

  // ---------------------------------------------------------------------------
  // Shapes. Flat lines and markers; only running gates glow. The Matplotlib
  // renderers draw the same halos (`figure_style.py`).
  // ---------------------------------------------------------------------------

  // Strength of the glow that every ion and processing zone has at all times.
  // Same value as `AMBIENT_GLOW` in `figure_style.py`.
  const AMBIENT_GLOW = 0.55;
  // How much darker an ion's ring gets toward its outer edge. Same value as
  // `RING_SHADE` in `style.py`.
  const RING_SHADE = 0.2;

  function translucent(color, alpha) {
    const [red, green, blue] = [1, 3, 5].map((index) =>
      parseInt(color.slice(index, index + 2), 16),
    );
    return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
  }

  function roundedRectangle(context, x, y, width, height, corner) {
    const radius = Math.min(corner, width / 2, height / 2);
    context.beginPath();
    context.moveTo(x + radius, y);
    context.arcTo(x + width, y, x + width, y + height, radius);
    context.arcTo(x + width, y + height, x, y + height, radius);
    context.arcTo(x, y + height, x, y, radius);
    context.arcTo(x, y, x + width, y, radius);
    context.closePath();
  }

  function paintBackground(context, left, top, width, height, colors) {
    context.fillStyle = colors.background;
    context.fillRect(left, top, width, height);
  }

  // A line that glows in its own color. Processing zones glow faintly at all
  // times (`strength` AMBIENT_GLOW) and fully while they run a gate.
  function paintLineGlow(context, x0, y0, x1, y1, width, color, strength = 1) {
    context.save();
    context.globalAlpha = strength;
    context.lineCap = "round";
    context.strokeStyle = translucent(color, 0.38);
    context.shadowColor = color;
    context.shadowBlur = 2.5 * width;
    context.lineWidth = 2.2 * width;
    context.beginPath();
    context.moveTo(x0, y0);
    context.lineTo(x1, y1);
    context.stroke();
    context.restore();
  }

  // A halo that fades out at `outer`. Every ion has a faint halo in its ring
  // color (`strength` AMBIENT_GLOW); the ions of a running gate get a full
  // halo in the color of the processing zone.
  function paintGlow(context, x, y, inner, outer, color, strength = 1) {
    const gradient = context.createRadialGradient(x, y, inner, x, y, outer);
    gradient.addColorStop(0, translucent(color, 0.5 * strength));
    gradient.addColorStop(1, translucent(color, 0));
    context.fillStyle = gradient;
    context.beginPath();
    context.arc(x, y, outer, 0, 2 * Math.PI);
    context.fill();
  }

  // The stroke of an ion's ring of width `ringWidth` around radius `middle`:
  // its color at the inner edge, darker toward the outer edge. Same rule as
  // `shade` in `style.py`.
  function ringStroke(context, x, y, middle, ringWidth, border) {
    const gradient = context.createRadialGradient(
      x,
      y,
      middle - ringWidth / 2,
      x,
      y,
      middle + ringWidth / 2,
    );
    gradient.addColorStop(0, border);
    gradient.addColorStop(1, shade(border, RING_SHADE));
    return gradient;
  }

  // Blend a `#rrggbb` color toward black by `amount`.
  function shade(color, amount) {
    return `#${[1, 3, 5]
      .map((index) =>
        Math.round(parseInt(color.slice(index, index + 2), 16) * (1 - amount))
          .toString(16)
          .padStart(2, "0"),
      )
      .join("")}`;
  }

  return {
    AMBIENT_GLOW,
    DARK_TEXT,
    FONT,
    ION_PALETTE,
    NEUTRAL_BORDER,
    WHITE,
    ZONE_PALETTE,
    distinctColor,
    ionColors,
    ionLabelFontSize,
    legendEntries,
    paintBackground,
    paintGlow,
    paintLineGlow,
    ringStroke,
    roundedRectangle,
    textColor,
    translucent,
    truncate,
  };
})();

if (typeof module === "object" && module.exports) module.exports = PlayerStyle;
