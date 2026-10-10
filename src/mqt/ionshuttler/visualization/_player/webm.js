// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// Write encoded VP8 or VP9 frames into a WebM (Matroska) file. The writer
// keeps the frames in memory, so it knows every element size when it writes
// the file. It adds a cue for each cluster that starts with a key frame, so
// players can seek.

"use strict";

const WebMWriter = (() => {
  class Writer {
    constructor(codecId, width, height, framesPerSecond) {
      this.codecId = codecId;
      this.width = width;
      this.height = height;
      this.frameDurationMs = 1000 / framesPerSecond;
      this.clusters = [];
      this.endMs = 0;
    }

    addChunk(chunk) {
      const data = new Uint8Array(chunk.byteLength);
      chunk.copyTo(data);
      this.addFrame(data, chunk.timestamp / 1000, chunk.type === "key");
    }

    addFrame(data, timeMs, isKey) {
      const time = Math.round(timeMs);
      const cluster = this.clusters.at(-1);
      if (cluster === undefined || isKey || time - cluster.time > 30000) {
        this.clusters.push({
          time,
          isKey,
          blocks: [{ relativeTime: 0, isKey, data }],
        });
      } else {
        cluster.blocks.push({ relativeTime: time - cluster.time, isKey, data });
      }
      this.endMs = Math.max(this.endMs, timeMs + this.frameDurationMs);
    }

    finish() {
      const header = element(0x1a45dfa3, [
        element(0x4286, uint(1)),
        element(0x42f7, uint(1)),
        element(0x42f2, uint(4)),
        element(0x42f3, uint(8)),
        element(0x4282, text("webm")),
        element(0x4287, uint(4)),
        element(0x4285, uint(2)),
      ]);
      const info = element(0x1549a966, [
        element(0x2ad7b1, uint(1000000)),
        element(0x4d80, text("mqt-ionshuttler")),
        element(0x5741, text("mqt-ionshuttler")),
        element(0x4489, float64(this.endMs)),
      ]);
      const tracks = element(0x1654ae6b, [
        element(0xae, [
          element(0xd7, uint(1)),
          element(0x73c5, uint(1)),
          element(0x83, uint(1)),
          element(0x86, text(this.codecId)),
          element(0x9c, uint(0)),
          element(0x23e383, uint(Math.round(this.frameDurationMs * 1e6))),
          element(0xe0, [
            element(0xb0, uint(this.width)),
            element(0xba, uint(this.height)),
          ]),
        ]),
      ]);
      const clusters = this.clusters.map((cluster) =>
        element(0x1f43b675, [
          element(0xe7, uint(cluster.time)),
          ...cluster.blocks.map((block) =>
            element(0xa3, [
              Uint8Array.of(
                0x81,
                (block.relativeTime >> 8) & 0xff,
                block.relativeTime & 0xff,
                block.isKey ? 0x80 : 0x00,
              ),
              block.data,
            ]),
          ),
        ]),
      );
      // Cue positions use a fixed width, so the cue size does not depend on them.
      const cues = (positions) =>
        element(
          0x1c53bb6b,
          this.clusters.flatMap((cluster, index) =>
            cluster.isKey
              ? [
                  element(0xbb, [
                    element(0xb3, uint(cluster.time, 8)),
                    element(0xb7, [
                      element(0xf7, uint(1)),
                      element(0xf1, uint(positions[index], 8)),
                    ]),
                  ]),
                ]
              : [],
          ),
        );
      const positions = [];
      let position =
        info.size + tracks.size + cues(this.clusters.map(() => 0)).size;
      for (const cluster of clusters) {
        positions.push(position);
        position += cluster.size;
      }
      const segment = element(0x18538067, [
        info,
        tracks,
        cues(positions),
        ...clusters,
      ]);
      return new Blob(flatten([header, segment]), { type: "video/webm" });
    }
  }

  // An EBML element is a list of byte arrays and nested elements with its total size.
  function element(id, content) {
    const parts = content instanceof Uint8Array ? [content] : content;
    const size = parts.reduce(
      (sum, part) =>
        sum + (part instanceof Uint8Array ? part.length : part.size),
      0,
    );
    const head = [elementId(id), sizeField(size)];
    return {
      parts: [...head, ...parts],
      size: head[0].length + head[1].length + size,
    };
  }

  function flatten(parts, output = []) {
    for (const part of parts) {
      if (part instanceof Uint8Array) output.push(part);
      else flatten(part.parts, output);
    }
    return output;
  }

  function elementId(id) {
    const bytes = [];
    for (let value = id; value > 0; value = Math.floor(value / 256))
      bytes.unshift(value % 256);
    return Uint8Array.from(bytes);
  }

  function sizeField(size) {
    for (let length = 1; length <= 8; length += 1) {
      if (size < 2 ** (7 * length) - 1) {
        const bytes = uint(size, length);
        bytes[0] |= 1 << (8 - length);
        return bytes;
      }
    }
    throw new RangeError("WebM element is too large");
  }

  function uint(value, length) {
    let byteCount = length;
    if (byteCount === undefined) {
      byteCount = 1;
      while (value >= 2 ** (8 * byteCount)) byteCount += 1;
    }
    const bytes = new Uint8Array(byteCount);
    let rest = value;
    for (let index = byteCount - 1; index >= 0; index -= 1) {
      bytes[index] = rest % 256;
      rest = Math.floor(rest / 256);
    }
    return bytes;
  }

  function text(value) {
    return new TextEncoder().encode(value);
  }

  function float64(value) {
    const bytes = new Uint8Array(8);
    new DataView(bytes.buffer).setFloat64(0, value);
    return bytes;
  }

  return Writer;
})();

if (typeof module === "object" && module.exports) module.exports = WebMWriter;
