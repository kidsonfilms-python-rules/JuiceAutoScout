(function () {
  "use strict";

  const CSV_COLUMNS = [
    "timestamp_s",
    "robot0_x_in", "robot0_y_in", "robot0_heading_rad", "robot0_visible",
    "robot1_x_in", "robot1_y_in", "robot1_heading_rad", "robot1_visible",
    "robot2_x_in", "robot2_y_in", "robot2_heading_rad", "robot2_visible",
    "robot3_x_in", "robot3_y_in", "robot3_heading_rad", "robot3_visible",
    "robot0_shot_result", "robot0_shot_x_in", "robot0_shot_y_in", "robot0_shot_goal",
    "robot1_shot_result", "robot1_shot_x_in", "robot1_shot_y_in", "robot1_shot_goal",
    "robot2_shot_result", "robot2_shot_x_in", "robot2_shot_y_in", "robot2_shot_goal",
    "robot3_shot_result", "robot3_shot_x_in", "robot3_shot_y_in", "robot3_shot_goal",
  ];

  const MAGIC_V1 = "JLOGv001";
  const MAGIC_V2 = "JLOGv002";
  const BLOCK_MAGIC = "JBLK";
  const DEFAULT_BLOCK_ROWS = 128;
  const SUPPORTED_MAJOR_V2 = 2;

  const KIND_SCALED_INT = 1;
  const KIND_BOOL = 2;
  const KIND_STRING = 3;
  const KIND_INT = 4;
  const KIND_FLOAT64 = 5;
  const KIND_NAMES = {
    [KIND_SCALED_INT]: "scaled_int",
    [KIND_BOOL]: "bool",
    [KIND_STRING]: "string",
    [KIND_INT]: "int",
    [KIND_FLOAT64]: "float64",
  };
  const KIND_CODES = {
    scaled_int: KIND_SCALED_INT,
    bool: KIND_BOOL,
    string: KIND_STRING,
    int: KIND_INT,
    float64: KIND_FLOAT64,
  };

  const FLAG_NULLABLE = 0x01;
  const HEADER_V1_SIZE = 22;
  const HEADER_V2_PREFIX_SIZE = 20;
  const BLOCK_HEADER_SIZE = 16;
  const textEncoder = new TextEncoder();
  const textDecoder = new TextDecoder();
  const crcTable = buildCrcTable();

  const ROBOT_POSE_SCHEMA = [
    { name: "timestamp_s", kind: "scaled_int", unit: "s", scale: 1000, nullable: false },
    { name: "robot0_x_in", kind: "scaled_int", unit: "in", scale: 100, nullable: false },
    { name: "robot0_y_in", kind: "scaled_int", unit: "in", scale: 100, nullable: false },
    { name: "robot0_heading_rad", kind: "scaled_int", unit: "rad", scale: 10000, nullable: false },
    { name: "robot0_visible", kind: "bool", unit: "", scale: 1, nullable: false },
    { name: "robot1_x_in", kind: "scaled_int", unit: "in", scale: 100, nullable: false },
    { name: "robot1_y_in", kind: "scaled_int", unit: "in", scale: 100, nullable: false },
    { name: "robot1_heading_rad", kind: "scaled_int", unit: "rad", scale: 10000, nullable: false },
    { name: "robot1_visible", kind: "bool", unit: "", scale: 1, nullable: false },
    { name: "robot2_x_in", kind: "scaled_int", unit: "in", scale: 100, nullable: false },
    { name: "robot2_y_in", kind: "scaled_int", unit: "in", scale: 100, nullable: false },
    { name: "robot2_heading_rad", kind: "scaled_int", unit: "rad", scale: 10000, nullable: false },
    { name: "robot2_visible", kind: "bool", unit: "", scale: 1, nullable: false },
    { name: "robot3_x_in", kind: "scaled_int", unit: "in", scale: 100, nullable: false },
    { name: "robot3_y_in", kind: "scaled_int", unit: "in", scale: 100, nullable: false },
    { name: "robot3_heading_rad", kind: "scaled_int", unit: "rad", scale: 10000, nullable: false },
    { name: "robot3_visible", kind: "bool", unit: "", scale: 1, nullable: false },
    { name: "robot0_shot_result", kind: "string", unit: "", scale: 1, nullable: true },
    { name: "robot0_shot_x_in", kind: "scaled_int", unit: "in", scale: 100, nullable: true },
    { name: "robot0_shot_y_in", kind: "scaled_int", unit: "in", scale: 100, nullable: true },
    { name: "robot0_shot_goal", kind: "string", unit: "", scale: 1, nullable: true },
    { name: "robot1_shot_result", kind: "string", unit: "", scale: 1, nullable: true },
    { name: "robot1_shot_x_in", kind: "scaled_int", unit: "in", scale: 100, nullable: true },
    { name: "robot1_shot_y_in", kind: "scaled_int", unit: "in", scale: 100, nullable: true },
    { name: "robot1_shot_goal", kind: "string", unit: "", scale: 1, nullable: true },
    { name: "robot2_shot_result", kind: "string", unit: "", scale: 1, nullable: true },
    { name: "robot2_shot_x_in", kind: "scaled_int", unit: "in", scale: 100, nullable: true },
    { name: "robot2_shot_y_in", kind: "scaled_int", unit: "in", scale: 100, nullable: true },
    { name: "robot2_shot_goal", kind: "string", unit: "", scale: 1, nullable: true },
    { name: "robot3_shot_result", kind: "string", unit: "", scale: 1, nullable: true },
    { name: "robot3_shot_x_in", kind: "scaled_int", unit: "in", scale: 100, nullable: true },
    { name: "robot3_shot_y_in", kind: "scaled_int", unit: "in", scale: 100, nullable: true },
    { name: "robot3_shot_goal", kind: "string", unit: "", scale: 1, nullable: true },
  ];

  function buildCrcTable() {
    const table = new Uint32Array(256);
    for (let i = 0; i < 256; i += 1) {
      let crc = i;
      for (let j = 0; j < 8; j += 1) {
        crc = (crc & 1) ? (0xEDB88320 ^ (crc >>> 1)) : (crc >>> 1);
      }
      table[i] = crc >>> 0;
    }
    return table;
  }

  function crc32(bytes) {
    let crc = 0xFFFFFFFF;
    for (let i = 0; i < bytes.length; i += 1) {
      crc = crcTable[(crc ^ bytes[i]) & 0xFF] ^ (crc >>> 8);
    }
    return (crc ^ 0xFFFFFFFF) >>> 0;
  }

  function asText(value) {
    return value == null ? "" : String(value);
  }

  function parseFloatValue(value, fallback = 0) {
    const text = asText(value).trim();
    if (!text) return fallback;
    const num = Number(text);
    return Number.isFinite(num) ? num : fallback;
  }

  function parseIntValue(value, fallback = 0) {
    const text = asText(value).trim();
    if (!text) return fallback;
    const num = Number(text);
    return Number.isFinite(num) ? Math.round(num) : fallback;
  }

  function boolFromValue(value) {
    const text = asText(value).trim().toLowerCase();
    return text === "1" || text === "true" || text === "yes" || text === "y";
  }

  function quantizeScaled(value, scale) {
    return Math.round(Number(value) * scale);
  }

  function decimalsFromScale(scale) {
    if (scale <= 1) return 0;
    let decimals = 0;
    let value = Number(scale);
    while (value > 1 && value % 10 === 0) {
      value /= 10;
      decimals += 1;
    }
    return value === 1 ? decimals : 6;
  }

  function formatScaled(value, scale) {
    return (value / scale).toFixed(decimalsFromScale(scale));
  }

  function pushUInt16LE(out, value) {
    const n = Number(value) >>> 0;
    out.push(n & 0xFF, (n >>> 8) & 0xFF);
  }

  function pushUInt32LE(out, value) {
    const n = Number(value) >>> 0;
    out.push(n & 0xFF, (n >>> 8) & 0xFF, (n >>> 16) & 0xFF, (n >>> 24) & 0xFF);
  }

  function pushFloat64LE(out, value) {
    const buffer = new ArrayBuffer(8);
    const view = new DataView(buffer);
    view.setFloat64(0, value, true);
    const bytes = new Uint8Array(buffer);
    for (let i = 0; i < bytes.length; i += 1) out.push(bytes[i]);
  }

  function encodeVarUInt(value, out) {
    let n = Number(value);
    if (n < 0) throw new Error("varuint cannot encode negative values");
    while (true) {
      const byte = n & 0x7F;
      n = Math.floor(n / 128);
      if (n) out.push(byte | 0x80);
      else {
        out.push(byte);
        return;
      }
    }
  }

  function decodeVarUInt(data, cursor) {
    let shift = 0;
    let value = 0;
    while (true) {
      if (cursor.pos >= data.length) throw new Error("Unexpected end of JLOG varuint");
      const byte = data[cursor.pos];
      cursor.pos += 1;
      value += (byte & 0x7F) * Math.pow(2, shift);
      if ((byte & 0x80) === 0) return value;
      shift += 7;
      if (shift > 35) throw new Error("JLOG varuint is too large");
    }
  }

  function zigzagEncode(value) {
    const n = Number(value) | 0;
    return ((n << 1) ^ (n >> 31)) >>> 0;
  }

  function zigzagDecode(value) {
    const n = Number(value) >>> 0;
    return (n >>> 1) ^ -(n & 1);
  }

  function encodeVarInt(value, out) {
    encodeVarUInt(zigzagEncode(value), out);
  }

  function decodeVarInt(data, cursor) {
    return zigzagDecode(decodeVarUInt(data, cursor));
  }

  function encodeString(text, out) {
    const payload = textEncoder.encode(text);
    encodeVarUInt(payload.length, out);
    for (let i = 0; i < payload.length; i += 1) out.push(payload[i]);
  }

  function decodeString(data, cursor) {
    const size = decodeVarUInt(data, cursor);
    const end = cursor.pos + size;
    if (end > data.length) throw new Error("Unexpected end of JLOG string");
    const text = textDecoder.decode(data.slice(cursor.pos, end));
    cursor.pos = end;
    return text;
  }

  function normalizeSchema(schema) {
    return schema.map((column) => ({
      name: column.name,
      kind: column.kind,
      unit: column.unit || "",
      scale: Math.max(1, Number(column.scale || 1)),
      nullable: Boolean(column.nullable),
    }));
  }

  function knownUnitForName(name) {
    if (name === "timestamp_s") return "s";
    if (name.endsWith("_heading_rad")) return "rad";
    if (name.endsWith("_x_in") || name.endsWith("_y_in")) return "in";
    return "";
  }

  function knownSchemaForNames(names) {
    if (names.length === CSV_COLUMNS.length && names.every((name, index) => name === CSV_COLUMNS[index])) {
      return normalizeSchema(ROBOT_POSE_SCHEMA);
    }
    return null;
  }

  function looksLikeBool(values) {
    if (!values.length) return false;
    const allowed = new Set(["0", "1", "true", "false", "yes", "no", "y", "n"]);
    return values.every((value) => allowed.has(value.trim().toLowerCase()));
  }

  function inferSchema(rows) {
    if (!rows.length) return [];
    const names = [];
    const seen = new Set();
    rows.forEach((row) => {
      Object.keys(row).forEach((key) => {
        if (!seen.has(key)) {
          seen.add(key);
          names.push(key);
        }
      });
    });
    const known = knownSchemaForNames(names);
    if (known) return known;

    return names.map((name) => {
      const values = rows.map((row) => asText(row[name] ?? ""));
      const present = values.filter((value) => value !== "");
      const nullable = present.length !== values.length;
      const unit = knownUnitForName(name);

      if (name.endsWith("_visible") || looksLikeBool(present)) {
        return { name, kind: "bool", unit, scale: 1, nullable };
      }

      let numericOk = present.length > 0;
      let maxDecimals = 0;
      const numericValues = [];
      for (let i = 0; i < present.length; i += 1) {
        const value = Number(present[i]);
        if (!Number.isFinite(value)) {
          numericOk = false;
          break;
        }
        numericValues.push(value);
        if (present[i].includes(".")) {
          maxDecimals = Math.max(maxDecimals, present[i].split(".", 2)[1].replace(/0+$/, "").length);
        }
      }
      if (numericOk && present.length) {
        if (numericValues.every((value) => Number.isInteger(value)) && maxDecimals === 0) {
          return { name, kind: "int", unit, scale: 1, nullable };
        }
        if (maxDecimals <= 6) {
          return { name, kind: "scaled_int", unit, scale: Math.pow(10, maxDecimals), nullable };
        }
        return { name, kind: "float64", unit, scale: 1, nullable };
      }
      return { name, kind: "string", unit, scale: 1, nullable };
    });
  }

  function normalizeRow(row, schema) {
    const normalized = {};
    schema.forEach((column) => {
      normalized[column.name] = row[column.name] ?? "";
    });
    return normalized;
  }

  function isPresent(column, value) {
    return column.nullable ? asText(value) !== "" : true;
  }

  function schemaToBytes(schema) {
    const out = [];
    encodeVarUInt(schema.length, out);
    schema.forEach((column) => {
      const desc = [];
      encodeString(column.name, desc);
      desc.push(KIND_CODES[column.kind]);
      desc.push(column.nullable ? FLAG_NULLABLE : 0);
      encodeVarUInt(Math.max(1, Number(column.scale || 1)), desc);
      encodeString(column.unit || "", desc);
      encodeVarUInt(0, desc);
      encodeVarUInt(desc.length, out);
      for (let i = 0; i < desc.length; i += 1) out.push(desc[i]);
    });
    return Uint8Array.from(out);
  }

  function schemaFromBytes(data) {
    const cursor = { pos: 0 };
    const count = decodeVarUInt(data, cursor);
    const schema = [];
    for (let i = 0; i < count; i += 1) {
      const descLen = decodeVarUInt(data, cursor);
      const end = cursor.pos + descLen;
      if (end > data.length) throw new Error("Unexpected end of JLOG schema");
      const name = decodeString(data, cursor);
      if (cursor.pos + 2 > end) throw new Error("Unexpected end of JLOG column descriptor");
      const kindCode = data[cursor.pos];
      cursor.pos += 1;
      const flags = data[cursor.pos];
      cursor.pos += 1;
      const scale = decodeVarUInt(data, cursor);
      const unit = decodeString(data, cursor);
      if (cursor.pos < end) decodeVarUInt(data, cursor);
      cursor.pos = end;
      if (!Object.prototype.hasOwnProperty.call(KIND_NAMES, kindCode)) {
        throw new Error(`Unsupported JLOG column kind: ${kindCode}`);
      }
      schema.push({
        name,
        kind: KIND_NAMES[kindCode],
        unit,
        scale: Math.max(1, Number(scale || 1)),
        nullable: Boolean(flags & FLAG_NULLABLE),
      });
    }
    return schema;
  }

  function packBits(values) {
    if (!values.length) return new Uint8Array(0);
    const out = new Uint8Array(Math.ceil(values.length / 8));
    values.forEach((value, idx) => {
      if (value) out[idx >> 3] |= 1 << (idx & 7);
    });
    return out;
  }

  function unpackBits(data, count) {
    const values = [];
    for (let idx = 0; idx < count; idx += 1) {
      values.push(Boolean(data[idx >> 3] & (1 << (idx & 7))));
    }
    return values;
  }

  function encodeBlockRows(rows, schema) {
    const normalizedRows = rows.map((row) => normalizeRow(row, schema));
    const stringIndices = schema.map((column, idx) => column.kind === "string" ? idx : -1).filter((idx) => idx >= 0);
    const nullableIndices = schema.map((column, idx) => column.nullable ? idx : -1).filter((idx) => idx >= 0);
    const boolIndices = schema.map((column, idx) => column.kind === "bool" ? idx : -1).filter((idx) => idx >= 0);

    const dictByCol = new Map();
    stringIndices.forEach((colIdx) => {
      const column = schema[colIdx];
      const seen = new Set();
      const ordered = [];
      normalizedRows.forEach((row) => {
        const value = asText(row[column.name]);
        if (!isPresent(column, value)) return;
        if (!seen.has(value)) {
          seen.add(value);
          ordered.push(value);
        }
      });
      dictByCol.set(colIdx, ordered);
    });

    const out = [];
    stringIndices.forEach((colIdx) => {
      const values = dictByCol.get(colIdx);
      encodeVarUInt(values.length, out);
      values.forEach((value) => encodeString(value, out));
    });

    const prevNumeric = new Array(schema.length).fill(0);
    normalizedRows.forEach((row) => {
      const presenceBits = nullableIndices.map((colIdx) => isPresent(schema[colIdx], row[schema[colIdx].name]));
      const boolBits = boolIndices.map((colIdx) => {
        const column = schema[colIdx];
        return isPresent(column, row[column.name]) && boolFromValue(row[column.name]);
      });
      const packedPresence = packBits(presenceBits);
      const packedBool = packBits(boolBits);
      for (let i = 0; i < packedPresence.length; i += 1) out.push(packedPresence[i]);
      for (let i = 0; i < packedBool.length; i += 1) out.push(packedBool[i]);

      schema.forEach((column, colIdx) => {
        const value = row[column.name];
        const present = isPresent(column, value);
        if (column.nullable && !present) return;
        if (column.kind === "bool") return;
        if (column.kind === "string") {
          const values = dictByCol.get(colIdx) || [];
          const map = new Map(values.map((entry, idx) => [entry, idx]));
          encodeVarUInt(map.get(asText(value)), out);
          return;
        }
        if (column.kind === "scaled_int") {
          const current = quantizeScaled(parseFloatValue(value), column.scale);
          encodeVarInt(current - prevNumeric[colIdx], out);
          prevNumeric[colIdx] = current;
          return;
        }
        if (column.kind === "int") {
          const current = parseIntValue(value);
          encodeVarInt(current - prevNumeric[colIdx], out);
          prevNumeric[colIdx] = current;
          return;
        }
        if (column.kind === "float64") {
          pushFloat64LE(out, parseFloatValue(value));
          return;
        }
        throw new Error(`Unsupported JLOG kind: ${column.kind}`);
      });
    });
    return Uint8Array.from(out);
  }

  function decodeBlockRows(payload, rowCount, schema) {
    let pos = 0;
    const stringIndices = schema.map((column, idx) => column.kind === "string" ? idx : -1).filter((idx) => idx >= 0);
    const nullableIndices = schema.map((column, idx) => column.nullable ? idx : -1).filter((idx) => idx >= 0);
    const boolIndices = schema.map((column, idx) => column.kind === "bool" ? idx : -1).filter((idx) => idx >= 0);
    const presenceBytesLen = Math.ceil(nullableIndices.length / 8);
    const boolBytesLen = Math.ceil(boolIndices.length / 8);

    const cursor = { pos: 0 };
    const dictByCol = new Map();
    stringIndices.forEach((colIdx) => {
      const count = decodeVarUInt(payload, cursor);
      const values = [];
      for (let i = 0; i < count; i += 1) values.push(decodeString(payload, cursor));
      dictByCol.set(colIdx, values);
    });
    pos = cursor.pos;

    const prevNumeric = new Array(schema.length).fill(0);
    const rows = [];
    for (let rowIdx = 0; rowIdx < rowCount; rowIdx += 1) {
      if (pos + presenceBytesLen + boolBytesLen > payload.length) throw new Error("Unexpected end of JLOG block");
      const presenceBits = unpackBits(payload.slice(pos, pos + presenceBytesLen), nullableIndices.length);
      pos += presenceBytesLen;
      const boolBits = unpackBits(payload.slice(pos, pos + boolBytesLen), boolIndices.length);
      pos += boolBytesLen;

      const row = {};
      const presenceMap = new Map(nullableIndices.map((colIdx, idx) => [colIdx, presenceBits[idx]]));
      const boolMap = new Map(boolIndices.map((colIdx, idx) => [colIdx, boolBits[idx]]));
      const valueCursor = { pos };

      schema.forEach((column, colIdx) => {
        const present = presenceMap.has(colIdx) ? presenceMap.get(colIdx) : true;
        if (!present) {
          row[column.name] = "";
          return;
        }
        if (column.kind === "bool") {
          row[column.name] = boolMap.get(colIdx) ? "1" : "0";
          return;
        }
        if (column.kind === "string") {
          const dictIndex = decodeVarUInt(payload, valueCursor);
          const values = dictByCol.get(colIdx) || [];
          if (dictIndex >= values.length) throw new Error("JLOG string dictionary index out of range");
          row[column.name] = values[dictIndex];
          return;
        }
        if (column.kind === "scaled_int") {
          prevNumeric[colIdx] += decodeVarInt(payload, valueCursor);
          row[column.name] = formatScaled(prevNumeric[colIdx], column.scale);
          return;
        }
        if (column.kind === "int") {
          prevNumeric[colIdx] += decodeVarInt(payload, valueCursor);
          row[column.name] = String(prevNumeric[colIdx]);
          return;
        }
        if (column.kind === "float64") {
          if (valueCursor.pos + 8 > payload.length) throw new Error("Unexpected end of JLOG float value");
          const view = new DataView(payload.buffer, payload.byteOffset + valueCursor.pos, 8);
          row[column.name] = String(view.getFloat64(0, true));
          valueCursor.pos += 8;
          return;
        }
        throw new Error(`Unsupported JLOG kind: ${column.kind}`);
      });
      pos = valueCursor.pos;
      rows.push(row);
    }
    return rows;
  }

  function schemaAsDicts(schema) {
    return normalizeSchema(schema);
  }

  function decodeV1Shots(data, cursor, poses) {
    if (cursor.pos >= data.length) throw new Error("Unexpected end of v1 JLOG shot mask");
    const shotMask = data[cursor.pos];
    cursor.pos += 1;
    const resultByCode = { 0: "", 1: "made", 2: "missed" };
    const goalByCode = { 0: "", 1: "red", 2: "blue" };
    const shots = Array.from({ length: 4 }, () => null);
    for (let robot = 0; robot < 4; robot += 1) {
      if ((shotMask & (1 << robot)) === 0) continue;
      if (cursor.pos >= data.length) throw new Error("Unexpected end of v1 JLOG shot metadata");
      const meta = data[cursor.pos];
      cursor.pos += 1;
      const resultCode = meta & 0x03;
      const goalCode = (meta >> 2) & 0x03;
      const result = resultCode === 3 ? decodeString(data, cursor) : (resultByCode[resultCode] || "");
      const goal = goalCode === 3 ? decodeString(data, cursor) : (goalByCode[goalCode] || "");
      const dxQ = decodeVarInt(data, cursor);
      const dyQ = decodeVarInt(data, cursor);
      shots[robot] = {
        result,
        goal,
        shotXQ: poses[robot][0] + dxQ,
        shotYQ: poses[robot][1] + dyQ,
      };
    }
    return shots;
  }

  function stateV1ToRow(state) {
    const row = {};
    CSV_COLUMNS.forEach((key) => { row[key] = ""; });
    row.timestamp_s = formatScaled(state.timestampMs, 1000);
    state.poses.forEach((pose, robot) => {
      row[`robot${robot}_x_in`] = formatScaled(pose[0], 100);
      row[`robot${robot}_y_in`] = formatScaled(pose[1], 100);
      row[`robot${robot}_heading_rad`] = formatScaled(pose[2], 10000);
      row[`robot${robot}_visible`] = (state.visibleMask & (1 << robot)) ? "1" : "0";
    });
    state.shots.forEach((shot, robot) => {
      if (!shot) return;
      row[`robot${robot}_shot_result`] = shot.result;
      row[`robot${robot}_shot_x_in`] = formatScaled(shot.shotXQ, 100);
      row[`robot${robot}_shot_y_in`] = formatScaled(shot.shotYQ, 100);
      row[`robot${robot}_shot_goal`] = shot.goal;
    });
    return row;
  }

  function decodeTable(buffer, options = {}) {
    const strict = options.strict === true;
    const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
    if (bytes.length < 8) throw new Error("File is too short to be a JLOG");
    const magic = textDecoder.decode(bytes.slice(0, 8));

    if (magic === MAGIC_V1) {
      if (bytes.length < HEADER_V1_SIZE) throw new Error("File is too short to be a v1 JLOG");
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      const versionMajor = view.getUint16(8, true);
      const versionMinor = view.getUint16(10, true);
      if (versionMajor !== 1) throw new Error(`Unsupported v1 JLOG major version: ${versionMajor}`);
      const rows = [];
      let pos = HEADER_V1_SIZE;
      while (pos < bytes.length) {
        if (pos + BLOCK_HEADER_SIZE > bytes.length) {
          if (strict) throw new Error("Truncated v1 JLOG block header");
          break;
        }
        const blockMagic = textDecoder.decode(bytes.slice(pos, pos + 4));
        if (blockMagic !== BLOCK_MAGIC) {
          if (strict) throw new Error("Invalid v1 JLOG block marker");
          break;
        }
        const payloadLen = view.getUint32(pos + 4, true);
        const rowCount = view.getUint32(pos + 8, true);
        const expectedCrc = view.getUint32(pos + 12, true);
        pos += BLOCK_HEADER_SIZE;
        if (pos + payloadLen > bytes.length) {
          if (strict) throw new Error("Truncated v1 JLOG block payload");
          break;
        }
        const payload = bytes.slice(pos, pos + payloadLen);
        pos += payloadLen;
        if (crc32(payload) !== expectedCrc) {
          if (strict) throw new Error("v1 JLOG block CRC mismatch");
          break;
        }
        const cursor = { pos: 0 };
        let prev = null;
        try {
          for (let rowIdx = 0; rowIdx < rowCount; rowIdx += 1) {
            let state;
            if (rowIdx === 0) {
              if (cursor.pos + 5 > payload.length) throw new Error("Unexpected end of v1 absolute row");
              const rowView = new DataView(payload.buffer, payload.byteOffset + cursor.pos, payload.length - cursor.pos);
              const timestampMs = rowView.getUint32(0, true);
              cursor.pos += 4;
              const visibleMask = payload[cursor.pos];
              cursor.pos += 1;
              const poses = [];
              for (let robot = 0; robot < 4; robot += 1) {
                if (cursor.pos + 6 > payload.length) throw new Error("Unexpected end of v1 pose");
                const poseView = new DataView(payload.buffer, payload.byteOffset + cursor.pos, 6);
                poses.push([
                  poseView.getInt16(0, true),
                  poseView.getInt16(2, true),
                  poseView.getInt16(4, true),
                ]);
                cursor.pos += 6;
              }
              const shots = decodeV1Shots(payload, cursor, poses);
              state = { timestampMs, visibleMask, poses, shots };
            } else {
              const dtMs = decodeVarUInt(payload, cursor);
              if (cursor.pos >= payload.length) throw new Error("Unexpected end of v1 visible mask");
              const visibleMask = payload[cursor.pos];
              cursor.pos += 1;
              const poses = [];
              for (let robot = 0; robot < 4; robot += 1) {
                const dxQ = decodeVarInt(payload, cursor);
                const dyQ = decodeVarInt(payload, cursor);
                const dHeadingQ = decodeVarInt(payload, cursor);
                poses.push([
                  prev.poses[robot][0] + dxQ,
                  prev.poses[robot][1] + dyQ,
                  prev.poses[robot][2] + dHeadingQ,
                ]);
              }
              const shots = decodeV1Shots(payload, cursor, poses);
              state = { timestampMs: prev.timestampMs + dtMs, visibleMask, poses, shots };
            }
            rows.push(stateV1ToRow(state));
            prev = state;
          }
        } catch (error) {
          if (strict) throw error;
          break;
        }
      }
      return {
        versionMajor,
        versionMinor,
        schema: schemaAsDicts(ROBOT_POSE_SCHEMA),
        rows,
      };
    }

    if (magic !== MAGIC_V2) throw new Error("Not a JLOG file");
    if (bytes.length < HEADER_V2_PREFIX_SIZE) throw new Error("File is too short to be a JLOG");

    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const versionMajor = view.getUint16(8, true);
    const versionMinor = view.getUint16(10, true);
    if (versionMajor !== SUPPORTED_MAJOR_V2) throw new Error(`Unsupported JLOG major version: ${versionMajor}`);
    const blockRows = view.getUint32(12, true);
    const schemaLen = view.getUint32(16, true);
    if (HEADER_V2_PREFIX_SIZE + schemaLen > bytes.length) throw new Error("Truncated JLOG schema");
    const schema = schemaFromBytes(bytes.slice(HEADER_V2_PREFIX_SIZE, HEADER_V2_PREFIX_SIZE + schemaLen));

    const rows = [];
    let pos = HEADER_V2_PREFIX_SIZE + schemaLen;
    while (pos < bytes.length) {
      if (pos + BLOCK_HEADER_SIZE > bytes.length) {
        if (strict) throw new Error("Truncated JLOG block header");
        break;
      }
      const blockMagic = textDecoder.decode(bytes.slice(pos, pos + 4));
      if (blockMagic !== BLOCK_MAGIC) {
        if (strict) throw new Error("Invalid JLOG block marker");
        break;
      }
      const payloadLen = view.getUint32(pos + 4, true);
      const rowCount = view.getUint32(pos + 8, true);
      const expectedCrc = view.getUint32(pos + 12, true);
      pos += BLOCK_HEADER_SIZE;
      if (pos + payloadLen > bytes.length) {
        if (strict) throw new Error("Truncated JLOG block payload");
        break;
      }
      const payload = bytes.slice(pos, pos + payloadLen);
      pos += payloadLen;
      if (crc32(payload) !== expectedCrc) {
        if (strict) throw new Error("JLOG block CRC mismatch");
        break;
      }
      try {
        rows.push(...decodeBlockRows(payload, rowCount, schema));
      } catch (error) {
        if (strict) throw error;
        break;
      }
    }

    return {
      versionMajor,
      versionMinor,
      blockRows,
      schema: schemaAsDicts(schema),
      rows,
    };
  }

  function decodeRows(buffer, options = {}) {
    return decodeTable(buffer, options).rows;
  }

  function encodeRows(rows, options = {}) {
    const blockRows = Math.max(1, Number(options.blockRows) || DEFAULT_BLOCK_ROWS);
    const schema = normalizeSchema(options.schema || inferSchema(rows));
    const schemaBytes = schemaToBytes(schema);
    const out = [];

    const magicBytes = textEncoder.encode(MAGIC_V2);
    for (let i = 0; i < magicBytes.length; i += 1) out.push(magicBytes[i]);
    pushUInt16LE(out, SUPPORTED_MAJOR_V2);
    pushUInt16LE(out, 0);
    pushUInt32LE(out, blockRows);
    pushUInt32LE(out, schemaBytes.length);
    for (let i = 0; i < schemaBytes.length; i += 1) out.push(schemaBytes[i]);

    for (let offset = 0; offset < rows.length; offset += blockRows) {
      const blockRowsSlice = rows.slice(offset, offset + blockRows);
      const payload = encodeBlockRows(blockRowsSlice, schema);
      const blockMagicBytes = textEncoder.encode(BLOCK_MAGIC);
      for (let i = 0; i < blockMagicBytes.length; i += 1) out.push(blockMagicBytes[i]);
      pushUInt32LE(out, payload.length);
      pushUInt32LE(out, blockRowsSlice.length);
      pushUInt32LE(out, crc32(payload));
      for (let i = 0; i < payload.length; i += 1) out.push(payload[i]);
    }

    return Uint8Array.from(out);
  }

  function rowsToBlob(rows, options = {}) {
    return new Blob([encodeRows(rows, options)], { type: "application/octet-stream" });
  }

  function sniffBuffer(buffer) {
    const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
    if (bytes.length < 8) return false;
    const magic = textDecoder.decode(bytes.slice(0, 8));
    return magic === MAGIC_V1 || magic === MAGIC_V2;
  }

  window.JuiceLog = {
    CSV_COLUMNS,
    ROBOT_POSE_SCHEMA: normalizeSchema(ROBOT_POSE_SCHEMA),
    decodeRows,
    decodeTable,
    encodeRows,
    inferSchema,
    rowsToBlob,
    sniffBuffer,
  };
}());
