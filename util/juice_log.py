import csv
import struct
import zlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

CSV_COLUMNS = [
    "timestamp_s",
    "robot0_x_in", "robot0_y_in", "robot0_heading_rad", "robot0_visible",
    "robot1_x_in", "robot1_y_in", "robot1_heading_rad", "robot1_visible",
    "robot2_x_in", "robot2_y_in", "robot2_heading_rad", "robot2_visible",
    "robot3_x_in", "robot3_y_in", "robot3_heading_rad", "robot3_visible",
    "robot0_shot_result", "robot0_shot_x_in", "robot0_shot_y_in", "robot0_shot_goal",
    "robot1_shot_result", "robot1_shot_x_in", "robot1_shot_y_in", "robot1_shot_goal",
    "robot2_shot_result", "robot2_shot_x_in", "robot2_shot_y_in", "robot2_shot_goal",
    "robot3_shot_result", "robot3_shot_x_in", "robot3_shot_y_in", "robot3_shot_goal",
]

MAGIC_V1 = b"JLOGv001"
MAGIC_V2 = b"JLOGv002"
BLOCK_MAGIC = b"JBLK"
SUPPORTED_MAJOR_V2 = 2
SUPPORTED_MINOR_V2 = 0
DEFAULT_BLOCK_ROWS = 128

KIND_SCALED_INT = 1
KIND_BOOL = 2
KIND_STRING = 3
KIND_INT = 4
KIND_FLOAT64 = 5

KIND_NAMES = {
    KIND_SCALED_INT: "scaled_int",
    KIND_BOOL: "bool",
    KIND_STRING: "string",
    KIND_INT: "int",
    KIND_FLOAT64: "float64",
}
KIND_CODES = {name: code for code, name in KIND_NAMES.items()}

FLAG_NULLABLE = 0x01

HEADER_V1_STRUCT = struct.Struct("<8sHHHHHHH")
HEADER_V2_PREFIX_STRUCT = struct.Struct("<8sHHII")
BLOCK_HEADER_STRUCT = struct.Struct("<4sIII")
POSE_V1_STRUCT = struct.Struct("<hhh")


@dataclass(frozen=True)
class JuiceLogColumn:
    name: str
    kind: str
    unit: str = ""
    scale: int = 1
    nullable: bool = False


ROBOT_POSE_SCHEMA = [
    JuiceLogColumn("timestamp_s", "scaled_int", unit="s", scale=1000, nullable=False),
    JuiceLogColumn("robot0_x_in", "scaled_int", unit="in", scale=100, nullable=False),
    JuiceLogColumn("robot0_y_in", "scaled_int", unit="in", scale=100, nullable=False),
    JuiceLogColumn("robot0_heading_rad", "scaled_int", unit="rad", scale=10000, nullable=False),
    JuiceLogColumn("robot0_visible", "bool", unit="", scale=1, nullable=False),
    JuiceLogColumn("robot1_x_in", "scaled_int", unit="in", scale=100, nullable=False),
    JuiceLogColumn("robot1_y_in", "scaled_int", unit="in", scale=100, nullable=False),
    JuiceLogColumn("robot1_heading_rad", "scaled_int", unit="rad", scale=10000, nullable=False),
    JuiceLogColumn("robot1_visible", "bool", unit="", scale=1, nullable=False),
    JuiceLogColumn("robot2_x_in", "scaled_int", unit="in", scale=100, nullable=False),
    JuiceLogColumn("robot2_y_in", "scaled_int", unit="in", scale=100, nullable=False),
    JuiceLogColumn("robot2_heading_rad", "scaled_int", unit="rad", scale=10000, nullable=False),
    JuiceLogColumn("robot2_visible", "bool", unit="", scale=1, nullable=False),
    JuiceLogColumn("robot3_x_in", "scaled_int", unit="in", scale=100, nullable=False),
    JuiceLogColumn("robot3_y_in", "scaled_int", unit="in", scale=100, nullable=False),
    JuiceLogColumn("robot3_heading_rad", "scaled_int", unit="rad", scale=10000, nullable=False),
    JuiceLogColumn("robot3_visible", "bool", unit="", scale=1, nullable=False),
    JuiceLogColumn("robot0_shot_result", "string", unit="", scale=1, nullable=True),
    JuiceLogColumn("robot0_shot_x_in", "scaled_int", unit="in", scale=100, nullable=True),
    JuiceLogColumn("robot0_shot_y_in", "scaled_int", unit="in", scale=100, nullable=True),
    JuiceLogColumn("robot0_shot_goal", "string", unit="", scale=1, nullable=True),
    JuiceLogColumn("robot1_shot_result", "string", unit="", scale=1, nullable=True),
    JuiceLogColumn("robot1_shot_x_in", "scaled_int", unit="in", scale=100, nullable=True),
    JuiceLogColumn("robot1_shot_y_in", "scaled_int", unit="in", scale=100, nullable=True),
    JuiceLogColumn("robot1_shot_goal", "string", unit="", scale=1, nullable=True),
    JuiceLogColumn("robot2_shot_result", "string", unit="", scale=1, nullable=True),
    JuiceLogColumn("robot2_shot_x_in", "scaled_int", unit="in", scale=100, nullable=True),
    JuiceLogColumn("robot2_shot_y_in", "scaled_int", unit="in", scale=100, nullable=True),
    JuiceLogColumn("robot2_shot_goal", "string", unit="", scale=1, nullable=True),
    JuiceLogColumn("robot3_shot_result", "string", unit="", scale=1, nullable=True),
    JuiceLogColumn("robot3_shot_x_in", "scaled_int", unit="in", scale=100, nullable=True),
    JuiceLogColumn("robot3_shot_y_in", "scaled_int", unit="in", scale=100, nullable=True),
    JuiceLogColumn("robot3_shot_goal", "string", unit="", scale=1, nullable=True),
]


def _as_text(value) -> str:
    if value is None:
        return ""
    return str(value)


def _encode_varuint(value: int, out: bytearray) -> None:
    if value < 0:
        raise ValueError("varuint cannot encode negative values")
    n = int(value)
    while True:
        byte = n & 0x7F
        n >>= 7
        if n:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return


def _decode_varuint(data: bytes, pos: int) -> Tuple[int, int]:
    shift = 0
    value = 0
    while True:
        if pos >= len(data):
            raise EOFError("unexpected end of block while reading varuint")
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, pos
        shift += 7
        if shift > 35:
            raise ValueError("varuint is too large")


def _zigzag_encode(value: int) -> int:
    n = int(value)
    return (n << 1) ^ (n >> 31)


def _zigzag_decode(value: int) -> int:
    n = int(value)
    return (n >> 1) ^ -(n & 1)


def _encode_varint(value: int, out: bytearray) -> None:
    _encode_varuint(_zigzag_encode(value), out)


def _decode_varint(data: bytes, pos: int) -> Tuple[int, int]:
    value, pos = _decode_varuint(data, pos)
    return _zigzag_decode(value), pos


def _encode_string(text: str, out: bytearray) -> None:
    payload = text.encode("utf-8")
    _encode_varuint(len(payload), out)
    out.extend(payload)


def _decode_string(data: bytes, pos: int) -> Tuple[str, int]:
    size, pos = _decode_varuint(data, pos)
    end = pos + size
    if end > len(data):
        raise EOFError("unexpected end of block while reading string")
    return data[pos:end].decode("utf-8"), end


def _bool_from_value(value) -> bool:
    text = _as_text(value).strip().lower()
    return text in ("1", "true", "yes", "y")


def _parse_float(value, default=0.0) -> float:
    text = _as_text(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _parse_int(value, default=0) -> int:
    text = _as_text(value).strip()
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        try:
            return int(round(float(text)))
        except ValueError:
            return default


def _quantize_scaled(value, scale: int) -> int:
    return int(round(float(value) * scale))


def _decimals_from_scale(scale: int) -> int:
    if scale <= 1:
        return 0
    decimals = 0
    value = int(scale)
    while value > 1 and value % 10 == 0:
        value //= 10
        decimals += 1
    return decimals if value == 1 else 6


def _format_scaled(value: int, scale: int) -> str:
    decimals = _decimals_from_scale(scale)
    return ("{:.%df}" % decimals).format(value / scale)


def _normalize_row(row: Dict[str, object], schema: Sequence[JuiceLogColumn]) -> Dict[str, object]:
    normalized = {}
    for column in schema:
        normalized[column.name] = row.get(column.name, "")
    return normalized


def _is_present(column: JuiceLogColumn, value) -> bool:
    if not column.nullable:
        return True
    return _as_text(value) != ""


def _known_unit_for_name(name: str) -> str:
    if name == "timestamp_s":
        return "s"
    if name.endswith("_heading_rad"):
        return "rad"
    if name.endswith("_x_in") or name.endswith("_y_in"):
        return "in"
    return ""


def _known_schema_for_names(names: Sequence[str]) -> Optional[List[JuiceLogColumn]]:
    if list(names) == CSV_COLUMNS:
        return list(ROBOT_POSE_SCHEMA)
    return None


def _looks_like_bool(values: Sequence[str]) -> bool:
    if not values:
        return False
    allowed = {"0", "1", "true", "false", "yes", "no", "y", "n"}
    return all(v.strip().lower() in allowed for v in values)


def _infer_schema(rows: Sequence[Dict[str, object]]) -> List[JuiceLogColumn]:
    if not rows:
        return []
    names = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                names.append(key)
                seen.add(key)
    known = _known_schema_for_names(names)
    if known is not None:
        return known

    schema = []
    for name in names:
        values = [_as_text(row.get(name, "")) for row in rows]
        present_values = [v for v in values if v != ""]
        nullable = len(present_values) != len(values)
        unit = _known_unit_for_name(name)

        if name.endswith("_visible") or _looks_like_bool(present_values):
            schema.append(JuiceLogColumn(name, "bool", unit=unit, scale=1, nullable=nullable))
            continue

        numeric_values = []
        numeric_ok = True
        max_decimals = 0
        for value in present_values:
            try:
                numeric_values.append(float(value))
                if "." in value:
                    max_decimals = max(max_decimals, len(value.split(".", 1)[1].rstrip("0")))
            except ValueError:
                numeric_ok = False
                break

        if numeric_ok and present_values:
            if all(float(v).is_integer() for v in numeric_values) and max_decimals == 0:
                schema.append(JuiceLogColumn(name, "int", unit=unit, scale=1, nullable=nullable))
            elif max_decimals <= 6:
                scale = 10 ** max_decimals if max_decimals > 0 else 1
                schema.append(JuiceLogColumn(name, "scaled_int", unit=unit, scale=scale, nullable=nullable))
            else:
                schema.append(JuiceLogColumn(name, "float64", unit=unit, scale=1, nullable=nullable))
            continue

        schema.append(JuiceLogColumn(name, "string", unit=unit, scale=1, nullable=nullable))
    return schema


def _schema_to_bytes(schema: Sequence[JuiceLogColumn]) -> bytes:
    out = bytearray()
    _encode_varuint(len(schema), out)
    for column in schema:
        desc = bytearray()
        _encode_string(column.name, desc)
        desc.append(KIND_CODES[column.kind])
        flags = FLAG_NULLABLE if column.nullable else 0
        desc.append(flags)
        _encode_varuint(max(1, int(column.scale or 1)), desc)
        _encode_string(column.unit, desc)
        _encode_varuint(0, desc)
        _encode_varuint(len(desc), out)
        out.extend(desc)
    return bytes(out)


def _schema_from_bytes(data: bytes) -> List[JuiceLogColumn]:
    pos = 0
    count, pos = _decode_varuint(data, pos)
    schema = []
    for _ in range(count):
        desc_len, pos = _decode_varuint(data, pos)
        end = pos + desc_len
        if end > len(data):
            raise EOFError("unexpected end of JLOG schema")
        name, pos = _decode_string(data, pos)
        if pos + 2 > end:
            raise EOFError("unexpected end of JLOG column descriptor")
        kind_code = data[pos]
        pos += 1
        flags = data[pos]
        pos += 1
        scale, pos = _decode_varuint(data, pos)
        unit, pos = _decode_string(data, pos)
        if pos < end:
            _reserved_len, pos = _decode_varuint(data, pos)
        pos = end
        if kind_code not in KIND_NAMES:
            raise ValueError("unsupported JLOG column kind: {}".format(kind_code))
        schema.append(JuiceLogColumn(
            name=name,
            kind=KIND_NAMES[kind_code],
            unit=unit,
            scale=max(1, int(scale or 1)),
            nullable=bool(flags & FLAG_NULLABLE),
        ))
    return schema


def _pack_bits(values: Sequence[bool]) -> bytes:
    if not values:
        return b""
    out = bytearray((len(values) + 7) // 8)
    for idx, value in enumerate(values):
        if value:
            out[idx // 8] |= 1 << (idx % 8)
    return bytes(out)


def _unpack_bits(data: bytes, count: int) -> List[bool]:
    values = []
    for idx in range(count):
        values.append(bool(data[idx // 8] & (1 << (idx % 8))))
    return values


def _encode_block_rows(rows: Sequence[Dict[str, object]], schema: Sequence[JuiceLogColumn]) -> bytes:
    normalized_rows = [_normalize_row(row, schema) for row in rows]
    string_indices = [idx for idx, column in enumerate(schema) if column.kind == "string"]
    nullable_indices = [idx for idx, column in enumerate(schema) if column.nullable]
    bool_indices = [idx for idx, column in enumerate(schema) if column.kind == "bool"]

    dict_by_col = {}
    for col_idx in string_indices:
        column = schema[col_idx]
        ordered = []
        seen = set()
        for row in normalized_rows:
            value = _as_text(row[column.name])
            if not _is_present(column, value):
                continue
            if value not in seen:
                seen.add(value)
                ordered.append(value)
        dict_by_col[col_idx] = ordered

    out = bytearray()
    for col_idx in string_indices:
        values = dict_by_col[col_idx]
        _encode_varuint(len(values), out)
        for value in values:
            _encode_string(value, out)

    prev_numeric = [0] * len(schema)
    for row in normalized_rows:
        presence_bits = [_is_present(schema[col_idx], row[schema[col_idx].name]) for col_idx in nullable_indices]
        out.extend(_pack_bits(presence_bits))

        bool_bits = []
        for col_idx in bool_indices:
            column = schema[col_idx]
            present = _is_present(column, row[column.name])
            bool_bits.append(present and _bool_from_value(row[column.name]))
        out.extend(_pack_bits(bool_bits))

        for col_idx, column in enumerate(schema):
            value = row[column.name]
            present = _is_present(column, value)
            if column.nullable and not present:
                continue
            if column.kind == "bool":
                continue
            if column.kind == "string":
                dictionary = dict_by_col[col_idx]
                value_to_index = {text: idx for idx, text in enumerate(dictionary)}
                _encode_varuint(value_to_index[_as_text(value)], out)
                continue
            if column.kind == "scaled_int":
                current = _quantize_scaled(_parse_float(value), column.scale)
                _encode_varint(current - prev_numeric[col_idx], out)
                prev_numeric[col_idx] = current
                continue
            if column.kind == "int":
                current = _parse_int(value)
                _encode_varint(current - prev_numeric[col_idx], out)
                prev_numeric[col_idx] = current
                continue
            if column.kind == "float64":
                out.extend(struct.pack("<d", _parse_float(value)))
                continue
            raise ValueError("unsupported JLOG kind: {}".format(column.kind))
    return bytes(out)


def _decode_block_rows(payload: bytes, row_count: int, schema: Sequence[JuiceLogColumn]) -> List[Dict[str, str]]:
    pos = 0
    string_indices = [idx for idx, column in enumerate(schema) if column.kind == "string"]
    nullable_indices = [idx for idx, column in enumerate(schema) if column.nullable]
    bool_indices = [idx for idx, column in enumerate(schema) if column.kind == "bool"]
    presence_bytes_len = (len(nullable_indices) + 7) // 8
    bool_bytes_len = (len(bool_indices) + 7) // 8

    dict_by_col = {}
    for col_idx in string_indices:
        count, pos = _decode_varuint(payload, pos)
        values = []
        for _ in range(count):
            value, pos = _decode_string(payload, pos)
            values.append(value)
        dict_by_col[col_idx] = values

    prev_numeric = [0] * len(schema)
    rows = []
    for _row_idx in range(row_count):
        if pos + presence_bytes_len + bool_bytes_len > len(payload):
            raise EOFError("unexpected end of JLOG block")
        presence_bits = _unpack_bits(payload[pos:pos + presence_bytes_len], len(nullable_indices))
        pos += presence_bytes_len
        bool_bits = _unpack_bits(payload[pos:pos + bool_bytes_len], len(bool_indices))
        pos += bool_bytes_len

        row = {}
        presence_map = {nullable_indices[idx]: value for idx, value in enumerate(presence_bits)}
        bool_map = {bool_indices[idx]: value for idx, value in enumerate(bool_bits)}

        for col_idx, column in enumerate(schema):
            present = presence_map.get(col_idx, True)
            if not present:
                row[column.name] = ""
                continue
            if column.kind == "bool":
                row[column.name] = "1" if bool_map.get(col_idx, False) else "0"
                continue
            if column.kind == "string":
                dict_index, pos = _decode_varuint(payload, pos)
                values = dict_by_col[col_idx]
                if dict_index >= len(values):
                    raise ValueError("JLOG string dictionary index out of range")
                row[column.name] = values[dict_index]
                continue
            if column.kind == "scaled_int":
                delta, pos = _decode_varint(payload, pos)
                prev_numeric[col_idx] += delta
                row[column.name] = _format_scaled(prev_numeric[col_idx], column.scale)
                continue
            if column.kind == "int":
                delta, pos = _decode_varint(payload, pos)
                prev_numeric[col_idx] += delta
                row[column.name] = str(prev_numeric[col_idx])
                continue
            if column.kind == "float64":
                if pos + 8 > len(payload):
                    raise EOFError("unexpected end of JLOG float value")
                value = struct.unpack_from("<d", payload, pos)[0]
                pos += 8
                row[column.name] = repr(value)
                continue
            raise ValueError("unsupported JLOG kind: {}".format(column.kind))
        rows.append(row)
    return rows


def _schema_as_dicts(schema: Sequence[JuiceLogColumn]) -> List[Dict[str, object]]:
    return [
        {
            "name": column.name,
            "kind": column.kind,
            "unit": column.unit,
            "scale": column.scale,
            "nullable": column.nullable,
        }
        for column in schema
    ]


def _read_v1_rows(path: str, strict: bool = False) -> List[Dict[str, str]]:
    rows = []
    with open(path, "rb") as f:
        header = f.read(HEADER_V1_STRUCT.size)
        if len(header) < HEADER_V1_STRUCT.size:
            raise ValueError("file is too short to be a v1 JLOG")
        magic, version_major, version_minor, robot_count, pos_scale, heading_scale, time_scale, _block_rows = HEADER_V1_STRUCT.unpack(header)
        if magic != MAGIC_V1:
            raise ValueError("not a v1 JLOG file")
        if version_major != 1:
            raise ValueError("unsupported v1 JLOG major version: {}".format(version_major))
        if version_minor > 0:
            raise ValueError("unsupported v1 JLOG minor version: {}".format(version_minor))
        if robot_count != 4 or pos_scale != 100 or heading_scale != 10000 or time_scale != 1000:
            raise ValueError("unsupported v1 JLOG schema parameters")

        while True:
            block_header = f.read(BLOCK_HEADER_STRUCT.size)
            if not block_header:
                break
            if len(block_header) < BLOCK_HEADER_STRUCT.size:
                if strict:
                    raise EOFError("truncated JLOG block header")
                break
            block_magic, payload_len, row_count, crc32_expected = BLOCK_HEADER_STRUCT.unpack(block_header)
            if block_magic != BLOCK_MAGIC:
                if strict:
                    raise ValueError("invalid v1 JLOG block marker")
                break
            payload = f.read(payload_len)
            if len(payload) < payload_len:
                if strict:
                    raise EOFError("truncated v1 JLOG block payload")
                break
            if (zlib.crc32(payload) & 0xFFFFFFFF) != crc32_expected:
                if strict:
                    raise ValueError("v1 JLOG block CRC mismatch")
                break

            pos = 0
            prev = None
            decoded = []
            try:
                for row_idx in range(row_count):
                    if row_idx == 0:
                        state, pos = _decode_v1_absolute_row(payload, pos)
                    else:
                        state, pos = _decode_v1_delta_row(prev, payload, pos)
                    decoded.append(state)
                    prev = state
            except (EOFError, ValueError):
                if strict:
                    raise
                break
            rows.extend(_v1_state_to_row(state) for state in decoded)
    return rows


def _decode_v1_shots(data: bytes, pos: int, poses) -> Tuple[List[Optional[Dict[str, object]]], int]:
    if pos >= len(data):
        raise EOFError("unexpected end of block while reading v1 shot mask")
    shot_mask = data[pos]
    pos += 1
    shots = [None] * 4
    result_by_code = {0: "", 1: "made", 2: "missed"}
    goal_by_code = {0: "", 1: "red", 2: "blue"}
    for robot_id in range(4):
        if not (shot_mask & (1 << robot_id)):
            continue
        if pos >= len(data):
            raise EOFError("unexpected end of block while reading v1 shot metadata")
        meta = data[pos]
        pos += 1
        result_code = meta & 0x03
        goal_code = (meta >> 2) & 0x03
        if result_code == 3:
            result_text, pos = _decode_string(data, pos)
        else:
            result_text = result_by_code.get(result_code, "")
        if goal_code == 3:
            goal_text, pos = _decode_string(data, pos)
        else:
            goal_text = goal_by_code.get(goal_code, "")
        dx_q, pos = _decode_varint(data, pos)
        dy_q, pos = _decode_varint(data, pos)
        pose_x_q, pose_y_q, _heading_q = poses[robot_id]
        shots[robot_id] = {
            "result": result_text,
            "goal": goal_text,
            "shot_x_q": pose_x_q + dx_q,
            "shot_y_q": pose_y_q + dy_q,
        }
    return shots, pos


def _decode_v1_absolute_row(data: bytes, pos: int) -> Tuple[Dict[str, object], int]:
    if pos + 5 > len(data):
        raise EOFError("unexpected end of block while reading v1 absolute row")
    timestamp_ms = struct.unpack_from("<I", data, pos)[0]
    pos += 4
    visible_mask = data[pos]
    pos += 1
    poses = []
    for _robot_id in range(4):
        if pos + POSE_V1_STRUCT.size > len(data):
            raise EOFError("unexpected end of block while reading v1 pose")
        poses.append(POSE_V1_STRUCT.unpack_from(data, pos))
        pos += POSE_V1_STRUCT.size
    shots, pos = _decode_v1_shots(data, pos, poses)
    return {
        "timestamp_ms": timestamp_ms,
        "visible_mask": visible_mask,
        "poses": poses,
        "shots": shots,
    }, pos


def _decode_v1_delta_row(prev: Dict[str, object], data: bytes, pos: int) -> Tuple[Dict[str, object], int]:
    dt_ms, pos = _decode_varuint(data, pos)
    if pos >= len(data):
        raise EOFError("unexpected end of block while reading v1 visible mask")
    visible_mask = data[pos]
    pos += 1
    poses = []
    for robot_id in range(4):
        dx_q, pos = _decode_varint(data, pos)
        dy_q, pos = _decode_varint(data, pos)
        dheading_q, pos = _decode_varint(data, pos)
        prev_x_q, prev_y_q, prev_heading_q = prev["poses"][robot_id]
        poses.append((prev_x_q + dx_q, prev_y_q + dy_q, prev_heading_q + dheading_q))
    shots, pos = _decode_v1_shots(data, pos, poses)
    return {
        "timestamp_ms": prev["timestamp_ms"] + dt_ms,
        "visible_mask": visible_mask,
        "poses": poses,
        "shots": shots,
    }, pos


def _v1_state_to_row(state: Dict[str, object]) -> Dict[str, str]:
    row = {key: "" for key in CSV_COLUMNS}
    row["timestamp_s"] = _format_scaled(state["timestamp_ms"], 1000)
    for robot_id, (x_q, y_q, heading_q) in enumerate(state["poses"]):
        row["robot{}_x_in".format(robot_id)] = _format_scaled(x_q, 100)
        row["robot{}_y_in".format(robot_id)] = _format_scaled(y_q, 100)
        row["robot{}_heading_rad".format(robot_id)] = _format_scaled(heading_q, 10000)
        row["robot{}_visible".format(robot_id)] = "1" if (state["visible_mask"] & (1 << robot_id)) else "0"
    for robot_id, shot in enumerate(state["shots"]):
        if shot is None:
            continue
        row["robot{}_shot_result".format(robot_id)] = shot["result"]
        row["robot{}_shot_x_in".format(robot_id)] = _format_scaled(shot["shot_x_q"], 100)
        row["robot{}_shot_y_in".format(robot_id)] = _format_scaled(shot["shot_y_q"], 100)
        row["robot{}_shot_goal".format(robot_id)] = shot["goal"]
    return row


class JuiceLogWriter:
    def __init__(
        self,
        path: str,
        block_rows: int = DEFAULT_BLOCK_ROWS,
        schema: Optional[Sequence[JuiceLogColumn]] = None,
    ):
        self.path = path
        self.block_rows = max(1, int(block_rows))
        self._file = open(path, "wb")
        self._rows: List[Dict[str, object]] = []
        self._closed = False
        self._header_written = False
        self._schema = list(schema) if schema is not None else None
        if self._schema is not None:
            self._write_header()

    @property
    def schema(self) -> Optional[List[JuiceLogColumn]]:
        return list(self._schema) if self._schema is not None else None

    def _write_header(self) -> None:
        if self._header_written:
            return
        if self._schema is None:
            raise ValueError("cannot write JLOG header without a schema")
        schema_bytes = _schema_to_bytes(self._schema)
        self._file.write(HEADER_V2_PREFIX_STRUCT.pack(
            MAGIC_V2,
            SUPPORTED_MAJOR_V2,
            SUPPORTED_MINOR_V2,
            self.block_rows,
            len(schema_bytes),
        ))
        self._file.write(schema_bytes)
        self._header_written = True

    def append_row(self, row: Dict[str, object]) -> None:
        if self._schema is None:
            self._schema = _infer_schema([row])
            self._write_header()
        self._rows.append(row)
        if len(self._rows) >= self.block_rows:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        if self._schema is None:
            self._schema = _infer_schema(self._rows)
        self._write_header()
        payload = _encode_block_rows(self._rows, self._schema)
        header = BLOCK_HEADER_STRUCT.pack(
            BLOCK_MAGIC,
            len(payload),
            len(self._rows),
            zlib.crc32(payload) & 0xFFFFFFFF,
        )
        self._file.write(header)
        self._file.write(payload)
        self._file.flush()
        self._rows = []

    def close(self) -> None:
        if self._closed:
            return
        if self._schema is None:
            self._schema = []
        self._write_header()
        self.flush()
        self._file.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def sniff_jlog(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(8) in (MAGIC_V1, MAGIC_V2)
    except OSError:
        return False


def read_table(path: str, strict: bool = False) -> Dict[str, object]:
    with open(path, "rb") as f:
        magic = f.read(8)
    if magic == MAGIC_V1:
        rows = _read_v1_rows(path, strict=strict)
        return {
            "version_major": 1,
            "version_minor": 0,
            "schema": _schema_as_dicts(ROBOT_POSE_SCHEMA),
            "rows": rows,
        }
    if magic != MAGIC_V2:
        raise ValueError("not a JLOG file")

    with open(path, "rb") as f:
        prefix = f.read(HEADER_V2_PREFIX_STRUCT.size)
        if len(prefix) < HEADER_V2_PREFIX_STRUCT.size:
            raise ValueError("file is too short to be a JLOG")
        magic, version_major, version_minor, _block_rows, schema_len = HEADER_V2_PREFIX_STRUCT.unpack(prefix)
        if magic != MAGIC_V2:
            raise ValueError("not a JLOG file")
        if version_major != SUPPORTED_MAJOR_V2:
            raise ValueError("unsupported JLOG major version: {}".format(version_major))
        schema_bytes = f.read(schema_len)
        if len(schema_bytes) < schema_len:
            raise EOFError("truncated JLOG schema")
        schema = _schema_from_bytes(schema_bytes)

        rows = []
        while True:
            block_header = f.read(BLOCK_HEADER_STRUCT.size)
            if not block_header:
                break
            if len(block_header) < BLOCK_HEADER_STRUCT.size:
                if strict:
                    raise EOFError("truncated JLOG block header")
                break
            block_magic, payload_len, row_count, crc_expected = BLOCK_HEADER_STRUCT.unpack(block_header)
            if block_magic != BLOCK_MAGIC:
                if strict:
                    raise ValueError("invalid JLOG block marker")
                break
            payload = f.read(payload_len)
            if len(payload) < payload_len:
                if strict:
                    raise EOFError("truncated JLOG block payload")
                break
            if (zlib.crc32(payload) & 0xFFFFFFFF) != crc_expected:
                if strict:
                    raise ValueError("JLOG block CRC mismatch")
                break
            try:
                rows.extend(_decode_block_rows(payload, row_count, schema))
            except (EOFError, ValueError):
                if strict:
                    raise
                break

    return {
        "version_major": version_major,
        "version_minor": version_minor,
        "schema": _schema_as_dicts(schema),
        "rows": rows,
    }


def read_rows(path: str, strict: bool = False) -> List[Dict[str, str]]:
    return read_table(path, strict=strict)["rows"]


def write_rows(
    path: str,
    rows: Iterable[Dict[str, object]],
    block_rows: int = DEFAULT_BLOCK_ROWS,
    schema: Optional[Sequence[JuiceLogColumn]] = None,
) -> None:
    rows_list = list(rows)
    inferred = list(schema) if schema is not None else _infer_schema(rows_list)
    with JuiceLogWriter(path, block_rows=block_rows, schema=inferred) as writer:
        for row in rows_list:
            writer.append_row(row)


def csv_row_to_list(row: Dict[str, object]) -> List[str]:
    normalized = {key: row.get(key, "") for key in CSV_COLUMNS}
    return [_as_text(normalized[key]) for key in CSV_COLUMNS]


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))
