import struct, sys

path = "./output/match_log.wpilog"
with open(path, "rb") as f:
    data = f.read()

# Check header
print("Magic:", data[:6])
major, minor = data[6], data[7]
extra_len = struct.unpack_from("<I", data, 8)[0]
print(f"Version: {major}.{minor}, extra header bytes: {extra_len}")

pos = 12 + extra_len
records = []

while pos < len(data):
    if pos >= len(data): break
    bitfield = data[pos]; pos += 1
    eid_len  = (bitfield & 0x03) + 1
    size_len = ((bitfield >> 2) & 0x03) + 1
    ts_len   = ((bitfield >> 4) & 0x07) + 1

    eid  = int.from_bytes(data[pos:pos+eid_len],  "little"); pos += eid_len
    size = int.from_bytes(data[pos:pos+size_len], "little"); pos += size_len
    ts   = int.from_bytes(data[pos:pos+ts_len],   "little"); pos += ts_len
    payload = data[pos:pos+size]; pos += size

    if eid == 0:
        ctrl_type = payload[0]
        entry_id  = struct.unpack_from("<I", payload, 1)[0]
        off = 5
        name_len = struct.unpack_from("<I", payload, off)[0]; off += 4
        name = payload[off:off+name_len].decode(); off += name_len
        type_len = struct.unpack_from("<I", payload, off)[0]; off += 4
        type_str = payload[off:off+type_len].decode()
        print(f"  START: id={entry_id} name={name!r} type={type_str!r}")
    else:
        if len(records) < 20:  # only show first 20 data records
            if size == 24:
                x, y, r = struct.unpack_from("<ddd", payload)
                print(f"  DATA:  eid={eid} ts={ts} size={size} → x={x:.4f} y={y:.4f} rot={r:.4f}")
            else:
                print(f"  DATA:  eid={eid} ts={ts} size={size} raw={payload.hex()}")
    records.append((eid, ts, size))

print(f"\nTotal records: {len(records)}")