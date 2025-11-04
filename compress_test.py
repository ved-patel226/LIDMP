import numpy as np
import zlib

arr = np.random.randint(0, 4096, size=(112, 112), dtype=np.uint16)


def pack_12bit(arr):
    flat = arr.flatten()
    packed = bytearray()
    buffer = 0
    bits_in_buffer = 0

    for value in flat:
        buffer = (buffer << 12) | value
        bits_in_buffer += 12
        while bits_in_buffer >= 8:
            bits_in_buffer -= 8
            packed.append((buffer >> bits_in_buffer) & 0xFF)

    if bits_in_buffer > 0:
        packed.append((buffer << (8 - bits_in_buffer)) & 0xFF)

    return bytes(packed)


packed_bytes = pack_12bit(arr)


print(f"Original size: {arr.nbytes} bytes")
print(f"Packed size: {len(packed_bytes)} bytes")


def unpack_12bit(data, length):
    result = []
    buffer = 0
    bits_in_buffer = 0
    for byte in data:
        buffer = (buffer << 8) | byte
        bits_in_buffer += 8
        while bits_in_buffer >= 12:
            bits_in_buffer -= 12
            result.append((buffer >> bits_in_buffer) & 0xFFF)
    return np.array(result[:length], dtype=np.uint16).reshape((112, 112))


restored_arr = unpack_12bit(packed_bytes, 112 * 112)
print("Arrays match:", np.array_equal(arr, restored_arr))
