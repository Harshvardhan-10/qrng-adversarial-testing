# xorshift128 bit generator (Marsaglia variant)
# Produces 1_000_000 bits and writes:
#  - 'xorshift_bits.txt' : ASCII '0'/'1' string of length 1_000_000
#  - 'xorshift_bits.bin' : compact binary file (bits packed into bytes, MSB-first in each byte)

# import struct

def xorshift128_next(state):
    # state is list of four uint32 values [s0, s1, s2, s3]
    t = state[3]
    s = state[0]
    t = (t ^ ((t << 11) & 0xFFFFFFFF)) & 0xFFFFFFFF
    t = (t ^ (t >> 8)) & 0xFFFFFFFF
    state[3] = state[2]
    state[2] = state[1]
    state[1] = s
    state[0] = (t ^ s ^ (s >> 19)) & 0xFFFFFFFF
    return state[0]

def generate_bits(n_bits, seed=None):
    if seed is None:
        # example nonzero seed; change if you want reproducible different stream
        seed = [0x12345678, 0x87654321, 0xDEADBEEF, 0xCAFEBABE]
    state = [s & 0xFFFFFFFF for s in seed]
    if all(x == 0 for x in state):
        raise ValueError("State must not be all zeros")

    bits = []
    while len(bits) < n_bits:
        r = xorshift128_next(state)  # 32-bit output
        # append 32 bits (from most-significant to least-significant)
        bits_from_r = format(r, '032b')  # e.g. '010101...'
        bits.extend(bits_from_r)
    # truncate to requested length
    return bits[:n_bits]

def save_bits_ascii(bits, filename):
    with open(filename, 'w') as f:
        f.write(''.join(bits))

def save_bits_packed(bits, filename):
    # pack 8 bits per byte, MSB-first in each byte
    b = bytearray()
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i+8]
        if len(byte_bits) < 8:
            # pad last byte with zeros on the right (least significant positions)
            byte_bits += ['0'] * (8 - len(byte_bits))
        byte_str = ''.join(byte_bits)
        b.append(int(byte_str, 2))
    with open(filename, 'wb') as f:
        f.write(b)

if __name__ == "__main__":
    N = 1_000_000
    # optional: set a custom seed: four 32-bit nonzero integers
    seed = [0x13579BDF, 0x2468ACE0, 0x0F1E2D3C, 0xA5A5A5A5]
    bits = generate_bits(N, seed=seed)
    print("Generated", len(bits), "bits.")
    save_bits_ascii(bits, "xorshift_bits.txt")
    save_bits_packed(bits, "xorshift_bits.bin")
    print("Saved xorshift_bits.txt and xorshift_bits.bin")
