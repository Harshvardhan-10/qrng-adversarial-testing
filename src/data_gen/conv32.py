def bits_to_decimals(input_file, output_file):
    with open(input_file, 'r') as f:
        bit_string = f.read().replace('\n', '').replace(' ', '')

    decimals = []
    for i in range(0, len(bit_string), 32):
        chunk = bit_string[i:i+32]
        if len(chunk) == 32:
            decimals.append(str(int(chunk, 2)))

    with open(output_file, 'w') as f:
        f.write('\n'.join(decimals))

# Example usage:
# bits_to_decimals('input.txt', 'output.txt')
if __name__ == "__main__":
    bits_to_decimals('data/prng_64M_bits.txt', 'data/prng_64M_decimals.txt')