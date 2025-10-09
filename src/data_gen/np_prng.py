import numpy as np

def generate_prng_bits(n_bits, filename="prng_bits.txt", seed=None):
    """
    Generate random bits using Python's pseudorandom number generator.
    
    Args:
        n_bits: Number of bits to generate
        filename: Output file name
        seed: Random seed for reproducibility (optional)
    """
    # Set seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random bits (0 or 1)
    bits = np.random.randint(0, 2, size=n_bits, dtype=np.uint8)
    
    # Convert to string
    bit_string = ''.join(map(str, bits))
    
    # Save to file
    with open(filename, 'w') as f:
        f.write(bit_string)
    
    print(f"Generated {n_bits} PRNG bits")
    print(f"Saved to: {filename}")
    print(f"First 50 bits: {bit_string[:50]}...")
    print(f"Last 50 bits:  ...{bit_string[-50:]}")
    
    # Basic statistics
    ones = np.sum(bits)
    zeros = n_bits - ones
    print(f"\nStatistics:")
    print(f"  Zeros: {zeros} ({zeros/n_bits*100:.2f}%)")
    print(f"  Ones:  {ones} ({ones/n_bits*100:.2f}%)")
    
    return bits

if __name__ == "__main__":
    # Generate 1 Mln bits
    bits = generate_prng_bits(n_bits=64000000, filename="prng_64M_bits.txt")