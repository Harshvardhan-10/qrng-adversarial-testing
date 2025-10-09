
import numpy as np

def read_bits_from_file(file_path):
    """
    Read a string of bits from a text file and convert to numpy array
    
    Args:
        file_path (str): Path to the text file containing bits
        
    Returns:
        np.ndarray: Array of individual bits as integers (0 or 1)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read the entire content and remove whitespace/newlines
            bit_string = file.read().strip().replace('\n', '').replace(' ', '')
        
        # Convert each character to integer and store in numpy array
        bits_array = np.array([int(bit) for bit in bit_string if bit in '01'], dtype=np.int8)
        
        print(f"Successfully read {len(bits_array)} bits from {file_path}")
        return bits_array
        
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def save_bits_as_npz(bits_array, output_path):
    """
    Save the bits array as NPZ format
    
    Args:
        bits_array (np.ndarray): Array of bits
        output_path (str): Output file path (without .npz extension)
    """
    try:
        # Add .npz extension if not present
        if not output_path.endswith('.npz'):
            output_path += '.npz'
            
        # Save the array
        np.savez_compressed(output_path, bits=bits_array)
        print(f"Bits array saved to {output_path}")
        print(f"Array shape: {bits_array.shape}")
        print(f"Array dtype: {bits_array.dtype}")
        
    except Exception as e:
        print(f"Error saving file: {e}")

# Main execution
if __name__ == "__main__":
    # File paths
    input_file = "xorshift_bits.txt"  # Input text file with bits
    output_file = "data/xorshift_1M_bits_array"  # Output NPZ file (extension will be added automatically)

    # Read bits from file
    bits_array = read_bits_from_file(input_file)
    
    if bits_array is not None:
        # Display some statistics
        print(f"\nBits statistics:")
        print(f"Total bits: {len(bits_array)}")
        print(f"Number of 0s: {np.sum(bits_array == 0)}")
        print(f"Number of 1s: {np.sum(bits_array == 1)}")
        print(f"First 20 bits: {bits_array[:20]}")
        
        # Save as NPZ
        save_bits_as_npz(bits_array, output_file)
        
        # Verify by loading the saved file
        try:
            loaded_data = np.load(output_file + '.npz')
            loaded_bits = loaded_data['bits']
            print(f"\nVerification: Successfully loaded {len(loaded_bits)} bits from NPZ file")
            print(f"Arrays are equal: {np.array_equal(bits_array, loaded_bits)}")
        except Exception as e:
            print(f"Error verifying saved file: {e}")
    else:
        print("Failed to read bits from file.")
