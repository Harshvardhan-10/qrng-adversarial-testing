import requests
import time
import numpy as np

def fetch_anu_qrng_bits(total_bits, output_file="anu_qrng_bits.txt", delay=1.0):
    """
    Fetch quantum random bits from ANU QRNG API.
    
    Args:
        total_bits: Total number of bits to fetch
        output_file: Output filename
        delay: Delay between API calls in seconds (be nice to their server!)
    
    API Details:
    - Endpoint: https://qrng.anu.edu.au/API/jsonI.php
    - Parameters: length (number of values), type (uint8 for bits), size (array size)
    - Rate limit: Be respectful, add delays between requests
    """
    
    base_url = "https://qrng.anu.edu.au/API/jsonI.php"
    bits_collected = []
    
    # API returns uint8 values (0-255), we need to convert to bits
    # Request 1024 uint8 values at a time = 1024*8 = 8192 bits per request
    values_per_request = 1024
    bits_per_request = values_per_request * 8
    
    total_requests = (total_bits + bits_per_request - 1) // bits_per_request
    
    print(f"Fetching {total_bits} quantum random bits from ANU QRNG...")
    print(f"Estimated requests needed: {total_requests}")
    print(f"Estimated time: ~{total_requests * delay / 60:.1f} minutes\n")
    
    for request_num in range(total_requests):
        try:
            # Prepare API request
            params = {
                'length': values_per_request,
                'type': 'uint8'
            }
            
            # Make request
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            
            if not data.get('success'):
                print(f"API returned error: {data}")
                continue
            
            # Extract uint8 values
            uint8_values = data['data']
            
            # Convert each uint8 to 8 bits
            for value in uint8_values:
                # Convert to binary string (8 bits, padded with zeros)
                binary_str = format(value, '08b')
                bits_collected.extend([int(b) for b in binary_str])
            
            # Progress update
            progress = len(bits_collected)
            print(f"Progress: {progress:,}/{total_bits:,} bits ({progress/total_bits*100:.1f}%) - Request {request_num+1}/{total_requests}")
            
            # Stop if we have enough bits
            if len(bits_collected) >= total_bits:
                break
            
            # Be nice to their server
            time.sleep(delay)
            
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)
            continue
        except KeyError as e:
            print(f"Unexpected API response format: {e}")
            continue
    
    # Trim to exact number of bits requested
    bits_collected = bits_collected[:total_bits]
    
    # Convert to numpy array
    bits_array = np.array(bits_collected, dtype=np.uint8)
    
    # Save to file
    bit_string = ''.join(map(str, bits_array))
    with open(output_file, 'w') as f:
        f.write(bit_string)
    
    print(f"\n✓ Successfully fetched {len(bits_collected):,} quantum random bits")
    print(f"✓ Saved to: {output_file}")
    
    # Statistics
    ones = np.sum(bits_array)
    zeros = len(bits_array) - ones
    print(f"\nStatistics:")
    print(f"  Zeros: {zeros:,} ({zeros/len(bits_array)*100:.2f}%)")
    print(f"  Ones:  {ones:,} ({ones/len(bits_array)*100:.2f}%)")
    print(f"  First 50 bits: {bit_string[:50]}")
    print(f"  Last 50 bits:  {bit_string[-50:]}")
    
    return bits_array

def fetch_anu_qrng_hex(total_bits, output_file="anu_qrng_bits.txt", delay=1.0):
    """
    Alternative method using hex16 type (might be faster).
    """
    base_url = "https://qrng.anu.edu.au/API/jsonI.php"
    bits_collected = []
    
    # Request hex values (each hex = 16 bits)
    values_per_request = 1024
    bits_per_request = values_per_request * 16
    
    total_requests = (total_bits + bits_per_request - 1) // bits_per_request
    
    print(f"Fetching {total_bits} quantum random bits from ANU QRNG (hex method)...")
    print(f"Estimated requests needed: {total_requests}")
    print(f"Estimated time: ~{total_requests * delay / 60:.1f} minutes\n")
    
    for request_num in range(total_requests):
        try:
            params = {
                'length': values_per_request,
                'type': 'hex16'
            }
            
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data.get('success'):
                print(f"API returned error: {data}")
                continue
            
            # Extract hex values and convert to bits
            for hex_str in data['data']:
                # Remove '0x' prefix if present
                hex_str = hex_str.replace('0x', '')
                # Convert hex to integer, then to 16-bit binary string
                value = int(hex_str, 16)
                binary_str = format(value, '016b')
                bits_collected.extend([int(b) for b in binary_str])
            
            progress = len(bits_collected)
            print(f"Progress: {progress:,}/{total_bits:,} bits ({progress/total_bits*100:.1f}%) - Request {request_num+1}/{total_requests}")
            
            if len(bits_collected) >= total_bits:
                break
            
            time.sleep(delay)
            
        except Exception as e:
            print(f"Error: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)
            continue
    
    bits_collected = bits_collected[:total_bits]
    bits_array = np.array(bits_collected, dtype=np.uint8)
    
    bit_string = ''.join(map(str, bits_array))
    with open(output_file, 'w') as f:
        f.write(bit_string)
    
    print(f"\n✓ Successfully fetched {len(bits_collected):,} quantum random bits")
    print(f"✓ Saved to: {output_file}")
    
    ones = np.sum(bits_array)
    zeros = len(bits_array) - ones
    print(f"\nStatistics:")
    print(f"  Zeros: {zeros:,} ({zeros/len(bits_array)*100:.2f}%)")
    print(f"  Ones:  {ones:,} ({ones/len(bits_array)*100:.2f}%)")
    
    return bits_array

if __name__ == "__main__":
    # Example 1: Fetch 100,000 bits (takes ~2-3 minutes)
    bits = fetch_anu_qrng_bits(
        total_bits=100000,
        output_file="anu_qrng_100k_bits.txt",
        delay=1.0  # 1 second between requests
    )
    
    # Example 2: Fetch 1 million bits (takes ~20-30 minutes)
    # bits = fetch_anu_qrng_bits(
    #     total_bits=1000000,
    #     output_file="anu_qrng_1M_bits.txt",
    #     delay=1.0
    # )
    
    # Example 3: Use hex method (potentially faster)
    # bits = fetch_anu_qrng_hex(
    #     total_bits=100000,
    #     output_file="anu_qrng_100k_bits.txt",
    #     delay=1.0
    # )