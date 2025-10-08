#!/usr/bin/env python3
"""Generate QRNG-like bitstrings using Qiskit simulator."""
import argparse
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from tqdm import tqdm


def generate_qiskit_bits(n_bits:int, n_qubits:int=8):
    # We'll generate n_bits by measuring circuits of n_qubits and concatenating outputs
    backend = Aer.get_backend('qasm_simulator')
    bits = []
    shots = min(1024, n_bits)
    qc = QuantumCircuit(n_qubits, n_qubits)

    for qb in tqdm(range(n_qubits)):
        qc.h(qb)
        
    qc.measure(range(n_qubits), range(n_qubits))

    while len(bits) < n_bits:
        job = execute(qc, backend, shots=shots)
        counts = job.result().get_counts()
    
    for k, v in tqdm(counts.items()):
    # k is bitstring like '0101'
        for _ in range(v):
            bits.extend(list(map(int, k[::-1]))) # reverse if needed
            if len(bits) >= n_bits:
                break
            if len(bits) >= n_bits:
                break
    return np.array(bits[:n_bits], dtype=np.uint8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_bits', type=int, default=1000)
    parser.add_argument('--n_qubits', type=int, default=8)
    parser.add_argument('--out', type=str, default='data/qrng_1k.npz')
    args = parser.parse_args()
    bits = generate_qiskit_bits(args.n_bits, args.n_qubits)
    np.savez_compressed(args.out, bits=bits)
    print(f"Saved {len(bits)} bits to {args.out}")