#!/usr/bin/env python3
"""Compute Walsh-Hadamard / Fourier coefficients on k-bit windows and report top coefficients."""
import numpy as np
from scipy.fftpack import fft

def windowed_walsh(bits, window=8):
    n = len(bits) - window + 1
    ints = np.array([int(''.join(map(str, bits[i:i+window])), 2) for i in range(n)])
    freq = np.bincount(ints, minlength=2**window)
    freq = freq / freq.sum()
    vec = 2*freq - 1
    coeffs = np.real(np.fft.fft(vec))
    return coeffs

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/qrng_1k.npz')
    parser.add_argument('--window', type=int, default=8)
    args = parser.parse_args()
    bits = np.load(args.data)['bits']
    coeffs = windowed_walsh(bits, args.window)
    topk = np.argsort(np.abs(coeffs))[-10:][::-1]
    print('Top coeff idxs and values:')
    for i in topk:
        print(i, coeffs[i])
