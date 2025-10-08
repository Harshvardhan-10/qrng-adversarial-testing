#!/usr/bin/env python3
"""Orchestrate a small experiment: generate QRNG + PRNG data, run ML predictor and fourier test."""
import subprocess, os
os.makedirs('data', exist_ok=True)
subprocess.run(['python', 'src/data_gen/qiskit_qrng.py', '--n_bits', '2000', '--out', 'data/qrng_2k.npz'])
subprocess.run(['python', 'src/adversaries/ml_predictor.py', '--data', 'data/qrng_2k.npz', '--epochs', '5'])
subprocess.run(['python', 'src/adversaries/fourier_distinguisher.py', '--data', 'data/qrng_2k.npz'])
