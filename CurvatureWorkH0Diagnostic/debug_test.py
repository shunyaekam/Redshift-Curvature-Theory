#!/usr/bin/env python3
"""
Debug script to isolate the hanging issue
"""
import sys
print("1. Starting debug script")

print("2. Importing basic modules...")
import numpy as np
import pandas as pd
import json
from pathlib import Path

print("3. Testing file access...")
config_path = "data/lens_config.json"
with open(config_path, 'r') as f:
    config = json.load(f)
print(f"✓ Config loaded with {len(config['h0licow_files'])} files")

print("4. Testing first data file...")
files = config['h0licow_files']
first_file = list(files.values())[0]
print(f"Loading {first_file}...")
data = pd.read_csv(first_file)
print(f"✓ Loaded {len(data)} rows")

print("5. Testing random number generation...")
samples = np.random.normal(70, 5, 10000)
print(f"✓ Generated {len(samples)} samples")

print("6. All basic operations work - issue must be in the class structure")