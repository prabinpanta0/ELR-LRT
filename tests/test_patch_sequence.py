import numpy as np
from elr_lrt import patch_sequence

# Convert byte string to a list of integers (byte values)
byte_sequence = list(b"hello world this is a test")

# Call patch_sequence with the converted sequence and sample parameters
patches = patch_sequence(byte_sequence, k=2, theta=1.5, theta_r=0.5)

# Print each patch as a byte string for readability
for i, patch in enumerate(patches):
    print(f"Patch {i+1}: {bytes(patch)}")