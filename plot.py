import struct
import numpy as np

import matplotlib.pyplot as plt

# Read the binary file
filename = 'mandel.dat'

with open(filename, 'rb') as f:
    # Read NX and NY (4-byte integers)
    nx = struct.unpack('i', f.read(4))[0]
    ny = struct.unpack('i', f.read(4))[0]
    
    # Read the array (NX x NY doubles in column-major order)
    data = np.fromfile(f, dtype=np.float64, count=nx*ny)
    
    # Reshape to (NX, NY) and transpose for column-major (Fortran) order
    array = data.reshape((nx, ny), order='F')

array[array>100] = 0.000001
array = np.log(array)

# Create pseudocolor plot
plt.figure(figsize=(10, 8))
plt.pcolormesh(array, shading='auto', cmap='viridis')
plt.colorbar(label='Value')
plt.xlabel('Y')
plt.ylabel('X')
plt.title('Mandelbrot Set')
plt.tight_layout()
plt.show()
