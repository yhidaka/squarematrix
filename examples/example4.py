import os

num_threads_to_use = 2
if num_threads_to_use is not None:
    # If this variable is `None`, the script will use up all available threads.
    os.environ["OMP_NUM_THREADS"] = str(num_threads_to_use)
    # Another environment variable "OPENBLAS_NUM_THREADS" works similarly.

import matplotlib.pyplot as plt

from squarematrix import scan_n_theta, xyspectrum

ar = scan_n_theta()
xyspectrum(ar[51])

plt.show()