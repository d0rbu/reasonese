"""Test-session setup.

The Bradley-Terry fit solves one small dense system per connected component,
and a component is a single (pair, assistant) block of a few hundred cells. On
a many-core machine OpenBLAS spends far longer synchronizing threads for a
matrix that size than it does on the arithmetic: a 180x180 `numpy.linalg.solve`
measured 2089 ms with the default thread count and 1.99 ms pinned to one
thread. These variables must be set before NumPy is first imported, which is
why they live at the top of `conftest.py` rather than in a fixture.
"""

from __future__ import annotations

import os

for _variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")
