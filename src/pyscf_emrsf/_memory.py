"""Small PySCF compatibility shim for restricted container environments.

Normal Linux/HPC installations never need this.  Some isolated test runners do
not expose ``/proc/<pid>/statm``; PySCF queries that file when sizing integral
buffers.  The helper keeps production behaviour unchanged and only supplies a
zero-memory answer when /proc is genuinely unavailable.
"""

from __future__ import annotations


def patch_pyscf_memory_probe() -> None:
    from pyscf import lib

    try:
        lib.current_memory()
    except FileNotFoundError:
        lib.current_memory = lambda: (0.0, 0.0)

