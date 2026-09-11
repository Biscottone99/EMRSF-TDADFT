# Release validation v0.5.1

The release is accepted only after all of the following checks pass:

- complete source test suite (`46 passed`);
- clean wheel installation followed by the same test suite (`46 passed`);
- forced robust-SCF fallback through all four stages, terminating in a
  converged Newton/CIAH restricted open-shell reference;
- explicit period-two density detection and averaged-guess test;
- standard strategy failure without hidden fallback;
- byte comparison showing that MRSF/EMRSF response source files other than
  reference/log/version plumbing are unchanged from v0.5.0;
- Python syntax compilation and Slurm `bash -n` validation;
- inspection of wheel and source-distribution contents.

The large 39-atom user molecule is an HPC validation case and is not claimed as
locally reproduced by the release test suite. Its log must show which rescue
stage converged before any MRSF/EMRSF energies are interpreted.
