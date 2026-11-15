# Mid-point report sources

The report is compiled on Overleaf with the course template. This folder holds what is needed to rebuild its figures and body.

| File | Role |
|---|---|
| `make_figures.py` | Builds Figures 2-4 into `figures/` (`exp1_success.pdf`, `exp1_surrogate.pdf`, `sim2sim.pdf`). Run `python report/midterm/make_figures.py`; needs numpy and matplotlib. |
| `data/lambda_sweep.json` | Figure 3(a): mean predicted and simulator-recomputed `J` on validation for each `lambda`, from the container run. |
| `data/calib.npz` | Figure 3(b): predicted vs. true `J` for the 2400 test branches (`pred`, `true`, candidate type `typ`), from the container run. |
| `method_diagram.tex` | Figure 1 (method overview), a standalone TikZ file: `pdflatex method_diagram.tex`. |
| `body.tex` | Report body, from the macro definitions to the appendices. On Overleaf it replaces everything after the template preamble in `main.tex`. |
| `main.bib` | The bibliography entries the body adds to the template's `main.bib`. |
| `figures/` | The four compiled figures. |

Figure names on Overleaf carry a `fig_` prefix: `figures/method_overview.pdf` is uploaded as `fig_method_overview.pdf`, `figures/exp1_success.pdf` as `fig_exp1_success.pdf`, and so on.

Data provenance:

- Figure 2 (closed-loop success counts) uses the GPU-workstation run with the official demonstrations, not the container run in `runs/`. The counts are written into `make_figures.py`.
- Figure 4 (sim-to-sim pilot) uses the Phase A scan and Phase C paired differences reported on the GPU workstation (5 seeds x 150 initial states); the values are also written into `make_figures.py`.
