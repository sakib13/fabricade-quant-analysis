# Behavioural Analysis of Decision Latency in a State-Dependent Narrative Environment

An exploratory, reproducible **Python** analysis of behavioural interaction-log and
decision-latency data from a within-subjects experiment I designed and ran. It demonstrates a
full pipeline for messy behavioural logs: parsing raw event data, modelling repeated-measures
response times with a **linear mixed-effects model**, quantifying condition and learning
effects with appropriate paired non-parametric tests and effect sizes, and reasoning explicitly
about **statistical power** at a small sample size.

## Background

The data come from *Fabricade*, a text-based narrative environment (built in Unity) whose
responses depend on hidden, accumulating internal state. Ten participants each played two
versions ("conditions", A and B) in a counterbalanced within-subjects design. Every session was
automatically instrumented, logging each scene transition and each choice together with the
**hesitation time** (decision latency) that preceded it, plus the final values of six internal
state variables. The parent study's findings are qualitative; this repository is a standalone
quantitative analysis of the behavioural logs.

## What it demonstrates

- Building a tidy analysis dataset from raw, irregular JSON event logs (`parse_logs.py`).
- A **linear mixed-effects model** of log decision latency with a random intercept per
  participant — the correct structure for repeated measures nested within subjects.
- Paired, non-parametric hypothesis tests (Wilcoxon signed-rank) with rank-biserial effect
  sizes, appropriate for *N* = 10.
- Separation of a **condition** effect from a **learning / order** effect via the
  counterbalanced design.
- Behavioural-trajectory and latent-state-space visualisations (PCA via NumPy SVD).
- An explicit **power analysis** and a candid limitations section, including confounds
  (e.g. reading time embedded in latency) that qualify the substantive interpretation.

## Participant data and the linking key

The datasets in `data/` are anonymised. Participants appear only as the codes
`P1`..`P10`, which match the codes used in the parent thesis.

The key that maps raw log names to those codes is **not in this repository** and
will not be. It is held separately from the data under the study's data
management plan. `parse_logs.py` reads it from a path given by the
`FABRICADE_KEY` environment variable, and exits with an explanation if it is
absent.

This only affects the parse step, which regenerates `data/` from the raw logs.
The analysis notebook runs end to end from the anonymised CSVs that are already
in the repository, so the results here are fully reproducible without the key.

Participants gave informed consent for the study described in the parent thesis.
Please do not attempt to re-identify anyone from this data.

## Repository structure

```
fabricade-quant-analysis/
├── README.md
├── requirements.txt
├── parse_logs.py                          # raw logs -> tidy, anonymised CSVs
├── build_notebook.py                      # regenerates the notebook from source
├── data/
│   ├── sessions.csv                       # 20 rows: participant x condition
│   └── choices.csv                        # 260 rows: one per recorded choice
├── figures/                               # exported figures (PNG)
└── fabricade_behavioural_analysis.ipynb   # the analysis (run end-to-end)
```

## Reproducing

```bash
pip install -r requirements.txt

# (Optional) regenerate the tidy data from the raw logs. The raw log file is not
# included here because it contains participant-supplied names; the derived data
# in data/ is fully anonymised (participants coded P1..P10).
python parse_logs.py

# Regenerate and run the notebook:
python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace fabricade_behavioural_analysis.ipynb
# ...or simply open the .ipynb and Run All.
```

## Data and ethics

Data were collected under informed consent for research use, with participants told their data
would be anonymised. Accordingly, only **anonymised, derived** data is included here
(`data/*.csv`, coded `P1`–`P10`); the raw log file with participant-supplied names is withheld.
The anonymisation mapping is applied in `parse_logs.py`.

## Note on scope

This is an **exploratory** analysis of a small sample (*N* = 10). It is intended to demonstrate
method and rigour on real behavioural data, not to make confirmatory claims. See Part 6 of the
notebook for the power analysis and limitations.
