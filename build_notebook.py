"""
build_notebook.py
Assemble the analysis notebook programmatically with nbformat, so the notebook
content is version-controllable and reproducible. Run:  python build_notebook.py
Then execute:  jupyter nbconvert --to notebook --execute --inplace <nb>
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s):   cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# ===========================================================================
md(r"""
# Behavioural Analysis of Decision Latency in a State-Dependent Narrative Environment

**An exploratory quantitative analysis of interaction-log and decision-latency data**

This notebook analyses behavioural logs from *Fabricade*, a within-subjects experiment I
designed and ran in which ten participants each experienced two versions ("conditions") of
a narrative environment whose responses depend on hidden, accumulating internal state. Each
gameplay session was automatically instrumented, producing per-choice **interaction logs**
and **decision-latency** (response-time) measurements.

The study's primary findings are qualitative and reported elsewhere. The purpose of *this*
notebook is narrower and methodological: to demonstrate a reproducible Python pipeline for
**behavioural interaction-log data** — parsing raw event logs, modelling repeated-measures
response times with a mixed-effects model, quantifying condition and learning effects, and
reasoning honestly about statistical power at this sample size.

> **Framing note.** With *N* = 10 participants this is an *exploratory* analysis. No claim of
> confirmatory inference is made. Where tests are reported they are accompanied by effect
> sizes and an explicit power analysis (Part 6). The value here is the method and the honest
> treatment of a small within-subjects dataset, not a significant *p*-value.

**Contents**
1. Data pipeline and design overview
2. Decision latency: a linear mixed-effects model
3. State-dependence: behaviour under two hidden-state conditions
4. Learning across repeated exposure (order effects)
5. Behavioural trajectories and the latent state space
6. Limitations and statistical power
""")

# --- setup -----------------------------------------------------------------
code(r"""
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.power import TTestPower

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
pd.set_option("display.precision", 3)

# Colourblind-safe (Okabe-Ito) condition palette, used consistently throughout.
CPAL = {"A": "#0072B2", "B": "#D55E00"}

sessions = pd.read_csv("data/sessions.csv")
choices  = pd.read_csv("data/choices.csv")

# Ordered categorical for scene, so plots and models respect narrative order.
scene_order = ["intake", "dinner", "bench", "corridor", "room", "discharge"]
for df in (choices,):
    df["scene"] = pd.Categorical(df["scene"], categories=scene_order, ordered=True)

# The 7 genuine narrative decisions per session (excludes 'Continue' navigation).
decisions = choices[choices.is_decision].copy()
decisions["log_hes"] = np.log(decisions["hesitation_s"])

print("sessions:", sessions.shape, "| choices:", choices.shape,
      "| decisions:", decisions.shape)
""")

# ===========================================================================
md(r"""
## Part 1 — Data pipeline and design overview

**Raw source.** Each gameplay session was exported as one CSV row containing metadata plus a
JSON array of time-stamped events (`scene_enter`, `choice_presented`, `choice_made`,
`story_end`). The number of leading CSV columns is inconsistent across rows, so
`parse_logs.py` locates the JSON payload by content and reads session start-time and duration
**from the events themselves** rather than from fragile positional columns. It emits two tidy,
**anonymised** tables (participants are coded `P1`–`P10`; no real identifiers reach the
derived data):

| file | grain | rows |
|---|---|---|
| `data/sessions.csv` | one participant × condition | 20 |
| `data/choices.csv`  | one recorded choice (long)  | 260 |

**A key cleaning step.** Of the 13 recorded choices per session, only **7 are genuine
narrative decisions**; the other 6 are `"Continue"` presses that merely advance text. Treating
those as decisions would badly contaminate any response-time analysis, so they are flagged
(`is_decision`) and excluded from the latency models below.

**Design.** Within-subjects, two conditions (A, B), counterbalanced order.
""")

code(r"""
# Design check: counterbalancing and grain.
subj = sessions.drop_duplicates("participant")
print("Participants:", subj.shape[0])
print("First-condition balance:", subj.first_condition.value_counts().to_dict())
print("Sessions per participant:", sessions.participant.value_counts().unique(),
      "(each plays both conditions)")
print("Decisions per session:", decisions.groupby(['participant','condition']).size().unique(),
      "(7 per session)")
sessions.head(4)
""")

code(r"""
# Session-duration overview by condition and by playthrough order.
dur = (sessions
       .groupby("condition")["duration_s"]
       .agg(["mean", "median", "std"]).round(1))
dur_order = (sessions
       .groupby("play_order")["duration_s"]
       .agg(["mean", "median", "std"]).round(1))
print("Duration (s) by condition:\n", dur, "\n")
print("Duration (s) by playthrough order:\n", dur_order)
""")

code(r"""
# Figure 1: session duration by condition (paired) and decision-latency distribution.
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

# (a) paired duration A vs B
wide = sessions.pivot(index="participant", columns="condition", values="duration_s")
for p, row in wide.iterrows():
    ax[0].plot(["A", "B"], [row["A"], row["B"]], color="0.7", lw=1, zorder=1)
ax[0].scatter(["A"]*len(wide), wide["A"], color=CPAL["A"], s=40, zorder=2, label="A")
ax[0].scatter(["B"]*len(wide), wide["B"], color=CPAL["B"], s=40, zorder=2, label="B")
ax[0].set_title("(a) Session duration is paired within participant")
ax[0].set_ylabel("Duration (s)"); ax[0].set_xlabel("Condition")

# (b) decision-latency distribution (raw), pooled
ax[1].hist(decisions["hesitation_s"], bins=24, color="#4C72B0", edgecolor="white")
ax[1].axvline(decisions["hesitation_s"].median(), color="k", ls="--", lw=1,
              label=f"median = {decisions['hesitation_s'].median():.1f}s")
ax[1].set_title("(b) Decision latency is right-skewed")
ax[1].set_xlabel("Hesitation (s)"); ax[1].set_ylabel("Count"); ax[1].legend()
plt.tight_layout(); plt.savefig("figures/fig1_overview.png", bbox_inches="tight"); plt.show()

print("Skewness of raw hesitation: %.2f | of log(hesitation): %.2f"
      % (stats.skew(decisions.hesitation_s), stats.skew(decisions.log_hes)))
""")

# ===========================================================================
md(r"""
## Part 2 — Decision latency: a linear mixed-effects model

Decision latencies are repeated measurements nested within participants, and they are strongly
right-skewed (Part 1b), so I model **log** hesitation with a **linear mixed-effects model**
with a random intercept per participant:

$$\log(\text{hesitation}_{ij}) = \beta_0 + \beta_1\,\text{condition}_{ij} + \boldsymbol{\beta}_s\,\text{scene}_{ij} + \beta_o\,\text{order}_{ij} + u_i + \varepsilon_{ij}, \qquad u_i \sim \mathcal{N}(0,\tau^2)$$

The random intercept $u_i$ accounts for the non-independence of the seven decisions made by the
same participant — the correct structure for this design, and the reason a plain OLS or a
naïve *t*-test would be inappropriate. Because the outcome is logged, exponentiated
coefficients read as **multiplicative** effects on latency.
""")

code(r"""
# Fit: fixed effects for condition, scene, playthrough order; random intercept by participant.
model = smf.mixedlm(
    "log_hes ~ C(condition) + C(scene) + C(play_order)",
    data=decisions,
    groups=decisions["participant"],
)
fit = model.fit(method="lbfgs", reml=True)
print(fit.summary())
""")

code(r"""
# Tidy fixed-effects table with exponentiated (multiplicative) effects and 95% CIs.
ci = fit.conf_int()
fe = pd.DataFrame({
    "coef": fit.fe_params,
    "p": fit.pvalues.reindex(fit.fe_params.index),
})
fe["ci_low"]  = ci.reindex(fe.index)[0]
fe["ci_high"] = ci.reindex(fe.index)[1]
fe["mult_effect"] = np.exp(fe["coef"])          # x-fold change in latency
fe["mult_ci_low"] = np.exp(fe["ci_low"])
fe["mult_ci_high"] = np.exp(fe["ci_high"])
fe = fe.round(3)

# Intraclass correlation: share of variance between participants.
tau2 = float(fit.cov_re.iloc[0, 0])
icc = tau2 / (tau2 + fit.scale)
print(f"Between-participant variance (tau^2) = {tau2:.3f}")
print(f"Residual variance (sigma^2)          = {fit.scale:.3f}")
print(f"Intraclass correlation (ICC)         = {icc:.2f}"
      f"  -> {icc*100:.0f}% of latency variance is between participants\n")
fe[["coef", "ci_low", "ci_high", "p", "mult_effect", "mult_ci_low", "mult_ci_high"]]
""")

md(r"""
**Reading the model.** The `C(scene)` terms show that latency is not flat across the narrative:
the coefficient on the **room** scene — where the environment discloses that the experienced
memories were fabricated — is the largest ($e^{1.17}\approx 3.2\times$ the intake baseline),
i.e. participants deliberated markedly longer there, consistent regardless of condition. The
`C(play_order)[T.2]` term ($e^{-0.20}\approx 0.82$) captures an ~18% speed-up on the second
playthrough (Part 4). The condition term ($e^{-0.27}\approx 0.76$) indicates ~24% shorter
latencies in Condition B — **but this must be interpreted cautiously**: Condition B presents
less on-screen text (ELARA's commentary is reduced), and session durations are correspondingly
shorter (Part 3), so much of this difference likely reflects reduced *reading* load rather than
faster *deliberation*. The measured latency blends the two (see Part 6). The high ICC (0.47)
confirms that pooling decisions without a per-participant term would violate independence.
""")

code(r"""
# Figure 2: model-relevant structure -- mean decision latency by scene and condition.
prof = (decisions.groupby(["scene", "condition"], observed=True)["hesitation_s"]
        .mean().reset_index())
fig, ax = plt.subplots(figsize=(8, 4.3))
for cond in ["A", "B"]:
    d = prof[prof.condition == cond]
    ax.plot(d["scene"], d["hesitation_s"], marker="o", color=CPAL[cond], label=f"Condition {cond}")
ax.set_title("Mean decision latency by scene — the 'room' revelation stands out")
ax.set_ylabel("Mean hesitation (s)"); ax.set_xlabel("Scene (narrative order)")
ax.legend(); plt.tight_layout()
plt.savefig("figures/fig2_latency_by_scene.png", bbox_inches="tight"); plt.show()
""")

# ===========================================================================
md(r"""
## Part 3 — State-dependence: behaviour under two hidden-state conditions

The two conditions differ in *how* the environment's hidden state is expressed back to the
player. Here I test whether that manipulation changes behaviour, using **paired, non-parametric**
tests (Wilcoxon signed-rank on the 10 within-participant pairs) with a rank-biserial effect
size. Non-parametric tests are appropriate given *N* = 10 and non-normal measures.
""")

code(r"""
def paired_wilcoxon(wide, label):
    a, b = wide["A"].values, wide["B"].values
    d = a - b
    dnz = d[d != 0]
    res = stats.wilcoxon(a, b)
    # rank-biserial correlation (matched pairs)
    r = stats.rankdata(np.abs(dnz))
    rbc = (r[dnz > 0].sum() - r[dnz < 0].sum()) / r.sum()
    return {
        "measure": label,
        "median_A": np.median(a), "median_B": np.median(b),
        "median_diff(A-B)": np.median(d),
        "W": res.statistic, "p": res.pvalue, "rank_biserial_r": rbc, "n_pairs": len(d),
    }

sess_lat = (decisions.groupby(["participant", "condition"])["hesitation_s"]
            .mean().reset_index()
            .pivot(index="participant", columns="condition", values="hesitation_s"))
sess_dur = sessions.pivot(index="participant", columns="condition", values="duration_s")

rows = [
    paired_wilcoxon(sess_lat, "Mean decision latency (s)"),
    paired_wilcoxon(sess_dur, "Session duration (s)"),
]
state_tbl = pd.DataFrame(rows).set_index("measure").round(3)
state_tbl
""")

code(r"""
# Final Ink state variables: are the accumulated hidden states different across conditions?
ink_vars = ["openness", "deflection", "resistance",
            "emotional_posture", "trust_in_system", "mystery_awareness"]
rows = []
for v in ink_vars:
    w = sessions.pivot(index="participant", columns="condition", values=v)
    try:
        res = stats.wilcoxon(w["A"], w["B"])
        p = res.pvalue
    except ValueError:      # all-zero differences
        p = np.nan
    rows.append({"ink_variable": v, "median_A": w["A"].median(),
                 "median_B": w["B"].median(), "p": p})
ink_tbl = pd.DataFrame(rows).set_index("ink_variable").round(3)
ink_tbl
""")

code(r"""
# Behavioural divergence: of the 7 decision points, how many did each participant
# resolve differently across the two conditions? (compared by option slot chosen)
piv = (decisions.pivot_table(index=["participant", "decision_number"],
                             columns="condition", values="option_index", aggfunc="first"))
piv["changed"] = (piv["A"] != piv["B"]).astype(int)
change_rate = piv.groupby("participant")["changed"].sum().sort_values(ascending=False)
change_rate.name = "decisions_changed_(of 7)"

fig, ax = plt.subplots(figsize=(8, 3.6))
order = sorted(change_rate.index, key=lambda x: int(x[1:]))
ax.bar([str(i) for i in order], change_rate.reindex(order).values, color="#4C72B0")
ax.set_ylabel("Decisions changed\n(of 7)"); ax.set_xlabel("Participant")
ax.set_title("Behavioural divergence between conditions varies widely across participants")
plt.tight_layout(); plt.savefig("figures/fig3_change_rate.png", bbox_inches="tight"); plt.show()
change_rate.to_frame()
""")

# ===========================================================================
md(r"""
## Part 4 — Learning across repeated exposure (order effects)

Every participant played twice. Because condition order is counterbalanced, first-vs-second
differences isolate a **learning / familiarity** effect rather than a condition effect. This
maps onto the project's interest in how representations are learned across repeated exposure.
""")

code(r"""
# Paired first- vs second-playthrough comparisons (within participant).
dur_ord = sessions.pivot(index="participant", columns="play_order", values="duration_s")
dur_ord.columns = ["first", "second"]

lat_ord = (decisions.groupby(["participant", "play_order"])["hesitation_s"].mean()
           .reset_index().pivot(index="participant", columns="play_order", values="hesitation_s"))
lat_ord.columns = ["first", "second"]

def paired_report(wide, label):
    a, b = wide["first"].values, wide["second"].values
    res = stats.wilcoxon(a, b)
    dz = np.mean(a - b) / np.std(a - b, ddof=1)
    return {"measure": label, "mean_first": a.mean(), "mean_second": b.mean(),
            "mean_change": np.mean(a - b), "W": res.statistic, "p": res.pvalue,
            "cohen_dz": dz}

order_tbl = pd.DataFrame([
    paired_report(dur_ord, "Session duration (s)"),
    paired_report(lat_ord, "Mean decision latency (s)"),
]).set_index("measure").round(3)
order_tbl
""")

code(r"""
# Figure 4: slopegraph -- duration drop from first to second playthrough.
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for p, row in dur_ord.iterrows():
    ax.plot(["first", "second"], [row["first"], row["second"]],
            marker="o", color="0.5", lw=1.2)
ax.plot(["first", "second"], [dur_ord["first"].mean(), dur_ord["second"].mean()],
        marker="o", color="#C44E52", lw=3, label="mean")
ax.set_ylabel("Session duration (s)"); ax.set_xlabel("Playthrough")
ax.set_title("Participants completed the second playthrough faster")
ax.legend(); plt.tight_layout()
plt.savefig("figures/fig4_order_effect.png", bbox_inches="tight"); plt.show()
""")

# ===========================================================================
md(r"""
## Part 5 — Behavioural trajectories and the latent state space

Two views of the within-session process. First, the **latency trajectory** across the six
scenes (a behavioural time-series through the narrative). Second, the **latent state space**:
each session ends in a point defined by six accumulated hidden-state variables; I standardise
these and project them to two dimensions with PCA (implemented directly with NumPy's SVD) to
see whether the two conditions occupy distinguishable regions of state space.
""")

code(r"""
# Figure 5a: per-participant latency trajectory across scenes, coloured by condition.
fig, ax = plt.subplots(figsize=(9, 4.5))
for (p, cond), g in decisions.groupby(["participant", "condition"], observed=True):
    g = g.sort_values("scene_pos")
    prof = g.groupby("scene_pos")["hesitation_s"].mean()
    ax.plot(prof.index, prof.values, color=CPAL[cond], alpha=0.35, lw=1)
# overlay condition means
for cond in ["A", "B"]:
    m = (decisions[decisions.condition == cond]
         .groupby("scene_pos")["hesitation_s"].mean())
    ax.plot(m.index, m.values, color=CPAL[cond], lw=3, marker="o", label=f"Condition {cond} (mean)")
ax.set_xticks(range(1, 7)); ax.set_xticklabels(scene_order, rotation=20)
ax.set_ylabel("Hesitation (s)"); ax.set_xlabel("Scene (narrative order)")
ax.set_title("Decision-latency trajectories through the narrative")
ax.legend(); plt.tight_layout()
plt.savefig("figures/fig5a_trajectories.png", bbox_inches="tight"); plt.show()
""")

code(r"""
# Figure 5b: PCA of the final latent-state vectors (NumPy SVD, no sklearn dependency).
ink_vars = ["openness", "deflection", "resistance",
            "emotional_posture", "trust_in_system", "mystery_awareness"]
X = sessions[ink_vars].to_numpy(dtype=float)
Xz = (X - X.mean(0)) / X.std(0, ddof=0)              # standardise
U, S, Vt = np.linalg.svd(Xz, full_matrices=False)    # PCA via SVD
scores = U[:, :2] * S[:2]
expl = (S**2 / (S**2).sum())[:2]

fig, ax = plt.subplots(figsize=(6.8, 5.2))
for cond in ["A", "B"]:
    m = (sessions["condition"] == cond).to_numpy()
    ax.scatter(scores[m, 0], scores[m, 1], color=CPAL[cond], s=55,
               edgecolor="white", label=f"Condition {cond}")
for i, code_ in enumerate(sessions["participant"]):
    ax.annotate(code_, (scores[i, 0], scores[i, 1]), fontsize=7, alpha=0.6,
                xytext=(3, 3), textcoords="offset points")
ax.set_xlabel(f"PC1 ({expl[0]*100:.0f}% var)")
ax.set_ylabel(f"PC2 ({expl[1]*100:.0f}% var)")
ax.set_title("Final latent-state vectors in 2-D state space")
ax.legend(); plt.tight_layout()
plt.savefig("figures/fig5b_state_space.png", bbox_inches="tight"); plt.show()

print("PC loadings (how each hidden-state variable weights on PC1, PC2):")
pd.DataFrame(Vt[:2].T, index=ink_vars, columns=["PC1", "PC2"]).round(2)
""")

# ===========================================================================
md(r"""
## Part 6 — Limitations and statistical power

Honest appraisal of what this dataset can and cannot support.

**Statistical power.** With ten within-participant pairs, only large effects are detectable. To
make this concrete, I take the observed paired effects and compute (i) achieved power at
*N* = 10 and (ii) the *N* required for 80% power, using a paired-*t* approximation.
""")

code(r"""
tp = TTestPower()

def power_row(wide, cols, label):
    a, b = wide[cols[0]].values, wide[cols[1]].values
    dz = np.mean(a - b) / np.std(a - b, ddof=1)
    achieved = tp.power(effect_size=abs(dz), nobs=10, alpha=0.05, alternative="two-sided")
    try:
        n80 = tp.solve_power(effect_size=abs(dz), power=0.8, alpha=0.05, alternative="two-sided")
    except Exception:
        n80 = np.nan
    return {"comparison": label, "cohen_dz": dz, "power_at_N=10": achieved,
            "N_for_80%_power": np.ceil(n80) if n80 == n80 else np.nan}

pwr = pd.DataFrame([
    power_row(dur_ord, ["first", "second"], "Order effect: duration (1st vs 2nd)"),
    power_row(sess_lat, ["A", "B"], "Condition effect: mean decision latency"),
]).set_index("comparison").round(3)
pwr
""")

md(r"""
**Interpretation.** The two rows deliberately contrast a large effect with a subtle one. The
**condition** effect on decision latency is large (Cohen's $d_z \approx 1.3$) and adequately
powered even at *N* = 10, so that contrast is on relatively firm statistical ground — though,
as Part 2 cautioned, its *substantive* interpretation is confounded by reading load. The
**order** effect on total session *duration* is small ($d_z \approx 0.4$) and clearly
*under*-powered here: roughly 50 participants would be needed to detect it at 80% power, so its
non-significance is a limitation of sample size, not evidence of no effect. Notably, the more
sensitive per-decision latency measure *does* reveal a robust order effect in the Part 2 mixed
model — a reminder that measurement choice, not just sample size, governs power. Reporting the
required *N* is the responsible way to handle an underpowered comparison rather than claiming a
null.

**Other limitations.**

- **Sample size (N = 10).** Exploratory only; effect estimates have wide intervals. No
  correction for multiple comparisons is applied because no confirmatory claims are made — the
  tests are descriptive aids, and *p*-values should be read as such.
- **Latency includes reading time.** `hesitation_s` is the interval from options appearing to
  selection, so it blends reading and deciding; it is a proxy for deliberation, not a pure
  decision time. A future design could log per-option dwell time (or eye-tracking) to separate
  them.
- **Behavioural-change measure.** Divergence is scored by the option slot chosen (Part 3);
  because Condition A also alters option wording, slot identity is a conservative proxy.
- **Latent state observed only at session end.** The six hidden-state variables are logged at
  `story_end`, so true within-session trajectories of the state vector cannot be reconstructed —
  only the behavioural (latency) trajectory can.

**Natural extensions.** Triangulate these behavioural measures against the per-condition
self-report questionnaire collected in the same study; add eye-tracking to decompose latency;
and replicate at a sample size informed by the power table above.
""")

code(r"""
# Reproducibility footer.
import sys, numpy, scipy, statsmodels, matplotlib, seaborn
print("Python     ", sys.version.split()[0])
for m in (numpy, scipy, statsmodels, pd, matplotlib, seaborn):
    print(f"{m.__name__:12s}", m.__version__)
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
with open("fabricade_behavioural_analysis.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote fabricade_behavioural_analysis.ipynb with", len(cells), "cells")
