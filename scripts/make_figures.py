"""Figures and LaTeX tables of the paper, from the analysis results. CPU only.

    python scripts/make_figures.py              # -> results/figures, results/tables
    python scripts/make_figures.py --out paper  # -> paper/figures, paper/tables

Figures: fig_rigor_by_config (Fig. 1), fig_convergence_ability (Fig. 2), fig_variance
(Fig. 3), fig_size_ladder (Fig. 4), fig_triad and fig_stability (appendix). Tables:
contrasts (Table 2), measures, paper_judges, size_ladder and prompts (appendix).
Model identity is encoded by color and marker shape.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / "results" / "confirmatory"
EXT = ROOT / "results" / "exploratory"
_ap = argparse.ArgumentParser(description="Paper figures and tables.")
_ap.add_argument("--out", type=Path, default=ROOT / "results", help="writes {out}/figures and {out}/tables")
OUTROOT = _ap.parse_args().out
OUT = OUTROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
cfg = yaml.safe_load(open(ROOT / "config.yaml"))

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODELS = [m["short"] for m in cfg["models"]]
MLABEL = {"qwen35-2b": "Qwen3.5-2B", "qwen35-4b": "Qwen3.5-4B", "gemma4-e2b": "Gemma-4-E2B",
          "gemma4-e4b": "Gemma-4-E4B"}
MCOLOR = dict(zip(MODELS, SLOTS))
MMARK = dict(zip(MODELS, ["o", "s", "^", "D"]))
PROMPTS = list(cfg["prompts"])
MEAS = ["R_raw", "R_c", "R_rasch"]
MNAME = {"R_raw": r"Raw flag rate $R_{\mathrm{raw}}$", "R_c": r"Criterion $R_c=-c$",
         "R_rasch": r"Rasch severity $R_{\mathrm{Rasch}}$ (logit)"}
MSHORT = {"R_raw": r"$R_{\mathrm{raw}}$", "R_c": r"$R_c$", "R_rasch": r"$R_{\mathrm{Rasch}}$"}
PAPER_MARK = {"gpt-4o-mini-2024-07-18": ("GPT-4o mini", "v"), "gpt-4o-2024-08-06": ("GPT-4o", "P"),
              "deepseek-chat": ("DeepSeek-V3", "X"), "sabia-3.1-2025-05-08": ("Sabiá-3.1", "*")}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.frameon": False, "lines.linewidth": 1.4,
    "pdf.fonttype": 42,
})
TEXTW = 5.5  # NeurIPS text width in inches


def save(fig, name):
    """Write vector PDF (for LaTeX) and a PNG preview."""
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def model_handles(extra=()):
    return [plt.Line2D([], [], color=MCOLOR[m], marker=MMARK[m], ms=4.5, lw=0, label=MLABEL[m])
            for m in MODELS] + list(extra)


T3 = pd.read_csv(AN / "measures_by_config.csv")
PJ = pd.read_csv(AN / "paper_judges.csv")


def errbars(ax, x_order, meas, dodge=0.13, ms=3.6):
    for k, m in enumerate(MODELS):
        g = T3[T3.model == m].set_index("prompt_id").loc[x_order]
        x = np.arange(len(x_order)) + (k - 1.5) * dodge
        ax.errorbar(x, g[meas], yerr=[g[meas] - g[f"{meas}_lo"], g[f"{meas}_hi"] - g[meas]],
                    color=MCOLOR[m], marker=MMARK[m], ms=ms, lw=0, elinewidth=1, capsize=0,
                    markeredgecolor="white", markeredgewidth=0.5)
    ax.set_xticks(range(len(x_order)), x_order)


# Rigor by configuration (Fig. 1) -------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 1.95))
for ax, meas in zip(axes, MEAS):
    errbars(ax, PROMPTS, meas, dodge=0.11, ms=3)
    for pm, (lab, mk) in PAPER_MARK.items():
        g = PJ[PJ.paper_model == pm].set_index("prompt_id").loc[["P1", "P2", "P3"]]
        x = np.array([PROMPTS.index(p) for p in g.index]) + 0.33
        ax.scatter(x, g[meas], marker=mk, s=12, color=INK2, zorder=3, linewidths=0.3)
    ax.axvspan(-0.5, 2.5, color="#f3f2ef", zorder=0, lw=0)
    ax.axvspan(5.5, 8.5, color="#f3f2ef", zorder=0, lw=0)
    ax.set_xlim(-0.5, 8.5)
    ax.set_title(MNAME[meas])
    ax.tick_params(axis="x", labelsize=6.5)
extra = [plt.Line2D([], [], color=INK2, marker=mk, lw=0, ms=4.5, label=f"{lab} (paper)")
         for lab, mk in PAPER_MARK.values()]
fig.legend(handles=model_handles(extra), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.2),
           columnspacing=1.0, handletextpad=0.2)
fig.tight_layout(w_pad=0.6)
save(fig, "fig_rigor_by_config")

# Variance decomposition (Fig. 3) ----------------------------------------------------------
dec = pd.read_csv(AN / "variance_decomposition.csv")
TERMS = ["family", "size", "prompt", "family:size", "family:prompt", "size:prompt", "residual"]
TLAB = {"family": "family", "size": "size", "prompt": "prompt", "family:size": "family×size",
        "family:prompt": "family×prompt", "size:prompt": "size×prompt", "residual": "residual (3-way)"}
fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 1.75), sharey=True)
for ax, (ver, title) in zip(axes, [("a_all9", "(a) all 9 prompts"), ("b_neutral4", "(b) neutral wordings only")]):
    left = np.zeros(3)
    for t, c in zip(TERMS, SLOTS):
        rows = [dec[(dec.version == ver) & (dec.measure == m) & (dec.term == t)].iloc[0] for m in MEAS]
        v = np.array([r.prop_ss for r in rows])
        ax.barh(range(3), v, left=left, color=c, height=0.62, edgecolor="white", linewidth=1, label=TLAB[t])
        for i, (val, l0) in enumerate(zip(v, left)):
            if val >= 0.12:
                light = c in ("#eda100", "#e87ba4", "#1baf7a")   # dark ink on light fills
                ax.text(l0 + val / 2, i, f"{val:.2f}", ha="center", va="center", fontsize=6,
                        color=INK if light else "white")
        left += v
    ax.set_yticks(range(3), [MSHORT[m] for m in MEAS])
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of sum of squares")
axes[0].invert_yaxis()   # shared y: invert once
axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), handlelength=1)
fig.tight_layout(w_pad=0.8)
save(fig, "fig_variance")

# Convergence (a) and ability (b, c) (Fig. 2) ------------------------------------------------
h3 = pd.read_csv(AN / "convergence.csv").set_index("pair")
h4 = pd.read_csv(AN / "ability.csv").set_index("quantity")
fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 1.8))
panels = [("R_raw", "R_c", "conv"), ("d_prime", "R_raw", "abil"), ("d_prime", "R_c", "abil")]
for ax, (xm, ym, kind) in zip(axes, panels):
    for m in MODELS:
        g = T3[T3.model == m]
        ax.scatter(g[xm], g[ym], color=MCOLOR[m], marker=MMARK[m], s=13, edgecolor="white", linewidth=0.4)
    if kind == "conv":
        r = h3.loc[f"{xm}~{ym}"]
        ax.set_title(rf"(a) $\rho$ = {r.spearman:.3f} [{r.lo:.3f}, {r.hi:.3f}]")
        ax.set_xlabel(MSHORT[xm])
    else:
        r = h4.loc[f"rho({ym}, d')"]
        tag = "(b)" if ym == "R_raw" else "(c)"
        ax.set_title(rf"{tag} $\rho$ = {r.estimate:.2f} [{r.lo:.2f}, {r.hi:.2f}]")
        ax.set_xlabel(r"$d'$")
    ax.set_ylabel(MSHORT[ym])
    ax.grid(axis="x")
fig.legend(handles=model_handles(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
fig.tight_layout(w_pad=0.6)
save(fig, "fig_convergence_ability")

# Exploratory size ladder (Fig. 4) -----------------------------------------------------
lad = pd.read_csv(EXT / "size_ladder_measures.csv")
FAM = {"qwen": ("Qwen3.5", SLOTS[0], "o"), "gemma": ("Gemma-4", SLOTS[1], "s")}
neutral = lad[lad.prompt_id.isin(["N", "N1", "N2", "N3"])].groupby(
    ["model", "family", "params_b"], as_index=False)[["R_c", "d_prime"]].mean()
p3 = lad[lad.prompt_id == "P3"]
fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 1.9))
for ax, m, title in [(axes[0], "d_prime", r"(a) ability $d'$"), (axes[1], "R_c", r"(b) rigor $R_c$")]:
    for fam, (lab, col, mk) in FAM.items():
        h = neutral[neutral.family == fam].sort_values("params_b")
        ax.plot(h.params_b, h[m], color=col, marker=mk, ms=4, label=f"{lab}, neutral (mean N–N3)",
                markeredgecolor="white", markeredgewidth=0.5)
        q = p3[p3.family == fam].sort_values("params_b")
        ax.plot(q.params_b, q[m], color=col, lw=1, ls="--", label=f"{lab}, P3 (paper prompt)")
        for _, r in q.iterrows():
            ax.scatter(r.params_b, r[m], marker=mk, s=12, zorder=3, linewidth=0.8, edgecolor=col,
                       facecolor="white" if r.model == "qwen35-0.8b" else col)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10], ["1", "2", "5", "10"])
    ax.set_xlabel("total params (B, log)")
    ax.set_title(title)
axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), handlelength=1.6)
fig.tight_layout(w_pad=0.8)
save(fig, "fig_size_ladder")

# Ranking stability and L/N/S triad (appendix) ------------------------------------------------
taus = pd.read_csv(AN / "stability_kendall_tau.csv")
fig, ax = plt.subplots(figsize=(2.6, 1.7))
vp = ax.violinplot([taus[taus.measure == m].tau.to_numpy() for m in MEAS], showmedians=True, showextrema=False)
for b in vp["bodies"]:
    b.set_facecolor(SLOTS[0])
    b.set_alpha(0.35)
vp["cmedians"].set_color(INK)
ax.set_xticks([1, 2, 3], [MSHORT[m] for m in MEAS])
ax.set_ylabel(r"Kendall $\tau$ (replicate vs. full)")
save(fig, "fig_stability")

fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 1.7))
for ax, meas in zip(axes, MEAS):
    errbars(ax, ["L", "N", "S"], meas, dodge=0.1)
    for k, m in enumerate(MODELS):
        g = T3[T3.model == m].set_index("prompt_id").loc[["L", "N", "S"]]
        ax.plot(np.arange(3) + (k - 1.5) * 0.1, g[meas], color=MCOLOR[m], lw=0.8, alpha=0.6)
    ax.set_title(MNAME[meas])
fig.legend(handles=model_handles(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
fig.tight_layout(w_pad=0.6)
save(fig, "fig_triad")
print(sorted(p.name for p in OUT.glob("*.pdf")))

# ---------------------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------------------
TAB = OUTROOT / "tables"
TAB.mkdir(parents=True, exist_ok=True)


def f(x, d=2):
    """Format a number for LaTeX with a true minus sign."""
    s = f"{x:.{d}f}"
    return s.replace("-", "$-$")


def ci(r, k, d=2):
    return f"{f(r[k], d)} [{f(r[k + '_lo'], d)}, {f(r[k + '_hi'], d)}]"


# Appendix: all 36 configurations
lines = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
         r"Model & Prompt & $H$ & $F$ & $d'$ & $R_{\mathrm{raw}}$ & $R_c$ [95\% CI] & $R_{\mathrm{Rasch}}$ [95\% CI] & miss. \\",
         r"\midrule"]
for k, m in enumerate(MODELS):
    for p in PROMPTS:
        r = T3[(T3.model == m) & (T3.prompt_id == p)].iloc[0]
        lines.append(f"{MLABEL[m] if p == PROMPTS[0] else ''} & {p} & {f(r.H)} & {f(r.F, 3)} & {f(r.d_prime)} & "
                     f"{f(r.R_raw, 3)} & {ci(r, 'R_c')} & {ci(r, 'R_rasch')} & {int(r.n_missing)} \\\\")
    if k < len(MODELS) - 1:
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "measures.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

# Main text: H1/H2 contrasts on R_c per model (+E3)
con = pd.read_csv(AN / "contrasts.csv")
lines = [r"\begin{tabular}{lccccc}", r"\toprule",
         r" & \multicolumn{2}{c}{H1 (manipulation)} & \multicolumn{2}{c}{H2 (paper prompts)} & E3 \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-6}",
         r"Model & S $-$ N & N $-$ L & P2 $-$ P1 & P3 $-$ P2 & P1 $-$ N \\", r"\midrule"]
for m in MODELS:
    cells = []
    for c in ["S-N", "N-L", "P2-P1", "P3-P2", "P1-N"]:
        r = con[(con.model == m) & (con.measure == "R_c") & (con.contrast == c)].iloc[0]
        s = f"{f(r.estimate)} [{f(r.lo)}, {f(r.hi)}]"
        cells.append(r"\textbf{" + s + "}" if r.ci_lt_0 else s)
    lines.append(f"{MLABEL[m]} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "contrasts.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

# Appendix: paper judges
lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r"Judge & Prompt & $H$ & $F$ & $d'$ & $R_{\mathrm{raw}}$ & $R_c$ & $R_{\mathrm{Rasch}}$ \\", r"\midrule"]
for pm, (lab, _) in PAPER_MARK.items():
    for p in ["P1", "P2", "P3"]:
        r = PJ[(PJ.paper_model == pm) & (PJ.prompt_id == p)].iloc[0]
        lines.append(f"{lab if p == 'P1' else ''} & {p} & {f(r.H)} & {f(r.F, 3)} & {f(r.d_prime)} & "
                     f"{f(r.R_raw, 3)} & {f(r.R_c)} & {f(r.R_rasch)} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "paper_judges.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

# Exploratory: size ladder
lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Model & params (B) & $R_c$ (N) & $R_c$ (P1) & $R_c$ (P3) & $d'$ (N) & missing \\", r"\midrule"]
names = {"qwen35-0.8b": "Qwen3.5-0.8B", "qwen35-2b": "Qwen3.5-2B", "qwen35-4b": "Qwen3.5-4B",
         "qwen35-9b": "Qwen3.5-9B", "gemma4-e2b": "Gemma-4-E2B", "gemma4-e4b": "Gemma-4-E4B",
         "gemma4-12b": "Gemma-4-12B"}
for m in names:
    g = lad[lad.model == m].set_index("prompt_id")
    miss = 100 * g.n_missing.sum() / (9 * 4238)
    lines.append(f"{names[m]} & {g.params_b.iat[0]:.2f} & {f(g.at['N', 'R_c'])} & {f(g.at['P1', 'R_c'])} & "
                 f"{f(g.at['P3', 'R_c'])} & {f(g.at['N', 'd_prime'])} & {miss:.1f}\\% \\\\")
    if m in ("qwen35-9b",):
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "size_ladder.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(sorted(p.name for p in TAB.glob("*.tex")))

# Appendix: verbatim system prompts
def tex_escape(t: str) -> str:
    """Escape LaTeX special characters in prompt text (kept verbatim otherwise)."""
    rep = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "_": r"\_", "#": r"\#", "%": r"\%",
           "&": r"\&", "$": r"\$", "^": r"\^{}", "~": r"\~{}", '"': r"\textquotedbl{}"}
    return "".join(rep.get(ch, ch) for ch in t)


PROMPT_ROLE = {"N": "neutral (base)", "L": "lenient = N + one sentence", "S": "strict = N + one sentence",
               "N1": "neutral paraphrase 1", "N2": "neutral paraphrase 2", "N3": "neutral paraphrase 3",
               "P1": "paper, Prompt 1 (Fig. 8)", "P2": "paper, Prompt 2 (Fig. 9)", "P3": "paper, Prompt 3 (Fig. 10)"}
lines = []
for pid in ["N", "L", "S", "N1", "N2", "N3", "P1", "P2", "P3"]:
    body = (ROOT / "prompts" / f"{pid}.txt").read_text(encoding="utf-8")
    lines.append(rf"\paragraph{{{pid} --- {PROMPT_ROLE[pid]}.}}")
    lines.append(r"{\footnotesize\raggedright " + r"\\ ".join(tex_escape(l) for l in body.splitlines()) + r"\par}")
(TAB / "prompts.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("prompts.tex written")
