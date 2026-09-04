"""Plot the validation evidence matrix for the DE-LU candidate edges.

Reads the validated edge table and renders a colour-coded matrix: one row per
cross-variable candidate, one column per validation test, plus a final tier
column. Green marks supportive evidence (test passed), red marks failing or
unsupportive evidence, grey marks a test that does not apply. Rows are ordered
by tier so the accepted edges sit at the top, above a separating line.

Green semantics per column:
  PCMCI      -> edge significant at alpha = 0.05
  Granger    -> edge supported
  Time shift -> edge NOT recovered after the placebo (survived_timeshift = False)
  Phase rand.-> edge NOT recovered after the placebo (survived_phaserand = False)
  Bootstrap  -> stability score >= 0.6
  Season     -> regime-robust across winter and summer

Output: figures/evidence_heatmap.pdf
"""

import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
EDGES_CSV = ROOT / "results" / "edges" / "delu_validated_edges.csv"
OUT_PDF = ROOT / "figures" / "evidence_heatmap.pdf"

BOOTSTRAP_THRESHOLD = 0.6

PASS_COLOR = "#7fbf7b"   # green: supportive evidence
FAIL_COLOR = "#f4a582"   # red: failing / unsupportive
NA_COLOR = "#d9d9d9"     # grey: not applicable
TIER2_COLOR = "#1b7837"  # dark green
TIER1_COLOR = "#7fbf7b"  # light green
REJECT_COLOR = "#eeeeee"

DISPLAY = {
    "temperature": "Temperature", "load": "Load", "price": "Price",
    "wind_speed": "Wind speed", "wind_generation": "Wind generation",
    "solar_radiation": "Solar radiation", "solar_generation": "Solar generation",
}

COLUMNS = ["PCMCI", "Granger", "Time shift", "Phase rand.", "Bootstrap", "Season"]


def load_sorted(csv_path: pathlib.Path) -> pd.DataFrame:
    """Load validated edges and order them by tier, then stability.

    Args:
        csv_path: Path to delu_validated_edges.csv.

    Returns:
        DataFrame sorted so Tier 2 edges come first and rejected edges last.
    """
    df = pd.read_csv(csv_path)
    df["tier"] = df.apply(
        lambda r: 2 if r["strongly_accepted"] else (1 if r["accepted"] else 0), axis=1
    )
    return df.sort_values(["tier", "stability_score"], ascending=[False, False]).reset_index(drop=True)


def cell_state(row: pd.Series, column: str) -> str:
    """Return the evidence state for one edge and one test column.

    Args:
        row: A row of the validated edge table.
        column: One of COLUMNS.

    Returns:
        "pass", "fail", or "na".
    """
    if column == "PCMCI":
        if pd.isna(row["pcmci_pvalue"]):
            return "na"  # contemporaneous edge, not tested (tau_min = 1)
        return "pass" if row["pcmci_supported"] else "fail"
    if column == "Granger":
        return "pass" if row["granger_supported"] else "fail"
    if column == "Time shift":
        return "pass" if not row["survived_timeshift"] else "fail"
    if column == "Phase rand.":
        return "pass" if not row["survived_phaserand"] else "fail"
    if column == "Bootstrap":
        return "pass" if row["stability_score"] >= BOOTSTRAP_THRESHOLD else "fail"
    if column == "Season":
        return "pass" if row["regime_robust"] else "fail"
    raise ValueError(f"unknown column {column}")


def plot_heatmap(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    """Render the evidence matrix and save it as a vector PDF.

    Args:
        df: Sorted validated edge table.
        out_path: Destination PDF path.
    """
    state_style = {
        "pass": (PASS_COLOR, "✓"),   # check
        "fail": (FAIL_COLOR, "✗"),   # cross
        "na": (NA_COLOR, "–"),       # dash
    }
    n_rows = len(df)
    n_cols = len(COLUMNS) + 1  # + tier column

    fig, ax = plt.subplots(figsize=(9.5, 0.5 * n_rows + 1.5))

    for i, (_, row) in enumerate(df.iterrows()):
        y = n_rows - 1 - i  # top row first
        for j, column in enumerate(COLUMNS):
            color, glyph = state_style[cell_state(row, column)]
            ax.add_patch(Rectangle((j, y), 1, 1, facecolor=color, edgecolor="white", linewidth=1.5))
            ax.text(j + 0.5, y + 0.5, glyph, ha="center", va="center", fontsize=11, color="#333333")
        # tier column
        j = len(COLUMNS)
        if row["strongly_accepted"]:
            tcolor, tlabel, tfont = TIER2_COLOR, "T2", "white"
        elif row["accepted"]:
            tcolor, tlabel, tfont = TIER1_COLOR, "T1", "#1a1a1a"
        else:
            tcolor, tlabel, tfont = REJECT_COLOR, "–", "#666666"
        ax.add_patch(Rectangle((j, y), 1, 1, facecolor=tcolor, edgecolor="white", linewidth=1.5))
        ax.text(j + 0.5, y + 0.5, tlabel, ha="center", va="center", fontsize=9,
                fontweight="bold", color=tfont)

    # separating line under the accepted edges
    n_accepted = int((df["tier"] > 0).sum())
    y_sep = n_rows - n_accepted
    ax.plot([0, n_cols], [y_sep, y_sep], color="#333333", linewidth=1.8)

    # row labels
    ax.set_yticks([n_rows - 1 - i + 0.5 for i in range(n_rows)])
    ax.set_yticklabels(
        [f"{DISPLAY[r['cause']]} → {DISPLAY[r['effect']]}" for _, r in df.iterrows()],
        fontsize=9,
    )
    # column labels
    ax.set_xticks([j + 0.5 for j in range(n_cols)])
    ax.set_xticklabels(COLUMNS + ["Tier"], fontsize=9, rotation=30, ha="left",
                       rotation_mode="anchor")
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect("equal")

    legend_handles = [
        Patch(facecolor=PASS_COLOR, label="Passed / supportive"),
        Patch(facecolor=FAIL_COLOR, label="Failed / unsupportive"),
        Patch(facecolor=NA_COLOR, label="Not applicable"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=3, frameon=False, fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate the evidence-matrix figure."""
    df = load_sorted(EDGES_CSV)
    plot_heatmap(df, OUT_PDF)
    n_acc = int((df["tier"] > 0).sum())
    print(f"Wrote {OUT_PDF} ({len(df)} edges, {n_acc} accepted)")


if __name__ == "__main__":
    main()
