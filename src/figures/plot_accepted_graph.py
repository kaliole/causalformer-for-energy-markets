"""Plot the final accepted causal graph for the DE-LU data.

Reads the validated edge table and draws only the edges that passed the
acceptance rule, coloured by tier: Tier 2 (strongly accepted) in dark green,
Tier 1 (accepted) in light green. Node positions match the candidate graph
(Figure candidate_graph) so the two figures can be compared directly: 17
candidate edges reduce to 5 accepted edges.

Output: figures/accepted_graph.pdf
"""

import pathlib

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import pandas as pd

from plot_candidate_graph import NODE_LABELS

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
EDGES_CSV = ROOT / "results" / "edges" / "delu_validated_edges.csv"
OUT_PDF = ROOT / "figures" / "accepted_graph.pdf"

TIER2_COLOR = "#1b7837"  # dark green: strongly accepted
TIER1_COLOR = "#7fbf7b"  # light green: accepted


def load_accepted_edges(csv_path: pathlib.Path) -> pd.DataFrame:
    """Load validated edges and keep only accepted ones (Tier 1 and Tier 2).

    Args:
        csv_path: Path to delu_validated_edges.csv.

    Returns:
        DataFrame with columns cause, effect, lag, strongly_accepted.
    """
    df = pd.read_csv(csv_path)
    acc = df[df["accepted"]].copy()
    return acc[["cause", "effect", "lag", "strongly_accepted"]].reset_index(drop=True)


def build_graph(edges: pd.DataFrame) -> nx.DiGraph:
    """Build a directed graph with lag and tier stored per edge.

    All seven variables are added as nodes (even if unconnected) so the layout
    matches the candidate graph.

    Args:
        edges: DataFrame with cause, effect, lag, strongly_accepted.

    Returns:
        Directed graph.
    """
    g = nx.DiGraph()
    g.add_nodes_from(NODE_LABELS.keys())
    for _, row in edges.iterrows():
        g.add_edge(
            row["cause"], row["effect"],
            lag=int(row["lag"]), tier2=bool(row["strongly_accepted"]),
        )
    return g


def plot_graph(g: nx.DiGraph, out_path: pathlib.Path) -> None:
    """Draw the accepted graph, coloured by tier, and save as a vector PDF.

    Args:
        g: Directed accepted graph.
        out_path: Destination PDF path.
    """
    pos = nx.circular_layout(g)
    connectionstyle = "arc3,rad=0.12"

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.set_axis_off()

    nx.draw_networkx_nodes(
        g, pos, ax=ax, node_size=4200, node_color="#eef2f7",
        edgecolors="#3a5a80", linewidths=1.5,
    )
    nx.draw_networkx_labels(
        g, pos, labels=NODE_LABELS, ax=ax, font_size=12, font_color="#1a1a1a",
    )

    edge_colors = [TIER2_COLOR if d["tier2"] else TIER1_COLOR
                   for _, _, d in g.edges(data=True)]
    edge_widths = [2.6 if d["tier2"] else 1.9 for _, _, d in g.edges(data=True)]
    nx.draw_networkx_edges(
        g, pos, ax=ax, node_size=4200, arrowsize=18, width=edge_widths,
        edge_color=edge_colors, connectionstyle=connectionstyle,
    )
    edge_labels = {(u, v): d["lag"] for u, v, d in g.edges(data=True)}
    nx.draw_networkx_edge_labels(
        g, pos, edge_labels=edge_labels, ax=ax, font_size=12,
        label_pos=0.5, connectionstyle=connectionstyle,
        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none"},
    )

    legend_handles = [
        Line2D([0], [0], color=TIER2_COLOR, lw=2.6, label="Tier 2 (strongly accepted)"),
        Line2D([0], [0], color=TIER1_COLOR, lw=1.9, label="Tier 1 (accepted)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=11)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate the accepted-graph figure."""
    edges = load_accepted_edges(EDGES_CSV)
    g = build_graph(edges)
    plot_graph(g, OUT_PDF)
    n2 = sum(1 for _, _, d in g.edges(data=True) if d["tier2"])
    print(f"Wrote {OUT_PDF} ({g.number_of_edges()} accepted edges, "
          f"{n2} Tier 2, {g.number_of_edges() - n2} Tier 1)")


if __name__ == "__main__":
    main()
