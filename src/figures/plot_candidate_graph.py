"""Plot the raw CausalFormer candidate graph for the DE-LU data.

Reads the raw WP4 edge list, keeps only cross-variable candidates (self-loops
omitted), and draws a directed graph with lag labels on the edges. The figure
shows the model's initial output before validation.

Output: figures/candidate_graph.pdf
"""

import pathlib

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

SEED = 42

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
EDGES_CSV = ROOT / "results" / "edges" / "delu_edges_raw.csv"
OUT_PDF = ROOT / "figures" / "candidate_graph.pdf"

# Display labels (line breaks keep long names from overlapping on the circle).
NODE_LABELS = {
    "temperature": "Temperature",
    "load": "Load",
    "price": "Price",
    "wind_speed": "Wind\nspeed",
    "wind_generation": "Wind\ngeneration",
    "solar_radiation": "Solar\nradiation",
    "solar_generation": "Solar\ngeneration",
}


def load_cross_variable_edges(csv_path: pathlib.Path) -> pd.DataFrame:
    """Load raw edges and drop self-loops.

    Args:
        csv_path: Path to delu_edges_raw.csv (cause, effect, lag, ...).

    Returns:
        DataFrame of cross-variable edges with columns cause, effect, lag.
    """
    df = pd.read_csv(csv_path)
    cross = df[df["cause"] != df["effect"]].copy()
    return cross[["cause", "effect", "lag"]].reset_index(drop=True)


def build_graph(edges: pd.DataFrame) -> nx.DiGraph:
    """Build a directed graph with lag stored as an edge attribute.

    Args:
        edges: DataFrame with columns cause, effect, lag.

    Returns:
        Directed graph; every node in NODE_LABELS is added so isolated
        variables still appear.
    """
    g = nx.DiGraph()
    g.add_nodes_from(NODE_LABELS.keys())
    for _, row in edges.iterrows():
        g.add_edge(row["cause"], row["effect"], lag=int(row["lag"]))
    return g


def plot_graph(g: nx.DiGraph, out_path: pathlib.Path) -> None:
    """Draw the candidate graph and save it as a vector PDF.

    Curved edges separate bidirectional pairs (e.g. temperature and price).

    Args:
        g: Directed candidate graph.
        out_path: Destination PDF path.
    """
    pos = nx.circular_layout(g)
    connectionstyle = "arc3,rad=0.12"

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.set_axis_off()

    nx.draw_networkx_nodes(
        g, pos, ax=ax, node_size=4200, node_color="#dfe7f2",
        edgecolors="#3a5a80", linewidths=1.5,
    )
    nx.draw_networkx_labels(
        g, pos, labels=NODE_LABELS, ax=ax, font_size=12, font_color="#1a1a1a",
    )
    nx.draw_networkx_edges(
        g, pos, ax=ax, node_size=4200, arrowsize=16, width=1.3,
        edge_color="#5a5a5a", connectionstyle=connectionstyle,
    )
    edge_labels = {(u, v): d["lag"] for u, v, d in g.edges(data=True)}
    nx.draw_networkx_edge_labels(
        g, pos, edge_labels=edge_labels, ax=ax, font_size=12,
        label_pos=0.5, connectionstyle=connectionstyle,
        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none"},
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate the candidate-graph figure."""
    edges = load_cross_variable_edges(EDGES_CSV)
    g = build_graph(edges)
    plot_graph(g, OUT_PDF)
    print(f"Wrote {OUT_PDF} ({g.number_of_edges()} cross-variable edges)")


if __name__ == "__main__":
    main()
