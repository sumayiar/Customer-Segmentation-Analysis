from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

_MPLCONFIGDIR = Path(__file__).resolve().parents[2] / ".mplconfig"
_MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="talk")


def _save_figure(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_model_selection(model_selection: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=model_selection, x="cluster_count", y="silhouette_score", marker="o", linewidth=2.5)
    plt.title("Silhouette Score By Cluster Count")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    best_row = model_selection.loc[model_selection["silhouette_score"].idxmax()]
    plt.axvline(best_row["cluster_count"], color="#d95f02", linestyle="--", linewidth=1.5)
    plt.text(
        best_row["cluster_count"] + 0.05,
        best_row["silhouette_score"],
        f"Best k={int(best_row['cluster_count'])}",
        color="#d95f02",
    )
    if "selected_for_business_use" in model_selection.columns:
        selected_rows = model_selection[model_selection["selected_for_business_use"]]
        if not selected_rows.empty:
            selected_row = selected_rows.iloc[0]
            plt.scatter(
                selected_row["cluster_count"],
                selected_row["silhouette_score"],
                color="#1b9e77",
                s=120,
                zorder=5,
            )
            plt.text(
                selected_row["cluster_count"] + 0.05,
                selected_row["silhouette_score"] - 0.008,
                f"Selected k={int(selected_row['cluster_count'])}",
                color="#1b9e77",
            )
    _save_figure(output_path)


def plot_segment_value(segment_profiles: pd.DataFrame, output_path: Path) -> None:
    ordered = segment_profiles.sort_values("revenue_share", ascending=False)
    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=ordered,
        x="revenue_share",
        y="segment_name",
        hue="segment_name",
        palette="crest",
        orient="h",
        dodge=False,
        legend=False,
    )
    plt.title("Revenue Contribution By Segment")
    plt.xlabel("Revenue Share")
    plt.ylabel("")
    plt.xlim(0, max(0.05, ordered["revenue_share"].max() * 1.15))
    for index, row in ordered.reset_index(drop=True).iterrows():
        plt.text(
            row["revenue_share"] + 0.004,
            index,
            f"{row['segment_size_pct']:.1%} of customers",
            va="center",
            fontsize=11,
        )
    _save_figure(output_path)


def plot_retention_matrix(segment_profiles: pd.DataFrame, output_path: Path) -> None:
    plot_data = segment_profiles.copy()
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=plot_data,
        x="recency_days",
        y="avg_order_value",
        size="revenue_share",
        sizes=(240, 1800),
        hue="segment_name",
        palette="tab10",
        legend=False,
    )
    plt.title("Retention Opportunity Matrix")
    plt.xlabel("Average Recency (Days)")
    plt.ylabel("Average Order Value")
    for row in plot_data.itertuples(index=False):
        plt.text(row.recency_days + 1.5, row.avg_order_value + 1.5, row.segment_name, fontsize=10)
    _save_figure(output_path)


def plot_cluster_projection(projected_customers: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=projected_customers,
        x="pca_component_1",
        y="pca_component_2",
        hue="segment_name",
        palette="tab10",
        alpha=0.75,
        s=65,
    )
    plt.title("Customer Segments In PCA Space")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(loc="best", fontsize=10)
    _save_figure(output_path)


def write_executive_summary(
    output_path: Path,
    summary_metrics: Dict[str, float],
    highlight_segments: Dict[str, str],
    opportunities: List[dict],
) -> None:
    if int(summary_metrics["best_k"]) != int(summary_metrics["silhouette_winner_k"]):
        clustering_line = (
            f"- Selected clustering configuration: KMeans with {int(summary_metrics['best_k'])} clusters, "
            f"validated against agglomerative clustering (ARI {summary_metrics['cluster_agreement']:.2f}). "
            f"This stayed within {summary_metrics['silhouette_gap']:.3f} silhouette points of the top statistical score "
            "while yielding a more actionable segment split."
        )
    else:
        clustering_line = (
            f"- Selected clustering configuration: KMeans with {int(summary_metrics['best_k'])} clusters, "
            f"validated against agglomerative clustering (ARI {summary_metrics['cluster_agreement']:.2f})."
        )

    lines = [
        "# Executive Summary",
        "",
        f"- Analysis window: trailing 12 months ending {summary_metrics['analysis_date']}.",
        f"- Customers analyzed: {int(summary_metrics['customer_count']):,}.",
        f"- Orders analyzed: {int(summary_metrics['order_count']):,}.",
        clustering_line,
        f"- Highest-value segment: {highlight_segments['top_value_segment']} with {summary_metrics['top_segment_revenue_share']:.1%} of revenue from {summary_metrics['top_segment_customer_share']:.1%} of customers.",
        f"- Highest-risk premium segment: {highlight_segments['at_risk_segment']} with {summary_metrics['at_risk_revenue_share']:.1%} of revenue and an average recency of {summary_metrics['at_risk_recency']:.0f} days.",
        "",
        "## Retention Opportunities",
        "",
    ]

    for item in opportunities:
        lines.append(
            f"- {item['segment_name']}: {item['opportunity']} Recommended action: {item['action']}"
        )

    lines.append("")
    lines.append("## Strategic Readout")
    lines.append("")
    lines.append(
        "- Prioritize lifecycle marketing around high-value segments, then protect margin by moving discount-led customers toward bundles and loyalty benefits instead of blanket promotions."
    )
    lines.append(
        "- Use recency, spend, and discount intensity together in CRM scoring so retention budgets go to segments where revenue upside and churn risk overlap."
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
