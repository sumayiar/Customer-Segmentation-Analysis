from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .data import build_customer_features, generate_customer_base, simulate_transactions
from .reporting import (
    plot_cluster_projection,
    plot_model_selection,
    plot_retention_matrix,
    plot_segment_value,
    write_executive_summary,
)


MODEL_COLUMNS = [
    "recency_days",
    "order_count",
    "revenue_per_month",
    "avg_order_value",
    "avg_discount_rate",
    "return_rate",
    "online_order_share",
    "unique_categories",
    "purchase_frequency_per_month",
    "active_month_ratio",
    "support_tickets",
]


def _prepare_model_matrix(customer_features: pd.DataFrame) -> pd.DataFrame:
    matrix = customer_features[MODEL_COLUMNS].copy()
    matrix["order_count"] = np.log1p(matrix["order_count"])
    matrix["revenue_per_month"] = np.log1p(matrix["revenue_per_month"])
    matrix["avg_order_value"] = np.log1p(matrix["avg_order_value"])
    matrix["recency_days"] = np.log1p(matrix["recency_days"])
    matrix["support_tickets"] = np.log1p(matrix["support_tickets"])
    return matrix


def evaluate_cluster_counts(
    model_matrix: pd.DataFrame,
    seed: int,
    cluster_range: range = range(4, 8),
) -> pd.DataFrame:
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(model_matrix)

    evaluations: List[dict] = []
    for cluster_count in cluster_range:
        kmeans = KMeans(n_clusters=cluster_count, random_state=seed, n_init=20)
        labels = kmeans.fit_predict(scaled_matrix)
        cluster_shares = pd.Series(labels).value_counts(normalize=True)
        evaluations.append(
            {
                "cluster_count": cluster_count,
                "inertia": float(kmeans.inertia_),
                "silhouette_score": float(silhouette_score(scaled_matrix, labels)),
                "calinski_harabasz_score": float(calinski_harabasz_score(scaled_matrix, labels)),
                "min_cluster_share": float(cluster_shares.min()),
                "max_cluster_share": float(cluster_shares.max()),
            }
        )

    return pd.DataFrame.from_records(evaluations)


def select_cluster_count(model_selection: pd.DataFrame) -> int:
    scored = model_selection.copy()
    best_silhouette = float(scored["silhouette_score"].max())
    eligible = scored[
        (scored["silhouette_score"] >= best_silhouette - 0.015)
        & (scored["min_cluster_share"] >= 0.12)
    ]

    if eligible.empty:
        chosen = scored.sort_values(["silhouette_score", "cluster_count"], ascending=[False, False]).iloc[0]
    else:
        chosen = eligible.sort_values(["cluster_count", "silhouette_score"], ascending=[False, False]).iloc[0]

    return int(chosen["cluster_count"])


def fit_customer_segments(
    customer_features: pd.DataFrame,
    model_selection: pd.DataFrame,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler, KMeans, float]:
    model_matrix = _prepare_model_matrix(customer_features)
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(model_matrix)

    best_k = select_cluster_count(model_selection)
    kmeans = KMeans(n_clusters=best_k, random_state=seed, n_init=20)
    cluster_labels = kmeans.fit_predict(scaled_matrix)

    agglomerative = AgglomerativeClustering(n_clusters=best_k)
    comparison_labels = agglomerative.fit_predict(scaled_matrix)
    cluster_agreement = float(adjusted_rand_score(cluster_labels, comparison_labels))

    segmented = customer_features.copy()
    segmented["cluster_id"] = cluster_labels

    projection = PCA(n_components=2, random_state=seed).fit_transform(scaled_matrix)
    projected_customers = segmented[["customer_id", "cluster_id"]].copy()
    projected_customers["pca_component_1"] = projection[:, 0]
    projected_customers["pca_component_2"] = projection[:, 1]

    return segmented, projected_customers, scaler, kmeans, cluster_agreement


def summarize_segments(segmented_customers: pd.DataFrame) -> pd.DataFrame:
    total_revenue = segmented_customers["total_spend"].sum()
    total_customers = len(segmented_customers)

    profile = (
        segmented_customers.groupby("cluster_id")
        .agg(
            customer_count=("customer_id", "count"),
            total_spend=("total_spend", "sum"),
            avg_order_value=("avg_order_value", "mean"),
            recency_days=("recency_days", "mean"),
            avg_discount_rate=("avg_discount_rate", "mean"),
            order_count=("order_count", "mean"),
            purchase_frequency_per_month=("purchase_frequency_per_month", "mean"),
            return_rate=("return_rate", "mean"),
            online_order_share=("online_order_share", "mean"),
            revenue_per_month=("revenue_per_month", "mean"),
            active_month_ratio=("active_month_ratio", "mean"),
            support_tickets=("support_tickets", "mean"),
        )
        .reset_index()
    )

    profile["revenue_share"] = profile["total_spend"] / total_revenue
    profile["segment_size_pct"] = profile["customer_count"] / total_customers
    return profile


def assign_segment_names(segment_profiles: pd.DataFrame) -> pd.DataFrame:
    labeled = segment_profiles.copy()

    labeled["spend_rank"] = labeled["revenue_per_month"].rank(method="dense", ascending=False)
    labeled["recency_rank"] = labeled["recency_days"].rank(method="dense", ascending=False)
    labeled["discount_rank"] = labeled["avg_discount_rate"].rank(method="dense", ascending=False)
    labeled["frequency_rank"] = labeled["purchase_frequency_per_month"].rank(method="dense", ascending=False)

    names: Dict[int, str] = {}

    champion_cluster = int(
        labeled.sort_values(["revenue_per_month", "purchase_frequency_per_month"], ascending=False)
        .iloc[0]["cluster_id"]
    )
    names[champion_cluster] = "Champions"

    at_risk_candidates = labeled[~labeled["cluster_id"].isin(names.keys())]
    if not at_risk_candidates.empty:
        at_risk_cluster = int(
            at_risk_candidates.sort_values(["revenue_per_month", "recency_days"], ascending=[False, False])
            .iloc[0]["cluster_id"]
        )
        names[at_risk_cluster] = "High-Value At Risk"

    discount_candidates = labeled[
        ~labeled["cluster_id"].isin(names.keys())
    ]
    if not discount_candidates.empty:
        discount_cluster = int(
            discount_candidates.sort_values(
                ["avg_discount_rate", "avg_order_value"],
                ascending=[False, True],
            ).iloc[0]["cluster_id"]
        )
        names[discount_cluster] = "Discount-Driven"

    dormant_candidates = labeled[~labeled["cluster_id"].isin(names.keys())]
    if not dormant_candidates.empty:
        dormant_cluster = int(
            dormant_candidates.sort_values(["recency_days", "revenue_per_month"], ascending=[False, True])
            .iloc[0]["cluster_id"]
        )
        names[dormant_cluster] = "Dormant"

    fallback_names = ["Growth Loyalists", "Steady Core", "Emerging Customers", "Digital Convenience"]
    fallback_index = 0
    for cluster_id in labeled["cluster_id"]:
        if int(cluster_id) not in names:
            names[int(cluster_id)] = fallback_names[fallback_index]
            fallback_index += 1

    labeled["segment_name"] = labeled["cluster_id"].map(names)
    return labeled.drop(columns=["spend_rank", "recency_rank", "discount_rank", "frequency_rank"])


def map_segment_names(segmented_customers: pd.DataFrame, segment_profiles: pd.DataFrame) -> pd.DataFrame:
    name_lookup = segment_profiles.set_index("cluster_id")["segment_name"]
    enriched = segmented_customers.copy()
    enriched["segment_name"] = enriched["cluster_id"].map(name_lookup)
    return enriched


def build_retention_playbook(segment_profiles: pd.DataFrame) -> List[dict]:
    playbook: List[dict] = []

    for row in segment_profiles.sort_values("revenue_share", ascending=False).itertuples(index=False):
        if row.segment_name == "Champions":
            opportunity = (
                f"Protect {row.revenue_share:.1%} of revenue coming from your most active customers before competitors do."
            )
            action = "Launch VIP early-access drops, concierge service, and referral incentives."
        elif row.segment_name == "High-Value At Risk":
            opportunity = (
                f"Win back premium shoppers showing a {row.recency_days:.0f}-day average gap between purchases."
            )
            action = "Trigger churn-save journeys with personalized offers and category-specific reminders."
        elif row.segment_name == "Discount-Driven":
            opportunity = (
                f"Reduce margin pressure from a segment averaging {row.avg_discount_rate:.1%} discount dependence."
            )
            action = "Shift from blanket discounts to bundles, loyalty points, and threshold-based promotions."
        elif row.segment_name == "Dormant":
            opportunity = (
                f"Contain spend on low-response audiences while testing efficient reactivation hooks."
            )
            action = "Use low-cost win-back campaigns, then suppress long-unresponsive customers from paid media."
        else:
            opportunity = (
                f"Accelerate growth in a segment with {row.purchase_frequency_per_month:.2f} purchases per month."
            )
            action = "Nurture with cross-sell journeys, replenishment prompts, and category education."

        playbook.append(
            {
                "segment_name": row.segment_name,
                "opportunity": opportunity,
                "action": action,
            }
        )

    return playbook


def build_summary_metrics(
    segmented_customers: pd.DataFrame,
    transactions: pd.DataFrame,
    model_selection: pd.DataFrame,
    cluster_agreement: float,
    segment_profiles: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> Dict[str, float]:
    best_k = select_cluster_count(model_selection)
    silhouette_winner = model_selection.sort_values(
        ["silhouette_score", "cluster_count"],
        ascending=[False, False],
    ).iloc[0]
    selected_row = model_selection.loc[model_selection["cluster_count"] == best_k].iloc[0]
    top_segment = segment_profiles.sort_values("revenue_share", ascending=False).iloc[0]

    at_risk_rows = segment_profiles[segment_profiles["segment_name"] == "High-Value At Risk"]
    if at_risk_rows.empty:
        at_risk_rows = segment_profiles.sort_values(["recency_days", "revenue_share"], ascending=[False, False]).head(1)
    at_risk_row = at_risk_rows.iloc[0]

    return {
        "analysis_date": str(as_of_date.date()),
        "customer_count": float(segmented_customers["customer_id"].nunique()),
        "order_count": float(transactions["order_id"].nunique()),
        "best_k": float(best_k),
        "selected_silhouette": float(selected_row["silhouette_score"]),
        "cluster_agreement": cluster_agreement,
        "silhouette_winner_k": float(silhouette_winner["cluster_count"]),
        "silhouette_gap": float(silhouette_winner["silhouette_score"] - selected_row["silhouette_score"]),
        "top_segment_revenue_share": float(top_segment["revenue_share"]),
        "top_segment_customer_share": float(top_segment["segment_size_pct"]),
        "at_risk_revenue_share": float(at_risk_row["revenue_share"]),
        "at_risk_recency": float(at_risk_row["recency_days"]),
    }


def export_outputs(
    project_root: Path,
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    model_selection: pd.DataFrame,
    segment_profiles: pd.DataFrame,
    segmented_customers: pd.DataFrame,
    projected_customers: pd.DataFrame,
    summary_metrics: Dict[str, float],
    playbook: List[dict],
) -> None:
    data_dir = project_root / "data" / "processed"
    output_dir = project_root / "outputs"
    figures_dir = output_dir / "figures"
    docs_dir = project_root / "docs"

    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    customers.to_csv(data_dir / "customer_base.csv", index=False)
    transactions.to_csv(data_dir / "transactions.csv", index=False)
    model_selection.to_csv(output_dir / "model_selection.csv", index=False)
    segment_profiles.to_csv(output_dir / "segment_profiles.csv", index=False)
    segmented_customers.to_csv(output_dir / "customer_segments.csv", index=False)
    pd.DataFrame.from_records(playbook).to_csv(output_dir / "retention_playbook.csv", index=False)

    plot_model_selection(model_selection, figures_dir / "model_selection.png")
    plot_segment_value(segment_profiles, figures_dir / "segment_value.png")
    plot_retention_matrix(segment_profiles, figures_dir / "retention_matrix.png")

    named_projection = projected_customers.merge(
        segment_profiles[["cluster_id", "segment_name"]],
        on="cluster_id",
        how="left",
    )
    plot_cluster_projection(named_projection, figures_dir / "cluster_projection.png")

    highlight_segments = {
        "top_value_segment": str(
            segment_profiles.sort_values("revenue_share", ascending=False).iloc[0]["segment_name"]
        ),
        "at_risk_segment": str(
            (
                segment_profiles[segment_profiles["segment_name"] == "High-Value At Risk"]
                .head(1)
                .get("segment_name", pd.Series(dtype=str))
            ).iloc[0]
            if not segment_profiles[segment_profiles["segment_name"] == "High-Value At Risk"].empty
            else segment_profiles.sort_values(["recency_days", "revenue_share"], ascending=[False, False]).iloc[0][
                "segment_name"
            ]
        ),
    }
    write_executive_summary(output_dir / "executive_summary.md", summary_metrics, highlight_segments, playbook)
    write_project_story(docs_dir / "project_story.md", summary_metrics, segment_profiles, playbook)


def write_project_story(
    output_path: Path,
    summary_metrics: Dict[str, float],
    segment_profiles: pd.DataFrame,
    playbook: List[dict],
) -> None:
    if int(summary_metrics["best_k"]) != int(summary_metrics["silhouette_winner_k"]):
        clustering_step = (
            f"3. Evaluated cluster counts with KMeans, then selected {int(summary_metrics['best_k'])} clusters because it stayed within "
            f"{summary_metrics['silhouette_gap']:.3f} silhouette points of the best score while creating more actionable segments."
        )
    else:
        clustering_step = (
            f"3. Evaluated cluster counts with KMeans and selected {int(summary_metrics['best_k'])} clusters using silhouette score."
        )

    top_segments = segment_profiles.sort_values("revenue_share", ascending=False).head(3)
    lines = [
        "# Project Story",
        "",
        "## Objective",
        "",
        "Build a portfolio-ready customer analytics project that segments shoppers by purchasing behavior, isolates the highest-value audiences, and turns those segments into practical retention and targeting recommendations.",
        "",
        "## Approach",
        "",
        "1. Generated a realistic 12-month retail transaction history for 1,200 customers.",
        "2. Engineered behavioral features such as recency, order frequency, spend, discount intensity, and channel mix.",
        clustering_step,
        "4. Translated clusters into business-facing segment names and retention actions.",
        "",
        "## Key Findings",
        "",
    ]

    for row in top_segments.itertuples(index=False):
        lines.append(
            f"- {row.segment_name}: {row.segment_size_pct:.1%} of customers contribute {row.revenue_share:.1%} of revenue with an average order value of ${row.avg_order_value:,.0f}."
        )

    lines.extend(
        [
            "",
            "## Retention Opportunities",
            "",
        ]
    )

    for item in playbook:
        lines.append(f"- {item['segment_name']}: {item['action']}")

    lines.extend(
        [
            "",
            "## Business Strategy Impact",
            "",
            "- Concentrate premium lifecycle messaging on high-value customers with strong spend but widening recency gaps.",
            "- Improve margin discipline by replacing always-on discounts with loyalty mechanics for price-sensitive segments.",
            "- Use the segment output in CRM, paid media suppression lists, and merchandising decisions to align spend with customer value.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(
    project_root: Path,
    customer_count: int = 1200,
    seed: int = 42,
    as_of_date: str = "2026-03-26",
) -> Dict[str, float]:
    root = Path(project_root)
    analysis_date = pd.Timestamp(as_of_date)

    customers = generate_customer_base(customer_count=customer_count, seed=seed, as_of_date=analysis_date)
    transactions = simulate_transactions(customers=customers, seed=seed, as_of_date=analysis_date)
    customer_features = build_customer_features(
        customers=customers,
        transactions=transactions,
        as_of_date=analysis_date,
        seed=seed,
    )

    model_selection = evaluate_cluster_counts(_prepare_model_matrix(customer_features), seed=seed)
    selected_cluster_count = select_cluster_count(model_selection)
    model_selection["selected_for_business_use"] = (
        model_selection["cluster_count"] == selected_cluster_count
    )
    segmented_customers, projected_customers, _scaler, _kmeans, cluster_agreement = fit_customer_segments(
        customer_features=customer_features,
        model_selection=model_selection,
        seed=seed,
    )

    segment_profiles = assign_segment_names(summarize_segments(segmented_customers))
    segmented_customers = map_segment_names(segmented_customers, segment_profiles)

    summary_metrics = build_summary_metrics(
        segmented_customers=segmented_customers,
        transactions=transactions,
        model_selection=model_selection,
        cluster_agreement=cluster_agreement,
        segment_profiles=segment_profiles,
        as_of_date=analysis_date,
    )
    playbook = build_retention_playbook(segment_profiles)

    export_outputs(
        project_root=root,
        customers=customers,
        transactions=transactions,
        model_selection=model_selection,
        segment_profiles=segment_profiles,
        segmented_customers=segmented_customers,
        projected_customers=projected_customers,
        summary_metrics=summary_metrics,
        playbook=playbook,
    )

    return summary_metrics
