from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Archetype:
    weight: float
    order_lambda: float
    average_order_value: float
    order_value_sigma: float
    recency_range: tuple[int, int]
    time_beta: tuple[float, float]
    discount_beta: tuple[float, float]
    return_beta: tuple[float, float]
    digital_beta: tuple[float, float]
    tenure_range: tuple[int, int]
    support_lambda: float
    category_weights: Dict[str, float]


ARCHETYPES: Dict[str, Archetype] = {
    "Champions": Archetype(
        weight=0.18,
        order_lambda=16,
        average_order_value=175,
        order_value_sigma=0.28,
        recency_range=(2, 12),
        time_beta=(1.6, 3.2),
        discount_beta=(2, 17),
        return_beta=(2, 45),
        digital_beta=(5, 5),
        tenure_range=(16, 36),
        support_lambda=0.5,
        category_weights={
            "Electronics": 0.32,
            "Fashion": 0.18,
            "Home": 0.18,
            "Beauty": 0.12,
            "Sports": 0.12,
            "Grocery": 0.08,
        },
    ),
    "Loyal Growth": Archetype(
        weight=0.24,
        order_lambda=14,
        average_order_value=125,
        order_value_sigma=0.32,
        recency_range=(6, 30),
        time_beta=(1.8, 2.8),
        discount_beta=(3, 10),
        return_beta=(2, 30),
        digital_beta=(8, 2),
        tenure_range=(10, 28),
        support_lambda=0.7,
        category_weights={
            "Fashion": 0.24,
            "Beauty": 0.21,
            "Home": 0.19,
            "Electronics": 0.16,
            "Sports": 0.12,
            "Grocery": 0.08,
        },
    ),
    "High-Value At Risk": Archetype(
        weight=0.14,
        order_lambda=11,
        average_order_value=210,
        order_value_sigma=0.35,
        recency_range=(90, 220),
        time_beta=(5.2, 1.5),
        discount_beta=(2.5, 9),
        return_beta=(3.4, 15),
        digital_beta=(5, 5),
        tenure_range=(12, 36),
        support_lambda=2.1,
        category_weights={
            "Electronics": 0.3,
            "Home": 0.24,
            "Fashion": 0.16,
            "Sports": 0.14,
            "Beauty": 0.1,
            "Grocery": 0.06,
        },
    ),
    "Discount-Led": Archetype(
        weight=0.25,
        order_lambda=9,
        average_order_value=82,
        order_value_sigma=0.3,
        recency_range=(18, 95),
        time_beta=(2.4, 2.4),
        discount_beta=(6, 7),
        return_beta=(2.4, 18),
        digital_beta=(8, 2),
        tenure_range=(8, 24),
        support_lambda=0.8,
        category_weights={
            "Fashion": 0.28,
            "Beauty": 0.22,
            "Home": 0.18,
            "Grocery": 0.15,
            "Sports": 0.1,
            "Electronics": 0.07,
        },
    ),
    "Dormant": Archetype(
        weight=0.19,
        order_lambda=4,
        average_order_value=58,
        order_value_sigma=0.36,
        recency_range=(150, 330),
        time_beta=(6.0, 1.4),
        discount_beta=(4, 8),
        return_beta=(2.4, 16),
        digital_beta=(4, 6),
        tenure_range=(6, 22),
        support_lambda=1.1,
        category_weights={
            "Grocery": 0.22,
            "Home": 0.21,
            "Beauty": 0.18,
            "Fashion": 0.16,
            "Sports": 0.12,
            "Electronics": 0.11,
        },
    ),
}


def _weighted_choice(rng: np.random.Generator, category_weights: Dict[str, float]) -> str:
    categories = list(category_weights.keys())
    weights = np.array(list(category_weights.values()), dtype=float)
    probabilities = weights / weights.sum()
    return str(rng.choice(categories, p=probabilities))


def generate_customer_base(
    customer_count: int,
    seed: int,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    archetype_names = list(ARCHETYPES.keys())
    probabilities = np.array([ARCHETYPES[name].weight for name in archetype_names], dtype=float)
    probabilities = probabilities / probabilities.sum()

    assigned = rng.choice(archetype_names, size=customer_count, p=probabilities)

    records: List[dict] = []
    for index, persona in enumerate(assigned, start=1):
        profile = ARCHETYPES[str(persona)]
        tenure_months = int(rng.integers(profile.tenure_range[0], profile.tenure_range[1] + 1))
        signup_date = as_of_date - pd.Timedelta(days=int(tenure_months * 30.4))
        support_ticket_bias = int(rng.poisson(profile.support_lambda))

        records.append(
            {
                "customer_id": f"C{index:05d}",
                "latent_persona": str(persona),
                "tenure_months": tenure_months,
                "signup_date": signup_date.normalize(),
                "support_ticket_bias": support_ticket_bias,
            }
        )

    return pd.DataFrame.from_records(records)


def simulate_transactions(
    customers: pd.DataFrame,
    seed: int,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 11)
    records: List[dict] = []

    for customer in customers.itertuples(index=False):
        profile = ARCHETYPES[customer.latent_persona]
        order_count = max(1, int(rng.poisson(profile.order_lambda)))
        window_days = max(45, min(365, int(customer.tenure_months * 30.4)))
        recency_days = int(
            rng.integers(
                profile.recency_range[0],
                min(profile.recency_range[1], window_days - 1) + 1,
            )
        )

        if order_count == 1:
            offsets = np.array([recency_days], dtype=int)
        else:
            historical_offsets = recency_days + rng.beta(
                profile.time_beta[0],
                profile.time_beta[1],
                size=order_count - 1,
            ) * (window_days - recency_days)
            offsets = np.concatenate([historical_offsets, [recency_days]]).round().astype(int)

        offsets = np.clip(offsets, 1, window_days)
        order_dates = pd.to_datetime(as_of_date - pd.to_timedelta(offsets, unit="D")).normalize()

        for order_idx, order_date in enumerate(order_dates, start=1):
            gross_sales = float(
                np.clip(
                    rng.lognormal(mean=np.log(profile.average_order_value), sigma=profile.order_value_sigma),
                    18,
                    profile.average_order_value * 4.5,
                )
            )
            discount_rate = float(np.clip(rng.beta(*profile.discount_beta), 0.01, 0.65))
            returned = int(rng.random() < float(np.clip(rng.beta(*profile.return_beta), 0.0, 0.55)))
            online_order = int(rng.random() < rng.beta(*profile.digital_beta))
            units = int(max(1, rng.poisson(2.4)))
            category = _weighted_choice(rng, profile.category_weights)
            net_sales = round(gross_sales * (1 - discount_rate), 2)

            records.append(
                {
                    "customer_id": customer.customer_id,
                    "order_id": f"{customer.customer_id}-O{order_idx:03d}",
                    "order_date": order_date,
                    "category": category,
                    "channel": "Online" if online_order else "Store",
                    "units": units,
                    "gross_sales": round(gross_sales, 2),
                    "discount_rate": round(discount_rate, 4),
                    "net_sales": net_sales,
                    "returned": returned,
                }
            )

    transactions = pd.DataFrame.from_records(records).sort_values(["customer_id", "order_date", "order_id"])
    return transactions.reset_index(drop=True)


def build_customer_features(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    as_of_date: pd.Timestamp,
    seed: int,
) -> pd.DataFrame:
    monthly_activity = transactions.assign(order_month=transactions["order_date"].dt.to_period("M").astype(str))
    aggregated = (
        monthly_activity.groupby("customer_id")
        .agg(
            order_count=("order_id", "count"),
            total_spend=("net_sales", "sum"),
            avg_order_value=("net_sales", "mean"),
            avg_discount_rate=("discount_rate", "mean"),
            return_rate=("returned", "mean"),
            online_order_share=("channel", lambda values: (values == "Online").mean()),
            unique_categories=("category", "nunique"),
            active_months=("order_month", "nunique"),
            total_units=("units", "sum"),
            last_order_date=("order_date", "max"),
            first_order_date=("order_date", "min"),
        )
        .reset_index()
    )

    customer_features = customers.merge(aggregated, on="customer_id", how="left")
    customer_features["recency_days"] = (
        pd.to_datetime(as_of_date).normalize() - customer_features["last_order_date"]
    ).dt.days
    customer_features["purchase_frequency_per_month"] = (
        customer_features["order_count"] / customer_features["tenure_months"].clip(lower=1)
    )
    customer_features["revenue_per_month"] = (
        customer_features["total_spend"] / customer_features["tenure_months"].clip(lower=1)
    )
    customer_features["active_month_ratio"] = (
        customer_features["active_months"] / customer_features["tenure_months"].clip(lower=1)
    )
    customer_features["avg_units_per_order"] = (
        customer_features["total_units"] / customer_features["order_count"].clip(lower=1)
    )

    rng = np.random.default_rng(seed + 23)
    ticket_adjustment = rng.poisson(
        np.clip(
            customer_features["return_rate"].fillna(0) * 3
            + customer_features["recency_days"].fillna(0) / 120,
            0.1,
            None,
        )
    )
    customer_features["support_tickets"] = customer_features["support_ticket_bias"] + ticket_adjustment

    numeric_columns = [
        "order_count",
        "total_spend",
        "avg_order_value",
        "avg_discount_rate",
        "return_rate",
        "online_order_share",
        "unique_categories",
        "active_months",
        "total_units",
        "recency_days",
        "purchase_frequency_per_month",
        "revenue_per_month",
        "active_month_ratio",
        "avg_units_per_order",
        "support_tickets",
    ]
    customer_features[numeric_columns] = customer_features[numeric_columns].fillna(0)

    rounded_columns = [
        "total_spend",
        "avg_order_value",
        "avg_discount_rate",
        "return_rate",
        "online_order_share",
        "purchase_frequency_per_month",
        "revenue_per_month",
        "active_month_ratio",
        "avg_units_per_order",
    ]
    customer_features[rounded_columns] = customer_features[rounded_columns].round(4)

    return customer_features.sort_values("customer_id").reset_index(drop=True)
