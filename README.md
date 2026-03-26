# Customer Segmentation Analysis

A portfolio-ready analytics project that segments retail customers by purchasing behavior, identifies high-value audiences, and turns clustering results into retention and targeting recommendations.

## Project Highlights

- Applied clustering techniques to group customers using recency, frequency, spend, discount dependence, channel mix, and engagement patterns.
- Identified high-value customer groups and a premium at-risk segment that represents meaningful revenue retention upside.
- Generated business-facing insights and a retention playbook to improve lifecycle targeting, promotional strategy, and CRM prioritization.
- Packaged the project with reproducible Python code, exported CSV outputs, and business-ready documentation.

## Key Results

- Selected a 5-cluster KMeans solution for the trailing 12 months ending **March 26, 2026**, then validated it against agglomerative clustering.
- **Growth Loyalists** represent **31.2%** of customers and drive **54.1%** of revenue.
- **High-Value At Risk** represent **13.8%** of customers, contribute **20.9%** of revenue, and show an average recency of **150 days**.
- **Champions** account for **13.5%** of customers with the highest purchase frequency at **1.15 orders per month**.
- **Discount-Driven** customers show **44.8%** average discount dependence, making them a clear margin-optimization opportunity.

## Method

1. Generated a realistic synthetic retail transaction history for 1,200 customers.
2. Engineered customer-level features from transaction behavior.
3. Evaluated cluster counts from 4 to 7 using silhouette score and selected the most actionable balanced solution.
4. Labeled each cluster with a business-friendly segment name and exported a retention playbook.

## Tech Stack

- Python 3.9+
- pandas and NumPy
- scikit-learn
- matplotlib and seaborn

## Quick Start

### 1. Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

### 2. Run the analysis

```bash
python3 -m src.customer_segmentation_analysis.cli run-analysis
```

This generates:

- Processed customer and transaction data in `data/processed/`
- Cluster scoring outputs in `outputs/`
- Narrative documentation in `docs/project_story.md`

## Outputs

- `outputs/customer_segments.csv`
- `outputs/segment_profiles.csv`
- `outputs/retention_playbook.csv`
- `outputs/model_selection.csv`
- `outputs/executive_summary.md`

## Business Recommendations

- Prioritize lifecycle campaigns around **Growth Loyalists** to deepen cross-sell and repeat purchase behavior.
- Launch win-back flows for **High-Value At Risk** customers before they fully churn.
- Protect **Champions** with loyalty perks, exclusives, and referral incentives.
- Move **Discount-Driven** customers toward bundles and threshold offers to preserve margin.
- Limit paid reactivation spend on deeply **Dormant** customers unless they respond to low-cost win-back tests.

## Repository Structure

```text
.
├── README.md
├── data/
│   └── processed/
├── docs/
│   └── project_story.md
├── outputs/
│   ├── customer_segments.csv
│   ├── executive_summary.md
│   ├── model_selection.csv
│   ├── retention_playbook.csv
│   └── segment_profiles.csv
├── src/
│   └── customer_segmentation_analysis/
│       ├── analysis.py
│       ├── cli.py
│       ├── data.py
│       └── reporting.py
└── tests/
    └── test_pipeline.py
```
