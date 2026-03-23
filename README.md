# Data Analysis & AI Portfolio

Data Analyst with 4+ years of experience in exploratory data analysis, statistical testing, predictive modeling, and AWS data infrastructure. This repository showcases hands-on projects demonstrating analytical depth, Python proficiency, and AI/ML capabilities.

## Projects

| # | Project | Skills Demonstrated |
|---|---------|-------------------|
| 01 | [EDA & Statistical Analysis](#01--eda--statistical-analysis-toolkit) | Hypothesis testing, correlation analysis, distribution profiling, A/B testing |
| 02 | [Data Cleaning Pipeline](#02--automated-data-cleaning--validation-pipeline) | Automated data quality checks, deduplication, outlier detection, validation |
| 03 | [Customer Churn — ML Pipeline](#03--customer-churn-analysis--predictive-modeling) | Feature engineering, classification models, XGBoost, model evaluation |
| 04 | [Time Series Forecasting](#04--time-series-forecasting--demand--capacity-analysis) | Trend analysis, seasonality decomposition, ARIMA, demand forecasting |
| 05 | [SQL Analytics Toolkit](#05--sql-analytics-toolkit) | CTEs, window functions, KPI queries, SLA reporting, ad-hoc analysis |
| 06 | [AWS Serverless Data Pipeline](#06--aws-serverless-data-pipeline) | AWS Glue, Lambda, S3, Redshift, ETL automation |

## Tech Stack

- **Python**: pandas, NumPy, SciPy, Matplotlib, Seaborn, scikit-learn, XGBoost, statsmodels, NLTK
- **SQL**: PostgreSQL, Redshift, complex joins, CTEs, window functions
- **AWS**: S3, Glue, Lambda, Redshift, Athena, CloudWatch
- **BI Tools**: Power BI, Tableau, Jupyter Notebooks
- **ML/AI**: Classification, regression, clustering, time series forecasting, NLP

---

## 01 — EDA & Statistical Analysis Toolkit

[`01_eda_statistical_analysis/eda_analysis.py`](./01_eda_statistical_analysis/eda_analysis.py)

Reusable Python toolkit for deep-dive exploratory data analysis on operational datasets.

### What It Does

- **Data Profiling** — Generates a full quality profile for every column: null rates, data types, duplicate counts, distribution stats (mean, median, skewness, kurtosis), and memory usage
- **Hypothesis Testing** — Runs t-tests (independent & paired) with normality checks (Shapiro-Wilk), equal variance testing (Levene's), and effect size calculation (Cohen's d)
- **Chi-Square Test** — Tests independence between categorical variables with Cramér's V for effect size
- **ANOVA** — One-way ANOVA across multiple groups with group mean comparison
- **A/B Testing** — Supports both proportion-based (conversion rate z-test) and continuous metric (Welch's t-test) A/B analyses with lift calculation and winner determination
- **Correlation Analysis** — Generates heatmaps and automatically flags highly correlated feature pairs above a configurable threshold
- **Outlier Detection** — Identifies outliers using IQR (1.5×IQR) or Z-score (|z| > 3) methods with per-column summary

### Usage

```python
from eda_analysis import profile_dataset, run_ttest, ab_test, detect_outliers

# Profile entire dataset
profile = profile_dataset(df)

# Hypothesis test — do escalated tickets take longer to resolve?
result = run_ttest(escalated_times, normal_times, alpha=0.05)

# A/B test — phone vs chat first-call resolution rate
ab_result = ab_test(phone_group, chat_group, metric_type="proportion")

# Outlier detection
outliers = detect_outliers(df, method="iqr")
```

### Sample Output

```
══════════════════════════════════════════════════
  A/B TEST RESULTS
══════════════════════════════════════════════════
  test: Two-Proportion Z-Test
  control_rate: 0.6512
  treatment_rate: 0.6634
  lift_pct: +1.87%
  p_value: 0.4231
  significant: False
  WINNER: NO SIGNIFICANT DIFFERENCE
══════════════════════════════════════════════════
```

### Tech Stack
`pandas` · `NumPy` · `SciPy` · `Matplotlib` · `Seaborn`

---

## 02 — Automated Data Cleaning & Validation Pipeline

[`02_data_cleaning_pipeline/data_cleaning_pipeline.py`](./02_data_cleaning_pipeline/data_cleaning_pipeline.py)

Production-grade data cleaning framework with chainable operations, configurable strategies, and built-in quality reporting.

### What It Does

- **Schema Validation** — Validates column presence and enforces data types with automatic coercion
- **Deduplication** — Removes exact duplicates based on configurable subset columns (keep first, last, or remove all)
- **Null Handling** — Three strategies: `smart` (median for numeric, mode for categorical, drop columns >50% null), `drop` (remove all null rows), or `custom` (user-defined fill values)
- **Outlier Treatment** — IQR or Z-score detection with `cap` (winsorize), `remove`, or `flag` (add boolean column) treatment options
- **Format Standardization** — Strips whitespace, normalizes multi-spaces, cleans string formatting
- **Quality Scoring** — Calculates a 0–100 composite score based on completeness (40%), uniqueness (30%), and consistency (30%)
- **Audit Trail** — Every action is logged with timestamps, row counts affected, and details for full reproducibility

### Usage

```python
from data_cleaning_pipeline import DataCleaningPipeline, data_quality_score

pipeline = DataCleaningPipeline(df)
clean_df, report = (pipeline
    .validate_schema({"id": "int", "date": "datetime", "amount": "float"})
    .remove_duplicates(subset=["id"])
    .handle_nulls(strategy="smart")
    .detect_and_treat_outliers(columns=["amount"], method="iqr", treatment="cap")
    .standardize_formats()
    .run())

# Check quality score
scores = data_quality_score(clean_df)
```

### Sample Output

```
══════════════════════════════════════════════════════════════
  RUNNING DATA CLEANING PIPELINE (5 steps)
══════════════════════════════════════════════════════════════

Step: Schema Validation
  ✓ Schema validated: 6 columns checked, 2 types coerced
Step: Deduplication
  ✓ Deduplication: 200 duplicates removed (1.9%)
Step: Null Handling
  ✓ Null handling (smart): 1,012 nulls resolved, 0 remaining
Step: Outlier Treatment
  ✓ Outliers (iqr/cap): 487 values treated across 2 columns
Step: Format Standardization
  ✓ Format standardization: 3,241 string values cleaned

══════════════════════════════════════════════════════════════
  PIPELINE COMPLETE
══════════════════════════════════════════════════════════════
  Input:  10,200 rows
  Output: 10,000 rows
  Removed: 200 rows (1.9%)
══════════════════════════════════════════════════════════════

  DATA QUALITY SCORECARD
────────────────────────────────────
  Overall Score:  96.4/100
  Completeness:   100.0%
  Uniqueness:     100.0%
  Consistency:    88.9%
────────────────────────────────────
```

### Tech Stack
`pandas` · `NumPy` · `SciPy`

---

## 03 — Customer Churn Analysis & Predictive Modeling

[`03_customer_churn_ml/churn_analysis.py`](./03_customer_churn_ml/churn_analysis.py)

End-to-end ML pipeline for telecom customer churn prediction on a 50K-record dataset.

### Pipeline

```
Data Generation → EDA & Statistical Profiling → Feature Engineering → Model Training → Evaluation → Business Recommendations
```

### What It Does

- **EDA & Statistical Profiling** — Churn rate segmentation by contract, internet service, payment method. Distribution comparison (churned vs retained). Statistical significance testing (t-test on tenure difference)
- **Feature Engineering** — Creates interaction features: charge-per-tenure, tickets-per-tenure, high-value customer flag, new customer flag. One-hot encoding for categoricals. Standard scaling for model input
- **Model Training & Comparison** — Trains and evaluates three classifiers side-by-side:
  - Logistic Regression
  - Random Forest
  - XGBoost
- **Evaluation** — 5-fold cross-validation, ROC-AUC curves, confusion matrix, precision/recall, feature importance ranking
- **Business Output** — Identifies top churn drivers and recommends targeted retention campaigns

### Results

```
══════════════════════════════════════════════════════════════
  MODEL COMPARISON
══════════════════════════════════════════════════════════════
  Logistic Regression       | Accuracy: 0.821 | F1: 0.742 | AUC: 0.878
  Random Forest             | Accuracy: 0.856 | F1: 0.791 | AUC: 0.912
  XGBoost                   | Accuracy: 0.871 | F1: 0.813 | AUC: 0.923

  Best model: XGBoost (AUC: 0.923)
══════════════════════════════════════════════════════════════
```

### Top Churn Drivers
1. Contract type (month-to-month = highest risk)
2. Tenure (new customers churn most)
3. Monthly charges (higher spend = higher churn)
4. Support ticket frequency
5. Payment method (electronic check = highest risk)

### Visualizations Generated
- `churn_by_segment.png` — Churn rate bar charts by contract, service, payment, partner status
- `churn_distributions.png` — Histogram comparison of churned vs retained customers
- `model_comparison.png` — ROC curves + confusion matrix
- `feature_importance.png` — Top 15 features ranked by importance

### Tech Stack
`pandas` · `scikit-learn` · `XGBoost` · `SciPy` · `Matplotlib` · `Seaborn`

---

## 04 — Time Series Forecasting — Demand & Capacity Analysis

[`04_time_series_forecasting/capacity_forecasting.py`](./04_time_series_forecasting/capacity_forecasting.py)

Analytical forecasting pipeline for infrastructure capacity planning using 3 years of simulated operational data (1,095 daily observations).

### What It Does

- **Trend Decomposition** — Separates observed data into trend, weekly/monthly seasonality, and residual components using additive decomposition
- **Stationarity Testing** — Augmented Dickey-Fuller (ADF) test with rolling mean and rolling standard deviation visualization to assess whether differencing is needed
- **Anomaly Detection** — Rolling mean ± 2.5σ band method to flag unusual capacity spikes or drops before they impact SLA
- **ARIMA Forecasting** — Fits ARIMA(2,1,2) model, evaluates on a 30-day holdout (MAE, RMSE), then refits on full data to produce 90-day forward projections with 95% confidence intervals

### Business Application
- 90-day demand projections used for quarterly infrastructure procurement planning
- Anomaly detection catches capacity spikes before they cause SLA breaches
- Seasonality analysis informs staffing and resource allocation decisions

### Visualizations Generated
- `decomposition.png` — Observed, trend, seasonal, and residual components
- `stationarity.png` — Rolling mean and standard deviation over time
- `anomalies.png` — Time series with anomaly points highlighted
- `forecast.png` — Historical data + 90-day forecast with confidence bands

### Sample Output

```
══════════════════════════════════════════════════
  ARIMA(2, 1, 2) FORECAST
══════════════════════════════════════════════════
  Holdout MAE:  2.34
  Holdout RMSE: 3.01
  AIC:          5842.17
  Forecast range: 73.2% — 82.6%
══════════════════════════════════════════════════
```

### Tech Stack
`pandas` · `statsmodels` · `SciPy` · `Matplotlib` · `Seaborn`

---

## 05 — SQL Analytics Toolkit

[`05_sql_analytics/analytical_queries.sql`](./05_sql_analytics/analytical_queries.sql)

Production analytical SQL queries for operational KPI reporting, SLA tracking, and ad-hoc investigation. Compatible with PostgreSQL, Amazon Redshift, and Snowflake.

### Queries Included

| # | Query | Techniques Used |
|---|-------|----------------|
| 1 | **SLA Compliance Dashboard** | Rolling 30-day breach rate, `LAG` for WoW change, conditional aggregation |
| 2 | **Resolution Time Percentile Analysis** | `NTILE(4)`, `PERCENTILE_CONT` (P50, P90, P95), partitioned stats by priority |
| 3 | **MoM & YoY Revenue Growth** | `LAG` with 1-month and 12-month offsets, rolling 3-month average |
| 4 | **Customer Cohort Retention** | `FIRST_VALUE`, month-offset calculation, retention percentage by cohort |
| 5 | **Pareto Analysis (80/20 Rule)** | Cumulative `SUM`, `RANK`, conditional grouping to identify top incident categories |
| 6 | **Agent Performance Ranking** | Multi-metric `RANK`, `NTILE` quartile bucketing, minimum threshold filtering |
| 7 | **Volume Anomaly Detection** | Rolling `AVG` + `STDDEV`, conditional flagging for high/low anomalies |

### Example — SLA Compliance with Rolling Trend

```sql
WITH daily_sla AS (
    SELECT
        DATE(created_at) AS report_date,
        COUNT(*) AS total_tickets,
        SUM(CASE WHEN resolved_at > sla_deadline THEN 1 ELSE 0 END) AS breached_sla
    FROM incidents
    GROUP BY DATE(created_at)
),
rolling_metrics AS (
    SELECT
        report_date,
        breached_sla,
        ROUND(breached_sla::DECIMAL / NULLIF(total_tickets, 0) * 100, 2) AS daily_breach_pct,
        AVG(breached_sla::DECIMAL / NULLIF(total_tickets, 0) * 100)
            OVER (ORDER BY report_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)
            AS rolling_30d_breach_pct
    FROM daily_sla
)
SELECT * FROM rolling_metrics ORDER BY report_date DESC;
```

### Compatible With
`PostgreSQL` · `Amazon Redshift` · `Snowflake`

---

## 06 — AWS Serverless Data Pipeline

[`06_aws_data_pipeline/aws_etl_pipeline.py`](./06_aws_data_pipeline/aws_etl_pipeline.py)

Production-ready data pipeline on AWS for analytical data processing, quality monitoring, and Redshift loading.

### Architecture

```
┌──────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────┐     ┌────────────┐
│ S3 Upload│────▶│ Lambda Trigger│────▶│  Glue ETL    │────▶│ Redshift │────▶│ CloudWatch │
└──────────┘     └──────────────┘     └──────────────┘     └──────────┘     └────────────┘
                  File validation      Schema validation     COPY + Upsert   Quality metrics
                  Job kickoff          Cleaning & transform                  Alerting
                                       Quality scoring
```

### Components

| Component | What It Does |
|-----------|-------------|
| **Lambda Handler** | Listens for S3 PutObject events, validates file type and size, starts Glue job, publishes CloudWatch metrics |
| **GlueETLPipeline Class** | Full ETL pipeline: `ingest()` → `validate_schema()` → `clean_and_transform()` → `compute_quality_score()` → `write_to_s3()` |
| **Redshift Loader** | Generates `COPY` commands for bulk loading and `UPSERT` patterns using staging tables |
| **CloudWatch Metrics** | Publishes custom metrics: QualityScore, RowsProcessed, FileSizeBytes, ProcessingErrors |
| **Quality Reporter** | Writes JSON quality report per run to S3 with completeness, uniqueness, and action audit trail |

### Features
- Row-hash deduplication using SHA-256 on business key columns
- Automated data quality scoring (0–100) per pipeline run
- Schema validation with automatic type coercion
- Quality report JSON persisted to S3 for historical tracking
- Local demo mode included — runs without AWS credentials

### Usage

```python
# On AWS (production)
pipeline = GlueETLPipeline(source_bucket="analytics-data", source_key="raw/sales.csv")
clean_df, report = pipeline.run(expected_schema={
    "transaction_id": "int",
    "amount": "float",
    "transaction_date": "datetime",
    "category": "str",
})

# Local demo (no AWS needed)
python aws_etl_pipeline.py
```

### Sample Output

```
  Raw data: 10,200 rows (includes 200 duplicates, ~500 nulls)
  After dedup: 10,000 rows (200 removed)

  Quality Score: 96.4/100
  Completeness:  100.0%
  Uniqueness:    100.0%
  Final rows:    10,000
```

### Tech Stack
`boto3` · `pandas` · `AWS S3` · `AWS Glue` · `AWS Lambda` · `AWS Redshift` · `CloudWatch`

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/thrikshagiriraju/data-analytics-portfolio.git
cd data-analytics-portfolio

# Install dependencies
pip install -r requirements.txt

# Run any project
cd 01_eda_statistical_analysis
python eda_analysis.py
```

> All projects include sample data generators — no external datasets or AWS credentials needed to run demos.

## 📫 Contact

- **Email**: thrikshagiriraju@gmail.com
- **LinkedIn**: [linkedin.com/in/thrikshagiriraju](https://linkedin.com/in/thrikshagiriraju)
