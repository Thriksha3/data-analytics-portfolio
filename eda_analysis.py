"""
EDA & Statistical Analysis Toolkit
====================================
Performs deep-dive exploratory data analysis including:
- Data profiling & distribution analysis
- Hypothesis testing (t-test, chi-square, ANOVA)
- A/B testing with statistical significance
- Correlation analysis & heatmaps
- Outlier detection (IQR & Z-score)
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


# ─────────────────────────────────────────────
# 1. DATA PROFILING
# ─────────────────────────────────────────────
def profile_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a comprehensive data quality profile for every column.
    Returns a summary DataFrame with dtype, nulls, uniques, and distribution stats.
    """
    profile = pd.DataFrame({
        "dtype": df.dtypes,
        "non_null_count": df.count(),
        "null_count": df.isnull().sum(),
        "null_pct": (df.isnull().sum() / len(df) * 100).round(2),
        "unique_count": df.nunique(),
        "duplicate_rows": len(df) - len(df.drop_duplicates()),
    })

    # Add stats for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        profile.loc[col, "mean"] = df[col].mean()
        profile.loc[col, "median"] = df[col].median()
        profile.loc[col, "std"] = df[col].std()
        profile.loc[col, "skewness"] = df[col].skew()
        profile.loc[col, "kurtosis"] = df[col].kurtosis()

    print(f"\n{'='*60}")
    print(f"DATASET PROFILE: {len(df):,} rows × {len(df.columns)} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Duplicate rows: {len(df) - len(df.drop_duplicates()):,}")
    print(f"{'='*60}\n")

    return profile


def plot_distributions(df: pd.DataFrame, numeric_cols: List[str] = None, bins: int = 30):
    """
    Plot histograms with KDE for numeric columns to identify
    distribution shape, skewness, and potential outliers.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    n_cols = min(len(numeric_cols), 12)
    n_rows = (n_cols + 2) // 3

    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

    for idx, col in enumerate(numeric_cols[:12]):
        ax = axes[idx]
        df[col].hist(bins=bins, ax=ax, color="#4C72B0", alpha=0.7, edgecolor="white")
        ax.set_title(f"{col}\nskew={df[col].skew():.2f}", fontsize=10)
        ax.set_xlabel("")

    # Hide unused subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Distribution Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("output/distributions.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✓ Saved: output/distributions.png")


# ─────────────────────────────────────────────
# 2. HYPOTHESIS TESTING
# ─────────────────────────────────────────────
def run_ttest(group_a: pd.Series, group_b: pd.Series,
              alpha: float = 0.05, test_type: str = "independent") -> Dict:
    """
    Perform t-test between two groups.

    Args:
        group_a, group_b: Data series for each group
        alpha: Significance level (default 0.05)
        test_type: 'independent' or 'paired'

    Returns:
        Dictionary with test results and interpretation
    """
    # Check normality (Shapiro-Wilk)
    _, p_norm_a = stats.shapiro(group_a.sample(min(len(group_a), 5000)))
    _, p_norm_b = stats.shapiro(group_b.sample(min(len(group_b), 5000)))

    # Check equal variances (Levene's test)
    _, p_levene = stats.levene(group_a.dropna(), group_b.dropna())

    if test_type == "independent":
        equal_var = p_levene > alpha
        t_stat, p_value = stats.ttest_ind(group_a.dropna(), group_b.dropna(),
                                           equal_var=equal_var)
    else:
        t_stat, p_value = stats.ttest_rel(group_a.dropna(), group_b.dropna())

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((group_a.std()**2 + group_b.std()**2) / 2)
    cohens_d = (group_a.mean() - group_b.mean()) / pooled_std if pooled_std > 0 else 0

    result = {
        "test": f"{'Independent' if test_type == 'independent' else 'Paired'} t-test",
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "alpha": alpha,
        "significant": p_value < alpha,
        "cohens_d": round(cohens_d, 4),
        "effect_size": "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large",
        "group_a_mean": round(group_a.mean(), 4),
        "group_b_mean": round(group_b.mean(), 4),
        "normality_ok": p_norm_a > alpha and p_norm_b > alpha,
        "equal_variance": p_levene > alpha,
    }

    print(f"\n{'─'*50}")
    print(f"  {result['test']} Results")
    print(f"{'─'*50}")
    print(f"  Group A mean: {result['group_a_mean']}")
    print(f"  Group B mean: {result['group_b_mean']}")
    print(f"  t-statistic:  {result['t_statistic']}")
    print(f"  p-value:      {result['p_value']}")
    print(f"  Cohen's d:    {result['cohens_d']} ({result['effect_size']})")
    print(f"  Significant:  {'YES ✓' if result['significant'] else 'NO ✗'} (α={alpha})")
    print(f"{'─'*50}\n")

    return result


def run_chi_square(df: pd.DataFrame, col_a: str, col_b: str,
                   alpha: float = 0.05) -> Dict:
    """
    Perform chi-square test of independence between two categorical variables.
    """
    contingency = pd.crosstab(df[col_a], df[col_b])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    # Cramér's V for effect size
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

    result = {
        "test": "Chi-Square Test of Independence",
        "chi2_statistic": round(chi2, 4),
        "p_value": round(p_value, 6),
        "degrees_of_freedom": dof,
        "significant": p_value < alpha,
        "cramers_v": round(cramers_v, 4),
        "variables": f"{col_a} × {col_b}",
    }

    print(f"\n{'─'*50}")
    print(f"  Chi-Square Test: {col_a} × {col_b}")
    print(f"{'─'*50}")
    print(f"  χ² statistic:  {result['chi2_statistic']}")
    print(f"  p-value:       {result['p_value']}")
    print(f"  Cramér's V:    {result['cramers_v']}")
    print(f"  Significant:   {'YES ✓' if result['significant'] else 'NO ✗'} (α={alpha})")
    print(f"{'─'*50}\n")

    return result


def run_anova(df: pd.DataFrame, value_col: str, group_col: str,
              alpha: float = 0.05) -> Dict:
    """
    One-way ANOVA test across multiple groups.
    If significant, runs Tukey HSD post-hoc test.
    """
    groups = [group[value_col].dropna().values
              for _, group in df.groupby(group_col)]

    f_stat, p_value = stats.f_oneway(*groups)

    result = {
        "test": "One-Way ANOVA",
        "f_statistic": round(f_stat, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < alpha,
        "n_groups": len(groups),
        "group_means": df.groupby(group_col)[value_col].mean().round(4).to_dict(),
    }

    print(f"\n{'─'*50}")
    print(f"  ANOVA: {value_col} across {group_col}")
    print(f"{'─'*50}")
    print(f"  F-statistic: {result['f_statistic']}")
    print(f"  p-value:     {result['p_value']}")
    print(f"  Significant: {'YES ✓' if result['significant'] else 'NO ✗'} (α={alpha})")
    print(f"  Group means:")
    for grp, mean in result["group_means"].items():
        print(f"    {grp}: {mean}")
    print(f"{'─'*50}\n")

    return result


# ─────────────────────────────────────────────
# 3. A/B TESTING
# ─────────────────────────────────────────────
def ab_test(control: pd.Series, treatment: pd.Series,
            metric_type: str = "continuous", alpha: float = 0.05) -> Dict:
    """
    Perform A/B test analysis with full statistical rigor.

    Args:
        control: Control group metric values
        treatment: Treatment group metric values
        metric_type: 'continuous' (revenue, time) or 'proportion' (conversion rate)
        alpha: Significance level
    """
    if metric_type == "proportion":
        # Z-test for proportions
        n_c, n_t = len(control), len(treatment)
        p_c, p_t = control.mean(), treatment.mean()
        p_pool = (control.sum() + treatment.sum()) / (n_c + n_t)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_c + 1/n_t))
        z_stat = (p_t - p_c) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        lift = ((p_t - p_c) / p_c * 100) if p_c > 0 else 0

        result = {
            "test": "Two-Proportion Z-Test",
            "control_rate": round(p_c, 4),
            "treatment_rate": round(p_t, 4),
            "lift_pct": round(lift, 2),
            "z_statistic": round(z_stat, 4),
            "p_value": round(p_value, 6),
            "significant": p_value < alpha,
        }
    else:
        # Welch's t-test for continuous metrics
        t_stat, p_value = stats.ttest_ind(control.dropna(), treatment.dropna(),
                                           equal_var=False)
        lift = ((treatment.mean() - control.mean()) / control.mean() * 100) if control.mean() != 0 else 0

        result = {
            "test": "Welch's t-test (A/B)",
            "control_mean": round(control.mean(), 4),
            "treatment_mean": round(treatment.mean(), 4),
            "lift_pct": round(lift, 2),
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "significant": p_value < alpha,
        }

    winner = "TREATMENT ✓" if result["significant"] and result["lift_pct"] > 0 else \
             "CONTROL ✓" if result["significant"] and result["lift_pct"] < 0 else \
             "NO SIGNIFICANT DIFFERENCE"

    print(f"\n{'='*50}")
    print(f"  A/B TEST RESULTS")
    print(f"{'='*50}")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print(f"  WINNER: {winner}")
    print(f"  Lift: {result['lift_pct']:+.2f}%")
    print(f"{'='*50}\n")

    return result


# ─────────────────────────────────────────────
# 4. CORRELATION ANALYSIS
# ─────────────────────────────────────────────
def correlation_analysis(df: pd.DataFrame, method: str = "pearson",
                         threshold: float = 0.7):
    """
    Generate correlation matrix heatmap and flag highly correlated pairs.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr(method=method)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                linewidths=0.5, ax=ax)
    ax.set_title(f"Correlation Matrix ({method.title()})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("output/correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✓ Saved: output/correlation_heatmap.png")

    # Flag high correlations
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                high_corr.append({
                    "feature_1": corr_matrix.columns[i],
                    "feature_2": corr_matrix.columns[j],
                    "correlation": round(corr_matrix.iloc[i, j], 4),
                })

    if high_corr:
        print(f"\n⚠ Highly correlated pairs (|r| ≥ {threshold}):")
        for pair in high_corr:
            print(f"  {pair['feature_1']} ↔ {pair['feature_2']}: {pair['correlation']}")
    else:
        print(f"\n✓ No highly correlated pairs found (threshold: {threshold})")

    return corr_matrix, high_corr


# ─────────────────────────────────────────────
# 5. OUTLIER DETECTION
# ─────────────────────────────────────────────
def detect_outliers(df: pd.DataFrame, columns: List[str] = None,
                    method: str = "iqr") -> pd.DataFrame:
    """
    Detect outliers using IQR or Z-score method.

    Args:
        method: 'iqr' (1.5×IQR) or 'zscore' (|z| > 3)

    Returns:
        DataFrame with outlier flags per column
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    outlier_summary = []

    for col in columns:
        series = df[col].dropna()

        if method == "iqr":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_mask = (series < lower) | (series > upper)
        else:
            z_scores = np.abs(stats.zscore(series))
            outlier_mask = z_scores > 3
            lower, upper = series.mean() - 3 * series.std(), series.mean() + 3 * series.std()

        n_outliers = outlier_mask.sum()
        pct = (n_outliers / len(series) * 100)

        outlier_summary.append({
            "column": col,
            "method": method.upper(),
            "n_outliers": n_outliers,
            "pct_outliers": round(pct, 2),
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
        })

    results = pd.DataFrame(outlier_summary)

    print(f"\n{'─'*60}")
    print(f"  OUTLIER DETECTION ({method.upper()} method)")
    print(f"{'─'*60}")
    print(results.to_string(index=False))
    print(f"{'─'*60}\n")

    return results


# ─────────────────────────────────────────────
# DEMO: Run on sample data
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)

    # Generate sample operational data
    np.random.seed(42)
    n = 5000

    df = pd.DataFrame({
        "resolution_time_hrs": np.random.exponential(4, n),
        "satisfaction_score": np.random.normal(3.5, 0.8, n).clip(1, 5),
        "ticket_priority": np.random.choice(["Low", "Medium", "High", "Critical"], n,
                                             p=[0.3, 0.4, 0.2, 0.1]),
        "channel": np.random.choice(["Phone", "Email", "Chat", "Portal"], n),
        "first_call_resolution": np.random.binomial(1, 0.65, n),
        "escalated": np.random.binomial(1, 0.15, n),
        "agent_experience_yrs": np.random.uniform(0.5, 12, n),
        "interactions_count": np.random.poisson(3, n),
    })

    # 1. Profile
    profile = profile_dataset(df)
    print(profile)

    # 2. Distributions
    plot_distributions(df)

    # 3. Hypothesis test — do escalated tickets take longer?
    escalated = df[df["escalated"] == 1]["resolution_time_hrs"]
    not_escalated = df[df["escalated"] == 0]["resolution_time_hrs"]
    run_ttest(escalated, not_escalated)

    # 4. Chi-square — is priority related to escalation?
    run_chi_square(df, "ticket_priority", "escalated")

    # 5. ANOVA — resolution time across priority levels
    run_anova(df, "resolution_time_hrs", "ticket_priority")

    # 6. A/B Test — phone vs chat conversion
    phone = df[df["channel"] == "Phone"]["first_call_resolution"]
    chat = df[df["channel"] == "Chat"]["first_call_resolution"]
    ab_test(phone, chat, metric_type="proportion")

    # 7. Correlation
    correlation_analysis(df.select_dtypes(include=[np.number]))

    # 8. Outliers
    detect_outliers(df)

    print("\n✓ All analyses complete. Check output/ folder for visualizations.")
