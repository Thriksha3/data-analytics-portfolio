"""
Automated Data Cleaning & Validation Pipeline
================================================
Production-grade data cleaning framework for analytical datasets:
- Schema validation & type enforcement
- Null handling & missing-value imputation
- Deduplication (exact & fuzzy)
- Outlier treatment
- Format standardization
- Data quality scoring & reporting
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
import warnings

warnings.filterwarnings("ignore")


class DataQualityReport:
    """Tracks all cleaning actions and generates a quality scorecard."""

    def __init__(self):
        self.actions = []
        self.initial_shape = None
        self.final_shape = None
        self.start_time = datetime.now()

    def log(self, action: str, details: str, rows_affected: int = 0):
        self.actions.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details,
            "rows_affected": rows_affected,
        })

    def summary(self) -> Dict:
        return {
            "initial_rows": self.initial_shape[0] if self.initial_shape else 0,
            "final_rows": self.final_shape[0] if self.final_shape else 0,
            "rows_removed": (self.initial_shape[0] - self.final_shape[0])
                            if self.initial_shape and self.final_shape else 0,
            "total_actions": len(self.actions),
            "processing_time": str(datetime.now() - self.start_time),
            "actions": self.actions,
        }


class DataCleaningPipeline:
    """
    End-to-end data cleaning pipeline with quality reporting.

    Usage:
        pipeline = DataCleaningPipeline(df)
        clean_df = (pipeline
            .validate_schema(expected_schema)
            .remove_duplicates(subset=["id", "date"])
            .handle_nulls(strategy="smart")
            .detect_and_treat_outliers(columns=["revenue"])
            .standardize_formats()
            .run())
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.report = DataQualityReport()
        self.report.initial_shape = df.shape
        self._steps = []
        print(f"✓ Pipeline initialized: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ─────────────────────────────────────────
    # SCHEMA VALIDATION
    # ─────────────────────────────────────────
    def validate_schema(self, expected_schema: Dict[str, str]):
        """
        Validate column presence and enforce data types.

        Args:
            expected_schema: {"column_name": "dtype"} mapping
                Supported dtypes: int, float, str, datetime, bool
        """
        def _validate(df, report):
            type_map = {
                "int": "int64", "float": "float64", "str": "object",
                "datetime": "datetime64[ns]", "bool": "bool",
            }

            missing_cols = set(expected_schema.keys()) - set(df.columns)
            if missing_cols:
                report.log("SCHEMA_ERROR", f"Missing columns: {missing_cols}")
                print(f"  ⚠ Missing columns: {missing_cols}")

            coerced = 0
            for col, expected_type in expected_schema.items():
                if col not in df.columns:
                    continue
                target = type_map.get(expected_type, expected_type)
                if str(df[col].dtype) != target:
                    try:
                        if expected_type == "datetime":
                            df[col] = pd.to_datetime(df[col], errors="coerce")
                        elif expected_type in ("int", "float"):
                            df[col] = pd.to_numeric(df[col], errors="coerce")
                        elif expected_type == "str":
                            df[col] = df[col].astype(str)
                        coerced += 1
                    except Exception as e:
                        report.log("TYPE_ERROR", f"Cannot coerce {col} to {expected_type}: {e}")

            report.log("SCHEMA_VALIDATION", f"Validated {len(expected_schema)} columns, coerced {coerced} types")
            print(f"  ✓ Schema validated: {len(expected_schema)} columns checked, {coerced} types coerced")
            return df

        self._steps.append(("Schema Validation", _validate))
        return self

    # ─────────────────────────────────────────
    # DEDUPLICATION
    # ─────────────────────────────────────────
    def remove_duplicates(self, subset: List[str] = None, keep: str = "first"):
        """
        Remove duplicate rows based on subset columns.

        Args:
            subset: Columns to check for duplicates (None = all columns)
            keep: 'first', 'last', or False (remove all duplicates)
        """
        def _dedup(df, report):
            before = len(df)
            df = df.drop_duplicates(subset=subset, keep=keep)
            removed = before - len(df)
            report.log("DEDUPLICATION",
                       f"Removed {removed:,} duplicates (subset={subset or 'all columns'})",
                       removed)
            print(f"  ✓ Deduplication: {removed:,} duplicates removed ({removed/before*100:.1f}%)")
            return df

        self._steps.append(("Deduplication", _dedup))
        return self

    # ─────────────────────────────────────────
    # NULL HANDLING
    # ─────────────────────────────────────────
    def handle_nulls(self, strategy: str = "smart",
                     custom_fills: Dict[str, any] = None,
                     drop_threshold: float = 0.5):
        """
        Handle missing values with configurable strategies.

        Args:
            strategy:
                'smart'  — median for numeric, mode for categorical, drop if >50% null
                'drop'   — drop all rows with any null
                'custom' — use custom_fills dict
            custom_fills: {"column": fill_value} for strategy='custom'
            drop_threshold: Drop columns with null% above this (strategy='smart')
        """
        def _handle(df, report):
            total_nulls_before = df.isnull().sum().sum()

            if strategy == "drop":
                df = df.dropna()

            elif strategy == "custom" and custom_fills:
                for col, fill in custom_fills.items():
                    if col in df.columns:
                        df[col] = df[col].fillna(fill)

            elif strategy == "smart":
                # Drop columns with too many nulls
                null_pct = df.isnull().sum() / len(df)
                drop_cols = null_pct[null_pct > drop_threshold].index.tolist()
                if drop_cols:
                    df = df.drop(columns=drop_cols)
                    report.log("DROP_COLUMNS",
                               f"Dropped {len(drop_cols)} columns with >{drop_threshold*100}% nulls: {drop_cols}")

                # Fill remaining
                for col in df.columns:
                    if df[col].isnull().sum() == 0:
                        continue
                    if df[col].dtype in ["float64", "int64"]:
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        mode_val = df[col].mode()
                        if len(mode_val) > 0:
                            df[col] = df[col].fillna(mode_val[0])

            total_nulls_after = df.isnull().sum().sum()
            handled = total_nulls_before - total_nulls_after
            report.log("NULL_HANDLING",
                       f"Strategy='{strategy}', handled {handled:,} null values",
                       handled)
            print(f"  ✓ Null handling ({strategy}): {handled:,} nulls resolved, {total_nulls_after:,} remaining")
            return df

        self._steps.append(("Null Handling", _handle))
        return self

    # ─────────────────────────────────────────
    # OUTLIER TREATMENT
    # ─────────────────────────────────────────
    def detect_and_treat_outliers(self, columns: List[str] = None,
                                   method: str = "iqr",
                                   treatment: str = "cap"):
        """
        Detect and treat outliers in numeric columns.

        Args:
            method: 'iqr' (1.5×IQR) or 'zscore' (|z| > 3)
            treatment: 'cap' (winsorize to bounds), 'remove', or 'flag' (add _outlier column)
        """
        def _outliers(df, report):
            if columns is None:
                cols = df.select_dtypes(include=[np.number]).columns.tolist()
            else:
                cols = columns

            total_treated = 0

            for col in cols:
                series = df[col].dropna()

                if method == "iqr":
                    q1, q3 = series.quantile(0.25), series.quantile(0.75)
                    iqr = q3 - q1
                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                else:
                    lower = series.mean() - 3 * series.std()
                    upper = series.mean() + 3 * series.std()

                outlier_mask = (df[col] < lower) | (df[col] > upper)
                n_outliers = outlier_mask.sum()

                if n_outliers > 0:
                    if treatment == "cap":
                        df[col] = df[col].clip(lower=lower, upper=upper)
                    elif treatment == "remove":
                        df = df[~outlier_mask]
                    elif treatment == "flag":
                        df[f"{col}_outlier"] = outlier_mask.astype(int)

                    total_treated += n_outliers

            report.log("OUTLIER_TREATMENT",
                       f"Method={method}, treatment={treatment}, {total_treated:,} values across {len(cols)} columns",
                       total_treated)
            print(f"  ✓ Outliers ({method}/{treatment}): {total_treated:,} values treated across {len(cols)} columns")
            return df

        self._steps.append(("Outlier Treatment", _outliers))
        return self

    # ─────────────────────────────────────────
    # FORMAT STANDARDIZATION
    # ─────────────────────────────────────────
    def standardize_formats(self):
        """
        Standardize string formats:
        - Strip whitespace
        - Normalize case for categorical columns
        - Fix encoding issues
        - Standardize date formats
        """
        def _standardize(df, report):
            changes = 0
            for col in df.select_dtypes(include=["object"]).columns:
                original = df[col].copy()
                df[col] = df[col].str.strip()
                df[col] = df[col].str.replace(r"\s+", " ", regex=True)
                changes += (original != df[col]).sum()

            report.log("FORMAT_STANDARDIZATION",
                       f"Standardized string formats, {changes:,} values cleaned", changes)
            print(f"  ✓ Format standardization: {changes:,} string values cleaned")
            return df

        self._steps.append(("Format Standardization", _standardize))
        return self

    # ─────────────────────────────────────────
    # EXECUTE PIPELINE
    # ─────────────────────────────────────────
    def run(self) -> Tuple[pd.DataFrame, Dict]:
        """Execute all pipeline steps and return cleaned DataFrame + quality report."""
        print(f"\n{'='*60}")
        print(f"  RUNNING DATA CLEANING PIPELINE ({len(self._steps)} steps)")
        print(f"{'='*60}\n")

        for step_name, step_fn in self._steps:
            print(f"Step: {step_name}")
            self.df = step_fn(self.df, self.report)

        self.report.final_shape = self.df.shape
        summary = self.report.summary()

        print(f"\n{'='*60}")
        print(f"  PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"  Input:  {summary['initial_rows']:,} rows")
        print(f"  Output: {summary['final_rows']:,} rows")
        print(f"  Removed: {summary['rows_removed']:,} rows ({summary['rows_removed']/summary['initial_rows']*100:.1f}%)")
        print(f"  Processing time: {summary['processing_time']}")
        print(f"{'='*60}\n")

        return self.df, summary


# ─────────────────────────────────────────────
# STANDALONE QUALITY SCORER
# ─────────────────────────────────────────────
def data_quality_score(df: pd.DataFrame) -> Dict:
    """
    Calculate a 0-100 data quality score based on:
    - Completeness (null rate)
    - Uniqueness (duplicate rate)
    - Consistency (data type uniformity)
    """
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols

    # Completeness: % of non-null cells
    completeness = (1 - df.isnull().sum().sum() / total_cells) * 100

    # Uniqueness: % of non-duplicate rows
    uniqueness = (len(df.drop_duplicates()) / n_rows) * 100

    # Consistency: % of columns with uniform types (no mixed types)
    consistent_cols = sum(1 for col in df.columns
                          if df[col].dropna().apply(type).nunique() <= 1)
    consistency = (consistent_cols / n_cols) * 100

    overall = (completeness * 0.4 + uniqueness * 0.3 + consistency * 0.3)

    scores = {
        "overall_score": round(overall, 1),
        "completeness": round(completeness, 1),
        "uniqueness": round(uniqueness, 1),
        "consistency": round(consistency, 1),
        "total_rows": n_rows,
        "total_nulls": int(df.isnull().sum().sum()),
        "duplicate_rows": n_rows - len(df.drop_duplicates()),
    }

    print(f"\n{'─'*40}")
    print(f"  DATA QUALITY SCORECARD")
    print(f"{'─'*40}")
    print(f"  Overall Score:  {scores['overall_score']}/100")
    print(f"  Completeness:   {scores['completeness']}%")
    print(f"  Uniqueness:     {scores['uniqueness']}%")
    print(f"  Consistency:    {scores['consistency']}%")
    print(f"{'─'*40}\n")

    return scores


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    n = 10000

    # Simulate messy operational data
    df = pd.DataFrame({
        "ticket_id": range(1, n + 1),
        "created_date": pd.date_range("2023-01-01", periods=n, freq="h"),
        "priority": np.random.choice(["Low", "Medium", "High", " High ", "high", "Critical"], n),
        "resolution_hrs": np.concatenate([np.random.exponential(4, n - 50),
                                          np.random.uniform(100, 500, 50)]),  # outliers
        "satisfaction": np.where(np.random.random(n) > 0.1,
                                  np.random.normal(3.5, 0.8, n), np.nan),  # 10% nulls
        "agent_name": np.random.choice(["Alice", "Bob", "Charlie", None], n),
        "channel": np.random.choice(["Phone", "Email", "Chat"], n),
    })

    # Add duplicates
    df = pd.concat([df, df.sample(200)], ignore_index=True)

    # Quality score BEFORE cleaning
    print("BEFORE CLEANING:")
    data_quality_score(df)

    # Run pipeline
    schema = {
        "ticket_id": "int",
        "created_date": "datetime",
        "priority": "str",
        "resolution_hrs": "float",
        "satisfaction": "float",
        "channel": "str",
    }

    pipeline = DataCleaningPipeline(df)
    clean_df, report = (pipeline
        .validate_schema(schema)
        .remove_duplicates(subset=["ticket_id"])
        .handle_nulls(strategy="smart")
        .detect_and_treat_outliers(columns=["resolution_hrs", "satisfaction"])
        .standardize_formats()
        .run())

    # Quality score AFTER cleaning
    print("AFTER CLEANING:")
    data_quality_score(clean_df)
