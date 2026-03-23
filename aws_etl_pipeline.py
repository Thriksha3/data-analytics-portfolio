"""
AWS Serverless Data Pipeline — ETL & Quality Monitoring
=========================================================
Production-ready AWS data pipeline components:
- S3 event-triggered Lambda handler
- Glue ETL job (schema validation, transformation, quality scoring)
- Redshift data loading with upsert logic
- CloudWatch metrics & alerting
- Data quality scoring and reporting

Architecture:
  S3 Upload → Lambda Trigger → Glue ETL → Redshift → CloudWatch Metrics
"""

import boto3
import pandas as pd
import numpy as np
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ─────────────────────────────────────────────
# 1. LAMBDA HANDLER — S3 Event Trigger
# ─────────────────────────────────────────────
def lambda_handler(event, context):
    """
    Triggered by S3 PutObject events.
    Validates the uploaded file and starts the Glue ETL job.
    """
    s3_client = boto3.client("s3")
    glue_client = boto3.client("glue")

    try:
        # Extract S3 event details
        record = event["Records"][0]
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]
        file_size = record["s3"]["object"]["size"]

        logger.info(f"Processing: s3://{bucket}/{key} ({file_size:,} bytes)")

        # Validate file
        if not key.endswith((".csv", ".xlsx", ".parquet")):
            logger.warning(f"Unsupported file type: {key}")
            return {"statusCode": 400, "body": "Unsupported file type"}

        if file_size == 0:
            logger.warning(f"Empty file: {key}")
            return {"statusCode": 400, "body": "Empty file"}

        # Start Glue ETL job
        response = glue_client.start_job_run(
            JobName="analytics-etl-pipeline",
            Arguments={
                "--source_bucket": bucket,
                "--source_key": key,
                "--processing_timestamp": datetime.utcnow().isoformat(),
            }
        )

        job_run_id = response["JobRunId"]
        logger.info(f"Started Glue job: {job_run_id}")

        # Log to CloudWatch
        publish_metric("FilesProcessed", 1, "Count")
        publish_metric("FileSizeBytes", file_size, "Bytes")

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "ETL job started",
                "job_run_id": job_run_id,
                "source": f"s3://{bucket}/{key}",
            })
        }

    except Exception as e:
        logger.error(f"Lambda error: {str(e)}")
        publish_metric("ProcessingErrors", 1, "Count")
        raise


# ─────────────────────────────────────────────
# 2. GLUE ETL JOB — Transform & Validate
# ─────────────────────────────────────────────
class GlueETLPipeline:
    """
    AWS Glue ETL pipeline for analytical data processing.
    Handles ingestion, validation, transformation, and loading.
    """

    def __init__(self, source_bucket: str, source_key: str):
        self.s3 = boto3.client("s3")
        self.source_bucket = source_bucket
        self.source_key = source_key
        self.quality_report = {"timestamp": datetime.utcnow().isoformat(), "checks": []}

    def ingest(self) -> pd.DataFrame:
        """Read source data from S3."""
        s3_path = f"s3://{self.source_bucket}/{self.source_key}"
        logger.info(f"Ingesting: {s3_path}")

        if self.source_key.endswith(".csv"):
            obj = self.s3.get_object(Bucket=self.source_bucket, Key=self.source_key)
            df = pd.read_csv(obj["Body"])
        elif self.source_key.endswith(".parquet"):
            df = pd.read_parquet(s3_path)
        elif self.source_key.endswith(".xlsx"):
            obj = self.s3.get_object(Bucket=self.source_bucket, Key=self.source_key)
            df = pd.read_excel(obj["Body"])
        else:
            raise ValueError(f"Unsupported format: {self.source_key}")

        logger.info(f"Ingested {len(df):,} rows × {len(df.columns)} columns")
        self.quality_report["source_rows"] = len(df)
        return df

    def validate_schema(self, df: pd.DataFrame, expected: Dict[str, str]) -> pd.DataFrame:
        """Validate and enforce column types."""
        errors = []

        # Check required columns
        missing = set(expected.keys()) - set(df.columns)
        if missing:
            errors.append(f"Missing columns: {missing}")

        # Coerce types
        type_map = {"int": "int64", "float": "float64", "str": "object", "datetime": "datetime64[ns]"}
        for col, dtype in expected.items():
            if col not in df.columns:
                continue
            try:
                if dtype == "datetime":
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                elif dtype in ("int", "float"):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception as e:
                errors.append(f"Type coercion failed for {col}: {e}")

        self.quality_report["checks"].append({
            "check": "schema_validation",
            "passed": len(errors) == 0,
            "errors": errors,
        })

        return df

    def clean_and_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning rules and business transformations."""
        initial_rows = len(df)

        # Remove exact duplicates
        df = df.drop_duplicates()
        dupes_removed = initial_rows - len(df)

        # Standardize strings
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].str.strip().str.lower()

        # Handle nulls in critical columns
        null_counts = df.isnull().sum()
        critical_nulls = null_counts[null_counts > 0]

        # Generate row hash for deduplication tracking
        hash_cols = df.columns[:4].tolist()  # first 4 columns as business keys
        df["row_hash"] = df[hash_cols].astype(str).apply(
            lambda row: hashlib.sha256("|||".join(row).encode()).hexdigest()[:16], axis=1
        )

        # Add metadata
        df["_loaded_at"] = datetime.utcnow()
        df["_source_file"] = self.source_key

        self.quality_report["checks"].append({
            "check": "cleaning",
            "duplicates_removed": dupes_removed,
            "null_summary": critical_nulls.to_dict(),
            "output_rows": len(df),
        })

        logger.info(f"Cleaned: {dupes_removed} duplicates removed, {len(df):,} rows remaining")
        return df

    def compute_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate 0-100 data quality score."""
        n_rows, n_cols = df.shape
        total_cells = n_rows * n_cols

        completeness = (1 - df.isnull().sum().sum() / total_cells) * 100
        uniqueness = (len(df.drop_duplicates()) / n_rows) * 100

        score = round(completeness * 0.6 + uniqueness * 0.4, 1)

        self.quality_report["quality_score"] = score
        self.quality_report["completeness"] = round(completeness, 1)
        self.quality_report["uniqueness"] = round(uniqueness, 1)

        logger.info(f"Quality score: {score}/100 (completeness={completeness:.1f}%, uniqueness={uniqueness:.1f}%)")
        return score

    def write_to_s3(self, df: pd.DataFrame, zone: str = "curated") -> str:
        """Write transformed data to curated S3 zone."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_key = f"{zone}/processed_{timestamp}.parquet"

        # Write parquet to S3
        buffer = df.to_parquet(index=False)
        self.s3.put_object(
            Bucket=self.source_bucket,
            Key=output_key,
            Body=buffer,
        )

        # Write quality report
        report_key = f"quality_reports/report_{timestamp}.json"
        self.s3.put_object(
            Bucket=self.source_bucket,
            Key=report_key,
            Body=json.dumps(self.quality_report, indent=2, default=str),
        )

        logger.info(f"Written to s3://{self.source_bucket}/{output_key}")
        return output_key

    def run(self, expected_schema: Dict[str, str]) -> Tuple[pd.DataFrame, Dict]:
        """Execute full ETL pipeline."""
        logger.info("Starting ETL pipeline...")

        df = self.ingest()
        df = self.validate_schema(df, expected_schema)
        df = self.clean_and_transform(df)
        score = self.compute_quality_score(df)
        output_path = self.write_to_s3(df)

        self.quality_report["output_path"] = output_path
        self.quality_report["final_rows"] = len(df)

        # Publish CloudWatch metrics
        publish_metric("QualityScore", score, "None")
        publish_metric("RowsProcessed", len(df), "Count")

        logger.info(f"ETL complete: {len(df):,} rows, quality={score}/100")
        return df, self.quality_report


# ─────────────────────────────────────────────
# 3. REDSHIFT LOADER
# ─────────────────────────────────────────────
def generate_redshift_copy_command(s3_path: str, table: str, iam_role: str) -> str:
    """Generate Redshift COPY command for bulk loading from S3."""
    return f"""
    COPY {table}
    FROM '{s3_path}'
    IAM_ROLE '{iam_role}'
    FORMAT AS PARQUET
    ACCEPTINVCHARS
    TRUNCATECOLUMNS
    COMPUPDATE ON
    STATUPDATE ON;
    """


def generate_upsert_query(staging_table: str, target_table: str,
                           key_columns: List[str], update_columns: List[str]) -> str:
    """Generate Redshift MERGE/UPSERT pattern using staging table."""
    key_join = " AND ".join([f"t.{k} = s.{k}" for k in key_columns])
    update_set = ", ".join([f"{c} = s.{c}" for c in update_columns])
    all_cols = ", ".join(key_columns + update_columns)

    return f"""
    -- Delete matching rows from target
    DELETE FROM {target_table}
    USING {staging_table}
    WHERE {key_join};

    -- Insert all from staging
    INSERT INTO {target_table} ({all_cols}, _loaded_at, _source_file)
    SELECT {all_cols}, _loaded_at, _source_file
    FROM {staging_table};

    -- Clean up staging
    DROP TABLE IF EXISTS {staging_table};
    """


# ─────────────────────────────────────────────
# 4. CLOUDWATCH METRICS
# ─────────────────────────────────────────────
def publish_metric(metric_name: str, value: float, unit: str,
                   namespace: str = "AnalyticsPipeline"):
    """Publish custom metric to CloudWatch for pipeline monitoring."""
    try:
        cw = boto3.client("cloudwatch")
        cw.put_metric_data(
            Namespace=namespace,
            MetricData=[{
                "MetricName": metric_name,
                "Value": value,
                "Unit": unit,
                "Timestamp": datetime.utcnow(),
                "Dimensions": [
                    {"Name": "Environment", "Value": "production"},
                    {"Name": "Pipeline", "Value": "analytics-etl"},
                ],
            }]
        )
    except Exception as e:
        logger.warning(f"CloudWatch metric publish failed: {e}")


# ─────────────────────────────────────────────
# 5. LOCAL DEMO (runs without AWS credentials)
# ─────────────────────────────────────────────
def demo_local():
    """Demonstrate pipeline logic locally with sample data."""
    print(f"\n{'='*60}")
    print(f"  AWS DATA PIPELINE — LOCAL DEMO")
    print(f"{'='*60}\n")

    # Generate sample data
    np.random.seed(42)
    n = 10000
    df = pd.DataFrame({
        "transaction_id": range(1, n + 1),
        "customer_id": np.random.randint(1000, 9999, n),
        "transaction_date": pd.date_range("2024-01-01", periods=n, freq="h"),
        "amount": np.random.lognormal(3, 1, n).round(2),
        "category": np.random.choice(["Electronics", "Clothing", "Food", "Services"], n),
        "region": np.random.choice(["US-East", "US-West", "EU", "APAC"], n),
        "channel": np.random.choice(["Online", "Store", "Mobile"], n),
    })

    # Add some messiness
    df.loc[df.sample(500).index, "amount"] = np.nan  # 5% nulls
    df = pd.concat([df, df.sample(200)], ignore_index=True)  # duplicates

    print(f"  Raw data: {len(df):,} rows (includes {200} duplicates, ~500 nulls)")

    # Schema
    schema = {
        "transaction_id": "int",
        "customer_id": "int",
        "transaction_date": "datetime",
        "amount": "float",
        "category": "str",
        "region": "str",
    }

    # Run cleaning (reusing the cleaning logic without S3)
    initial = len(df)
    df = df.drop_duplicates()
    print(f"  After dedup: {len(df):,} rows ({initial - len(df)} removed)")

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip().str.lower()

    df["amount"] = df["amount"].fillna(df["amount"].median())

    # Quality score
    completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    uniqueness = (len(df.drop_duplicates()) / len(df)) * 100
    score = round(completeness * 0.6 + uniqueness * 0.4, 1)

    print(f"\n  Quality Score: {score}/100")
    print(f"  Completeness:  {completeness:.1f}%")
    print(f"  Uniqueness:    {uniqueness:.1f}%")
    print(f"  Final rows:    {len(df):,}")

    # Sample Redshift commands
    print(f"\n  --- Generated Redshift Commands ---")
    print(generate_redshift_copy_command(
        "s3://analytics-bucket/curated/processed_20240101.parquet",
        "analytics.transactions",
        "arn:aws:iam::123456789:role/RedshiftLoadRole"
    ))

    print(f"\n{'='*60}")
    print(f"  DEMO COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    demo_local()
