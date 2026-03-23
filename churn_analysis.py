"""
Customer Churn Analysis & Predictive Modeling
================================================
End-to-end ML pipeline for churn prediction:
- EDA & statistical profiling
- Feature engineering (encoding, scaling, interaction features)
- Model training & comparison (Logistic Regression, Random Forest, XGBoost)
- Hyperparameter tuning (GridSearchCV)
- Model evaluation (confusion matrix, ROC-AUC, classification report)
- Feature importance analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, precision_recall_curve,
                              f1_score, accuracy_score)
import warnings
import os

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠ XGBoost not installed. Install with: pip install xgboost")


# ─────────────────────────────────────────────
# 1. DATA GENERATION (Simulated Telecom Dataset)
# ─────────────────────────────────────────────
def generate_telecom_data(n: int = 50000) -> pd.DataFrame:
    """Generate realistic telecom customer dataset for churn analysis."""
    np.random.seed(42)

    tenure = np.random.exponential(24, n).clip(1, 72).astype(int)
    monthly_charges = np.random.normal(65, 25, n).clip(20, 120)
    total_charges = monthly_charges * tenure + np.random.normal(0, 50, n)

    # Churn probability depends on tenure, charges, contract type
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.5, 0.3, 0.2])
    contract_risk = np.where(contract == "Month-to-month", 0.3,
                    np.where(contract == "One year", 0.1, 0.05))
    tenure_risk = np.clip(1 - tenure / 72, 0, 1) * 0.2
    charge_risk = np.clip((monthly_charges - 50) / 70, 0, 1) * 0.15
    churn_prob = np.clip(contract_risk + tenure_risk + charge_risk + np.random.normal(0, 0.05, n), 0.02, 0.95)
    churn = np.random.binomial(1, churn_prob)

    df = pd.DataFrame({
        "customer_id": [f"CUST_{i:05d}" for i in range(n)],
        "gender": np.random.choice(["Male", "Female"], n),
        "senior_citizen": np.random.binomial(1, 0.16, n),
        "partner": np.random.choice(["Yes", "No"], n),
        "dependents": np.random.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "tenure_months": tenure,
        "phone_service": np.random.choice(["Yes", "No"], n, p=[0.9, 0.1]),
        "internet_service": np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.2]),
        "contract": contract,
        "payment_method": np.random.choice(
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n),
        "monthly_charges": monthly_charges.round(2),
        "total_charges": total_charges.clip(0).round(2),
        "num_support_tickets": np.random.poisson(2, n),
        "num_referrals": np.random.poisson(0.5, n),
        "churn": churn,
    })

    return df


# ─────────────────────────────────────────────
# 2. EDA & STATISTICAL PROFILING
# ─────────────────────────────────────────────
def churn_eda(df: pd.DataFrame):
    """Perform targeted EDA for churn analysis."""
    os.makedirs("output", exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  CHURN ANALYSIS — EDA")
    print(f"{'='*50}")
    print(f"  Total customers: {len(df):,}")
    print(f"  Churn rate: {df['churn'].mean()*100:.1f}%")
    print(f"  Churned: {df['churn'].sum():,} | Retained: {(1-df['churn']).sum():,}")

    # Churn rate by key segments
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, col in zip(axes.flatten(), ["contract", "internet_service", "payment_method", "partner"]):
        churn_rates = df.groupby(col)["churn"].mean().sort_values(ascending=False)
        churn_rates.plot(kind="bar", ax=ax, color="#E74C3C", alpha=0.8, edgecolor="white")
        ax.set_title(f"Churn Rate by {col}", fontweight="bold")
        ax.set_ylabel("Churn Rate")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)
        for i, v in enumerate(churn_rates):
            ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("output/churn_by_segment.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Distribution comparison: churned vs retained
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col in zip(axes, ["tenure_months", "monthly_charges", "num_support_tickets"]):
        df[df["churn"] == 0][col].hist(ax=ax, bins=30, alpha=0.6, label="Retained", color="#2ECC71")
        df[df["churn"] == 1][col].hist(ax=ax, bins=30, alpha=0.6, label="Churned", color="#E74C3C")
        ax.set_title(col, fontweight="bold")
        ax.legend()

    plt.tight_layout()
    plt.savefig("output/churn_distributions.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Statistical test: is tenure significantly different?
    churned_tenure = df[df["churn"] == 1]["tenure_months"]
    retained_tenure = df[df["churn"] == 0]["tenure_months"]
    t_stat, p_val = stats.ttest_ind(churned_tenure, retained_tenure)
    print(f"\n  t-test (tenure: churned vs retained):")
    print(f"    Churned mean: {churned_tenure.mean():.1f} months")
    print(f"    Retained mean: {retained_tenure.mean():.1f} months")
    print(f"    p-value: {p_val:.6f} {'(significant)' if p_val < 0.05 else '(not significant)'}")

    print(f"\n  ✓ EDA plots saved to output/")


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create analytical features for churn prediction.
    """
    df = df.copy()

    # Interaction features
    df["charge_per_month_tenure"] = df["total_charges"] / df["tenure_months"].clip(1)
    df["tickets_per_tenure"] = df["num_support_tickets"] / df["tenure_months"].clip(1)
    df["high_value_customer"] = (df["monthly_charges"] > df["monthly_charges"].quantile(0.75)).astype(int)
    df["new_customer"] = (df["tenure_months"] <= 6).astype(int)
    df["long_tenure"] = (df["tenure_months"] >= 48).astype(int)

    # Encode categoricals
    label_cols = ["gender", "partner", "dependents", "phone_service",
                  "internet_service", "contract", "payment_method"]
    df_encoded = pd.get_dummies(df, columns=label_cols, drop_first=True)

    # Drop non-feature columns
    df_encoded = df_encoded.drop(columns=["customer_id"], errors="ignore")

    print(f"  ✓ Feature engineering: {df_encoded.shape[1]} features created")
    return df_encoded


# ─────────────────────────────────────────────
# 4. MODEL TRAINING & COMPARISON
# ─────────────────────────────────────────────
def train_and_compare(df: pd.DataFrame) -> dict:
    """
    Train multiple classifiers and compare performance.
    Returns the best model and evaluation metrics.
    """
    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models to compare
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(n_estimators=100, random_state=42,
                                           use_label_encoder=False, eval_metric="logloss")

    results = {}

    print(f"\n{'='*60}")
    print(f"  MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"  Train set: {len(X_train):,} | Test set: {len(X_test):,}")
    print(f"  Churn rate (train): {y_train.mean():.1%} | (test): {y_test.mean():.1%}\n")

    for name, model in models.items():
        # Use scaled data for logistic regression, raw for tree-based
        X_tr = X_train_scaled if "Logistic" in name else X_train
        X_te = X_test_scaled if "Logistic" in name else X_test

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        # Cross-validation
        cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring="roc_auc")

        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

        print(f"  {name:25s} | Accuracy: {accuracy:.3f} | F1: {f1:.3f} | "
              f"AUC: {roc_auc:.3f} | CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Best model
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\n  🏆 Best model: {best_name} (AUC: {results[best_name]['roc_auc']:.3f})")

    # Plot ROC curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})", linewidth=2)

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_title("ROC Curves", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()

    # Confusion matrix for best model
    cm = confusion_matrix(y_test, results[best_name]["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                xticklabels=["Retained", "Churned"],
                yticklabels=["Retained", "Churned"])
    axes[1].set_title(f"Confusion Matrix — {best_name}", fontweight="bold")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig("output/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Feature importance (best tree-based model)
    if hasattr(results[best_name]["model"], "feature_importances_"):
        importances = pd.Series(
            results[best_name]["model"].feature_importances_,
            index=X.columns
        ).sort_values(ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(10, 6))
        importances.plot(kind="barh", ax=ax, color="#3498DB", edgecolor="white")
        ax.set_title(f"Top 15 Features — {best_name}", fontweight="bold")
        ax.set_xlabel("Importance")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig("output/feature_importance.png", dpi=150, bbox_inches="tight")
        plt.show()

    # Classification report
    print(f"\n  Classification Report — {best_name}:")
    print(classification_report(y_test, results[best_name]["y_pred"],
                                 target_names=["Retained", "Churned"]))

    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    # Generate data
    print("Generating telecom dataset...")
    df = generate_telecom_data(50000)

    # EDA
    churn_eda(df)

    # Feature engineering
    df_features = engineer_features(df)

    # Train & compare
    results = train_and_compare(df_features)

    print("\n✓ All outputs saved to output/ folder.")
