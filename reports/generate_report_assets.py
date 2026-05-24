from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"
FIGURES = REPORTS / "figures"


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TABLES / f"{name}.csv", index=False)
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = []
    for row in df.itertuples(index=False):
        rows.append("| " + " | ".join(map(str, row)) + " |")
    (TABLES / f"{name}.md").write_text(
        "\n".join([header, separator, *rows]) + "\n", encoding="utf-8"
    )


def save_current_figure(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close()


def evaluate_classifier(name, model, features_test, target_test):
    y_pred_model = model.predict(features_test)
    if hasattr(model, "predict_proba"):
        y_score_model = model.predict_proba(features_test)[:, 1]
    else:
        decision_scores = model.decision_function(features_test)
        score_range = decision_scores.max() - decision_scores.min()
        if score_range == 0:
            y_score_model = np.full_like(decision_scores, fill_value=0.5, dtype=float)
        else:
            y_score_model = (decision_scores - decision_scores.min()) / score_range

    return {
        "Model": name,
        "Accuracy": accuracy_score(target_test, y_pred_model),
        "Precision": precision_score(target_test, y_pred_model, zero_division=0),
        "Recall": recall_score(target_test, y_pred_model, zero_division=0),
        "F1 Score": f1_score(target_test, y_pred_model, zero_division=0),
        "ROC-AUC": roc_auc_score(target_test, y_score_model),
    }


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="white", palette="Set2")
    plt.rcParams["figure.figsize"] = (9, 5)
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.grid"] = True

    df = pd.read_csv(ROOT / "data" / "telco-customer-churn.csv")
    dataset_overview = pd.DataFrame(
        [
            ["Original observations", len(df)],
            ["Original columns", df.shape[1]],
            ["Blank TotalCharges values", df["TotalCharges"].astype(str).str.strip().eq("").sum()],
        ],
        columns=["Item", "Value"],
    )

    clean_df = df.copy()
    clean_df["TotalCharges"] = pd.to_numeric(clean_df["TotalCharges"], errors="coerce")
    clean_df = clean_df.dropna(subset=["TotalCharges"]).copy()
    clean_df["ChurnBinary"] = clean_df["Churn"].map({"No": 0, "Yes": 1})

    cleaning_summary = pd.DataFrame(
        [
            ["Cleaned observations", len(clean_df)],
            ["Rows removed", len(df) - len(clean_df)],
            ["Churned customers", int(clean_df["ChurnBinary"].sum())],
            ["Churn rate (%)", round(clean_df["ChurnBinary"].mean() * 100, 2)],
        ],
        columns=["Item", "Value"],
    )
    save_table(dataset_overview, "dataset_overview")
    save_table(cleaning_summary, "cleaning_summary")

    churn_distribution = (
        clean_df["Churn"]
        .value_counts()
        .rename_axis("Churn")
        .reset_index(name="Count")
    )
    churn_distribution["Percentage"] = (
        churn_distribution["Count"] / churn_distribution["Count"].sum() * 100
    ).round(2)
    save_table(churn_distribution, "churn_distribution")

    ax = sns.countplot(data=clean_df, x="Churn", hue="Churn", legend=False)
    ax.set_title("Churn Class Distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Number of customers")
    save_current_figure("churn_class_distribution")

    categorical_features_to_compare = [
        "Contract",
        "InternetService",
        "PaymentMethod",
        "TechSupport",
        "OnlineSecurity",
    ]
    fig, axes = plt.subplots(len(categorical_features_to_compare), 1, figsize=(10, 22))
    for ax, feature in zip(axes, categorical_features_to_compare, strict=True):
        churn_rate_by_feature = (
            clean_df.groupby(feature, observed=False)["ChurnBinary"]
            .mean()
            .sort_values(ascending=False)
            .mul(100)
            .reset_index(name="ChurnRate")
        )
        sns.barplot(
            data=churn_rate_by_feature,
            x="ChurnRate",
            y=feature,
            hue=feature,
            ax=ax,
            legend=False,
        )
        ax.set_title(f"Churn Rate by {feature}")
        ax.set_xlabel("Churn rate (%)")
        ax.set_ylabel(feature)
    save_current_figure("categorical_churn_rates")

    target = "ChurnBinary"
    excluded_columns = ["customerID", "Churn", target]
    X = clean_df.drop(columns=excluded_columns)
    y = clean_df[target]
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [column for column in X.columns if column not in numeric_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    baseline_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    baseline_model.fit(X_train, y_train)
    y_pred = baseline_model.predict(X_test)
    y_pred_proba = baseline_model.predict_proba(X_test)[:, 1]
    baseline_metrics = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
            "Value": [
                accuracy_score(y_test, y_pred),
                precision_score(y_test, y_pred),
                recall_score(y_test, y_pred),
                f1_score(y_test, y_pred),
                roc_auc_score(y_test, y_pred_proba),
            ],
        }
    )
    baseline_metrics["Value"] = baseline_metrics["Value"].round(4)
    save_table(baseline_metrics, "baseline_metrics")
    (TABLES / "baseline_classification_report.txt").write_text(
        classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]),
        encoding="utf-8",
    )

    engineered_df = clean_df.copy()
    service_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    protection_columns = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    streaming_columns = ["StreamingTV", "StreamingMovies"]

    engineered_df["ServiceCount"] = engineered_df[service_columns].eq("Yes").sum(axis=1)
    engineered_df["ProtectionServiceCount"] = engineered_df[protection_columns].eq("Yes").sum(axis=1)
    engineered_df["HasOnlineProtection"] = np.where(
        engineered_df[protection_columns].eq("Yes").any(axis=1), "Yes", "No"
    )
    engineered_df["HasStreamingService"] = np.where(
        engineered_df[streaming_columns].eq("Yes").any(axis=1), "Yes", "No"
    )
    engineered_df["UsesAutomaticPayment"] = np.where(
        engineered_df["PaymentMethod"].str.contains("automatic", case=False), "Yes", "No"
    )
    engineered_df["IsMonthToMonthContract"] = np.where(
        engineered_df["Contract"].eq("Month-to-month"), "Yes", "No"
    )
    engineered_df["AverageChargePerTenureMonth"] = engineered_df["TotalCharges"] / engineered_df[
        "tenure"
    ].replace(0, np.nan)
    engineered_df["AverageChargePerTenureMonth"] = engineered_df[
        "AverageChargePerTenureMonth"
    ].fillna(engineered_df["MonthlyCharges"])
    engineered_df["TenureGroup"] = pd.cut(
        engineered_df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"],
        include_lowest=True,
    )
    engineered_df["MonthlyChargeGroup"] = pd.cut(
        engineered_df["MonthlyCharges"],
        bins=[0, 35, 70, 90, np.inf],
        labels=["low", "medium", "high", "very high"],
        include_lowest=True,
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    sns.barplot(
        data=engineered_df.groupby("TenureGroup", observed=False)["ChurnBinary"]
        .mean()
        .mul(100)
        .reset_index(),
        x="TenureGroup",
        y="ChurnBinary",
        ax=axes[0],
    )
    axes[0].set_title("Churn Rate by Tenure Group")
    axes[0].set_xlabel("Tenure group")
    axes[0].set_ylabel("Churn rate (%)")
    axes[0].tick_params(axis="x", rotation=25)
    sns.barplot(
        data=engineered_df.groupby("ServiceCount", observed=False)["ChurnBinary"]
        .mean()
        .mul(100)
        .reset_index(),
        x="ServiceCount",
        y="ChurnBinary",
        ax=axes[1],
    )
    axes[1].set_title("Churn Rate by Number of Services")
    axes[1].set_xlabel("Service count")
    axes[1].set_ylabel("Churn rate (%)")
    sns.boxplot(data=engineered_df, x="Churn", y="AverageChargePerTenureMonth", ax=axes[2])
    axes[2].set_title("Average Charge per Tenure Month by Churn")
    axes[2].set_xlabel("Churn")
    axes[2].set_ylabel("Average charge per tenure month")
    sns.barplot(
        data=engineered_df.groupby("UsesAutomaticPayment", observed=False)["ChurnBinary"]
        .mean()
        .mul(100)
        .reset_index(),
        x="UsesAutomaticPayment",
        y="ChurnBinary",
        ax=axes[3],
    )
    axes[3].set_title("Churn Rate by Automatic Payment Usage")
    axes[3].set_xlabel("Uses automatic payment")
    axes[3].set_ylabel("Churn rate (%)")
    save_current_figure("engineered_feature_patterns")

    X_engineered = engineered_df.drop(columns=excluded_columns)
    y_engineered = engineered_df[target]
    numeric_features_engineered = X_engineered.select_dtypes(include=["number"]).columns.tolist()
    categorical_features_engineered = [
        column for column in X_engineered.columns if column not in numeric_features_engineered
    ]
    X_train_eng, X_test_eng, y_train_eng, y_test_eng = train_test_split(
        X_engineered,
        y_engineered,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_engineered,
    )

    def make_engineered_preprocessor() -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features_engineered),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_features_engineered,
                ),
            ],
            verbose_feature_names_out=False,
        )

    def make_model_pipeline(estimator):
        return Pipeline(
            steps=[
                ("preprocessor", make_engineered_preprocessor()),
                ("classifier", estimator),
            ]
        )

    candidate_models = {
        "Dummy Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "SVM": SVC(kernel="rbf", random_state=RANDOM_STATE),
        "Gaussian NB": GaussianNB(),
    }
    trained_models = {}
    model_results = []
    cv_scoring = {
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }

    for model_name, candidate_classifier in candidate_models.items():
        model_pipeline = make_model_pipeline(candidate_classifier)
        cv_scores = cross_validate(
            model_pipeline, X_train_eng, y_train_eng, scoring=cv_scoring, cv=5, n_jobs=1
        )
        model_pipeline.fit(X_train_eng, y_train_eng)
        trained_models[model_name] = model_pipeline
        model_result = evaluate_classifier(model_name, model_pipeline, X_test_eng, y_test_eng)
        model_result.update(
            {
                "CV Precision Mean": cv_scores["test_precision"].mean(),
                "CV Precision Std": cv_scores["test_precision"].std(),
                "CV Recall Mean": cv_scores["test_recall"].mean(),
                "CV Recall Std": cv_scores["test_recall"].std(),
                "CV F1 Mean": cv_scores["test_f1"].mean(),
                "CV F1 Std": cv_scores["test_f1"].std(),
                "CV ROC-AUC Mean": cv_scores["test_roc_auc"].mean(),
                "CV ROC-AUC Std": cv_scores["test_roc_auc"].std(),
            }
        )
        model_results.append(model_result)

    model_comparison = (
        pd.DataFrame(model_results)
        .sort_values(by=["F1 Score", "ROC-AUC"], ascending=False)
        .reset_index(drop=True)
    )
    save_table(model_comparison.round(4), "model_comparison")

    baseline_logistic_summary = baseline_metrics.set_index("Metric")["Value"].to_dict()
    baseline_vs_engineered = pd.DataFrame(
        [
            {
                "Model": "Logistic Regression - Original Features",
                "Accuracy": baseline_logistic_summary["Accuracy"],
                "Precision": baseline_logistic_summary["Precision"],
                "Recall": baseline_logistic_summary["Recall"],
                "F1 Score": baseline_logistic_summary["F1 Score"],
                "ROC-AUC": baseline_logistic_summary["ROC-AUC"],
            },
            evaluate_classifier(
                "Logistic Regression - Engineered Features",
                trained_models["Logistic Regression"],
                X_test_eng,
                y_test_eng,
            ),
        ]
    )
    save_table(baseline_vs_engineered.round(4), "baseline_vs_engineered")

    comparison_plot_df = model_comparison.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
        var_name="Metric",
        value_name="Score",
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(data=comparison_plot_df, x="Score", y="Model", hue="Metric")
    plt.title("Model Comparison Across Evaluation Metrics")
    plt.xlabel("Score")
    plt.ylabel("Model")
    plt.xlim(0, 1)
    plt.legend(loc="lower right")
    save_current_figure("model_comparison_metrics")

    best_model_name = model_comparison.loc[0, "Model"]
    best_model = trained_models[best_model_name]
    best_y_pred = best_model.predict(X_test_eng)
    (TABLES / "initial_best_classification_report.txt").write_text(
        f"Best model by F1 score: {best_model_name}\n"
        + classification_report(y_test_eng, best_y_pred, target_names=["No Churn", "Churn"]),
        encoding="utf-8",
    )

    feature_selection_preprocessor = make_engineered_preprocessor()
    X_train_processed = feature_selection_preprocessor.fit_transform(X_train_eng)
    feature_names = feature_selection_preprocessor.get_feature_names_out()
    mutual_information = mutual_info_classif(
        X_train_processed, y_train_eng, random_state=RANDOM_STATE
    )
    feature_selection_table = (
        pd.DataFrame({"Feature": feature_names, "Mutual Information": mutual_information})
        .sort_values("Mutual Information", ascending=False)
        .reset_index(drop=True)
    )
    save_table(feature_selection_table.head(20).round(4), "top_mutual_information_features")
    plt.figure(figsize=(10, 7))
    sns.barplot(data=feature_selection_table.head(20), x="Mutual Information", y="Feature")
    plt.title("Top 20 Features by Mutual Information")
    plt.xlabel("Mutual information")
    plt.ylabel("Feature")
    save_current_figure("top_mutual_information_features")

    top_k = min(20, len(feature_names))
    selected_feature_model = Pipeline(
        steps=[
            ("preprocessor", make_engineered_preprocessor()),
            ("feature_selection", SelectKBest(score_func=mutual_info_classif, k=top_k)),
            ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )
    selected_feature_model.fit(X_train_eng, y_train_eng)
    selected_vs_full = pd.DataFrame(
        [
            evaluate_classifier(
                "Logistic Regression - Full Features",
                trained_models["Logistic Regression"],
                X_test_eng,
                y_test_eng,
            ),
            evaluate_classifier(
                f"Logistic Regression - Top {top_k} Features",
                selected_feature_model,
                X_test_eng,
                y_test_eng,
            ),
        ]
    )
    save_table(selected_vs_full.round(4), "selected_vs_full_features")

    random_forest_model = trained_models["Random Forest"]
    rf_classifier = random_forest_model.named_steps["classifier"]
    rf_feature_names = random_forest_model.named_steps["preprocessor"].get_feature_names_out()
    rf_importance_table = (
        pd.DataFrame({"Feature": rf_feature_names, "Importance": rf_classifier.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    save_table(rf_importance_table.head(20).round(4), "top_random_forest_importances")
    plt.figure(figsize=(10, 7))
    sns.barplot(data=rf_importance_table.head(20), x="Importance", y="Feature")
    plt.title("Top 20 Random Forest Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    save_current_figure("top_random_forest_importances")

    tuning_searches = {
        "Tuned Random Forest": GridSearchCV(
            estimator=make_model_pipeline(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)),
            param_grid={
                "classifier__n_estimators": [200, 400],
                "classifier__max_depth": [None, 8, 14],
                "classifier__min_samples_split": [2, 10],
                "classifier__class_weight": [None, "balanced"],
            },
            scoring="f1",
            cv=5,
            n_jobs=1,
        ),
        "Tuned Gradient Boosting": GridSearchCV(
            estimator=make_model_pipeline(GradientBoostingClassifier(random_state=RANDOM_STATE)),
            param_grid={
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [2, 3],
            },
            scoring="f1",
            cv=5,
            n_jobs=1,
        ),
    }
    tuned_models = {}
    tuning_results = []
    for search_name, search in tuning_searches.items():
        search.fit(X_train_eng, y_train_eng)
        tuned_models[search_name] = search.best_estimator_
        result = evaluate_classifier(search_name, search.best_estimator_, X_test_eng, y_test_eng)
        result["Best CV F1"] = search.best_score_
        result["Best Parameters"] = str(search.best_params_)
        tuning_results.append(result)

    hyperparameter_results = (
        pd.DataFrame(tuning_results)
        .sort_values(by=["F1 Score", "ROC-AUC"], ascending=False)
        .reset_index(drop=True)
    )
    save_table(hyperparameter_results.round(4), "hyperparameter_results")

    final_comparison = (
        pd.concat(
            [
                model_comparison,
                hyperparameter_results.drop(columns=["Best CV F1", "Best Parameters"]),
                selected_vs_full[selected_vs_full["Model"].str.contains("Top")],
            ],
            ignore_index=True,
        )
        .sort_values(by=["F1 Score", "ROC-AUC"], ascending=False)
        .reset_index(drop=True)
    )
    save_table(final_comparison.round(4), "final_comparison")

    final_best_model_name = final_comparison.loc[0, "Model"]
    if final_best_model_name in tuned_models:
        final_best_model = tuned_models[final_best_model_name]
    elif final_best_model_name == f"Logistic Regression - Top {top_k} Features":
        final_best_model = selected_feature_model
    else:
        final_best_model = trained_models[final_best_model_name]

    final_y_pred = final_best_model.predict(X_test_eng)
    final_y_score = (
        final_best_model.predict_proba(X_test_eng)[:, 1]
        if hasattr(final_best_model, "predict_proba")
        else final_best_model.decision_function(X_test_eng)
    )
    (TABLES / "final_classification_report.txt").write_text(
        f"Final best model by test F1 score: {final_best_model_name}\n"
        + classification_report(y_test_eng, final_y_pred, target_names=["No Churn", "Churn"]),
        encoding="utf-8",
    )
    ConfusionMatrixDisplay.from_predictions(
        y_test_eng,
        final_y_pred,
        display_labels=["No Churn", "Churn"],
        cmap="Blues",
        values_format="d",
    )
    plt.title(f"Final Confusion Matrix - {final_best_model_name}")
    save_current_figure("final_confusion_matrix")

    RocCurveDisplay.from_predictions(y_test_eng, final_y_score)
    plt.title(f"Final ROC Curve - {final_best_model_name}")
    save_current_figure("final_roc_curve")


if __name__ == "__main__":
    main()
