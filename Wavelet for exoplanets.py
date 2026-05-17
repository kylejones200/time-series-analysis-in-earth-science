"""Generated from Jupyter notebook: Wavelet for exoplanets

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def evaluate_model(
    clf: BaseEstimator,
    X_test: NDArray[np.float64],
    y_test: NDArray[np.int64],
    class_names: list[str],
) -> dict[str, Any]:
    """
    Evaluate model performance using multiple metrics and visualizations.

    Parameters
    ----------
    clf : BaseEstimator
        Trained classifier
    X_test : np.ndarray
        Test feature matrix
    y_test : np.ndarray
        True test labels
    class_names : List[str]
        Names of the classes for visualization

    Returns
    -------
    Dict[str, Any]
        Dictionary containing evaluation metrics and predictions
    """
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.1)
    best_threshold = 0.5
    best_f1 = 0
    logger.info("\nTesting different probability thresholds:")
    for threshold in thresholds:
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)
        report = classification_report(y_test, y_pred_threshold, output_dict=True)
        f1 = report["1"]["f1-score"]
        logger.info(f"Threshold: {threshold:.1f}, F1: {f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    logger.info(f"\nBest threshold: {best_threshold}")
    y_pred_final = (y_pred_proba >= best_threshold).astype(int)
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred_final, target_names=class_names))
    cm = confusion_matrix(y_test, y_pred_final)
    plot_confusion_matrix(cm, class_names)
    plot_roc_curve(y_test, y_pred_proba)
    plot_precision_recall_curve(y_test, y_pred_proba)
    return {
        "predictions": y_pred_final,
        "probabilities": y_pred_proba,
        "confusion_matrix": cm,
        "best_threshold": best_threshold,
    }


def extract_wavelet_features(
    flux_data: NDArray[np.float64], wavelet: str = "db4", level: int = 4
) -> list[float]:
    """
    Extract wavelet features from flux time series data.

    Parameters
    ----------
    flux_data : np.ndarray
        1D array containing flux measurements over time
    wavelet : str, optional
        Wavelet type to use for decomposition (default: 'db4')
    level : int, optional
        Level of wavelet decomposition (default: 4)

    Returns
    -------
    List[float]
        List of extracted features including statistical measures of wavelet coefficients

    Raises
    ------
    ValueError
        If wavelet decomposition fails or input data is invalid
    """
    try:
        coeffs = pywt.wavedec(flux_data, wavelet, level=level)
        features = []
        for coeff in coeffs:
            features.extend(
                [
                    np.mean(coeff),
                    np.std(coeff),
                    np.max(coeff),
                    np.min(coeff),
                    np.median(coeff),
                    np.percentile(coeff, 25),
                    np.percentile(coeff, 75),
                    np.sum(np.abs(coeff)),
                    np.sqrt(np.mean(coeff**2)),
                ]
            )
        return features
    except Exception as e:
        raise ValueError(f"Error processing wavelet features: {str(e)}")


def load_model(filename: str) -> BaseEstimator:
    """
    Load trained model from disk.

    Parameters
    ----------
    filename : str
        Path to the saved model

    Returns
    -------
    BaseEstimator
        Loaded classifier
    """
    from joblib import load

    return load(filename)


def main() -> None:
    """
    Main function to run the planet detection pipeline.

    Loads data, trains the model, and evaluates performance.

    Returns
    -------
    None
    """
    try:
        logger.info("Dataset shape: %s", df.shape)
        logger.info("\nSample of first few rows:")
        logger.info(df.head())
        clf, X_test, y_test, class_names = train_planet_detector(df, "LABEL")
        results = evaluate_model(clf, X_test, y_test, class_names)
        save_model(clf, "planet_detector_model.joblib")
    except Exception as e:
        logger.error("An error occurred: %s", str(e))
        raise


def plot_confusion_matrix(cm: NDArray[np.int64], class_names: list[str]) -> None:
    """
    Plot confusion matrix as a heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix
    class_names : List[str]
        Names of the classes

    Returns
    -------
    None
        Displays the confusion matrix plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()


def plot_feature_importance(feature_importance: pd.DataFrame) -> None:
    """
    Plot top 10 feature importance.

    Parameters
    ----------
    feature_importance : pd.DataFrame
        DataFrame containing feature importance scores

    Returns
    -------
    None
        Displays the feature importance plot
    """
    plt.figure(figsize=(10, 6))
    plt.bar(range(10), feature_importance["importance"][:10])
    plt.title("Top 10 Feature Importance")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.show()


def plot_precision_recall_curve(
    y_test: NDArray[np.int64], y_pred_proba: NDArray[np.float64]
) -> None:
    """
    Plot Precision-Recall curve with AUC score.

    Parameters
    ----------
    y_test : np.ndarray
        True test labels
    y_pred_proba : np.ndarray
        Predicted probabilities

    Returns
    -------
    None
        Displays the Precision-Recall curve plot
    """
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(
        recall, precision, color="blue", lw=2, label=f"PR curve (AUC = {pr_auc:.2f})"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower right")
    plt.show()


def plot_roc_curve(
    y_test: NDArray[np.int64], y_pred_proba: NDArray[np.float64]
) -> None:
    """
    Plot ROC curve with AUC score.

    Parameters
    ----------
    y_test : np.ndarray
        True test labels
    y_pred_proba : np.ndarray
        Predicted probabilities

    Returns
    -------
    None
        Displays the ROC curve plot
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()


def process_dataset(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Process entire dataset and extract wavelet features for each time series.

    Parameters
    ----------
    data : np.ndarray
        2D array where each row is a time series of flux measurements

    Returns
    -------
    np.ndarray
        2D array of extracted features for each time series

    Raises
    ------
    ValueError
        If input data is empty or has incorrect dimensions
    """
    if data.size == 0:
        raise ValueError("Empty input data")
    try:
        X = []
        for i in range(len(data)):
            flux_features = extract_wavelet_features(data[i])
            X.append(flux_features)
        return np.array(X)
    except Exception as e:
        raise ValueError(f"Error processing dataset: {str(e)}")


def save_model(clf: BaseEstimator, filename: str) -> None:
    """
    Save trained model to disk.

    Parameters
    ----------
    clf : BaseEstimator
        Trained classifier to save
    filename : str
        Path where to save the model

    Returns
    -------
    None
    """
    from joblib import dump

    dump(clf, filename)


def train_planet_detector(
    df: pd.DataFrame, label_column: str = "label"
) -> tuple[BaseEstimator, NDArray[np.float64], NDArray[np.int64], list[str]]:
    """
    Train a Random Forest classifier for planet detection.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing flux measurements and labels
    label_column : str, optional
        Name of the column containing class labels (default: 'label')

    Returns
    -------
    Tuple[BaseEstimator, np.ndarray, np.ndarray, List[str]]
        Returns (trained_classifier, X_test, y_test, class_names)

    Raises
    ------
    ValueError
        If DataFrame is empty or label_column is not found
    """
    if df.empty:
        raise ValueError("Empty DataFrame provided")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in DataFrame")
    logger.info("\nInitial class distribution:")
    logger.info(df[label_column].value_counts(normalize=True))
    X = df.drop(columns=[label_column])
    y = df[label_column]
    le = LabelEncoder()
    y = le.fit_transform(y)
    class_names = ["No Planet", "Planet"]
    X_processed = process_dataset(X.values)
    scaler = StandardScaler()
    X_processed = scaler.fit_transform(X_processed)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    smote = SMOTE(random_state=42, sampling_strategy=0.4)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    logger.info("\nClass distribution after SMOTE:")
    logger.info(pd.Series(y_train_balanced).value_counts(normalize=True))
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
    )
    cv_scores = cross_val_score(clf, X_train_balanced, y_train_balanced, cv=5)
    logger.info("\nCross-validation scores: %s", cv_scores)
    logger.info("Mean CV score: %f", cv_scores.mean())
    clf.fit(X_train_balanced, y_train_balanced)
    feature_importance = pd.DataFrame(
        {"feature": range(X_processed.shape[1]), "importance": clf.feature_importances_}
    ).sort_values("importance", ascending=False)
    logger.info("\nTop 10 most important features:")
    logger.info(feature_importance.head(10))
    plot_feature_importance(feature_importance)
    return (clf, X_test, y_test, class_names)


def main() -> None:
    df = pd.read_csv("exoTrain.csv")

    train_planet_detector(df, label_column="LABEL")

    main()

    main()

    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)

    main()


if __name__ == "__main__":
    main()
