import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy.

    Args:
        y_true: True labels (1D array of integers).
        y_pred: Predicted labels (1D array of integers).

    Returns:
        Accuracy score (float).
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    return np.mean(y_true == y_pred)


def calculate_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate macro F1 score for multi-class classification.

    Args:
        y_true: True labels (1D array of integers).
        y_pred: Predicted labels (1D array of integers).

    Returns:
        Macro F1 score (float).
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1_scores = []

    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return np.mean(f1_scores)


def calculate_topk_accuracy(y_true: np.ndarray, probabilities: np.ndarray, k: int = 5) -> float:
    """
    Calculate top-k accuracy.

    Args:
        y_true: True labels (1D array of integers).
        probabilities: Predicted probabilities (2D array: n_samples x n_classes).
        k: Number of top predictions to consider.

    Returns:
        Top-k accuracy score (float).
    """
    if len(y_true) != probabilities.shape[0]:
        raise ValueError("y_true and probabilities must have compatible shapes.")

    top_k_preds = np.argsort(probabilities, axis=1)[:, -k:]
    correct_count = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_k_preds[i]:
            correct_count += 1
    return correct_count / len(y_true)


def calculate_roc_auc(y_true: np.ndarray, probabilities: np.ndarray) -> Tuple[float, Dict]:
    """
    Calculate ROC AUC score for multi-class classification, with NaN/Inf handling.

    Args:
        y_true: True labels (1D array of integers).
        probabilities: Predicted probabilities (2D array: n_samples x n_classes).

    Returns:
        Tuple of (macro AUC, detailed results).  The detailed results dictionary
        includes FPR, TPR, and AUC for each class, as well as macro-averaged
        FPR, TPR, and AUC.
    """
    if len(y_true) != probabilities.shape[0]:
       raise ValueError("y_true and probabilities must have the same number of samples.")

    classes = np.unique(y_true)
    n_classes = len(classes)

    # Binarize the labels
    y_bin = label_binarize(y_true, classes=classes)

    if n_classes == 1:
        y_bin = y_bin.reshape(-1, 1)

    if y_bin.shape[1] != probabilities.shape[1]:
        raise ValueError("Number of classes in y_true does not match probabilities.")


    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])  # Calculate AUC even if fpr/tpr are short



    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)

    return macro_auc, {
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'macro_fpr': all_fpr,
        'macro_tpr': mean_tpr,
        'macro_auc': macro_auc
    }


def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate confusion matrix.

    Args:
        y_true: True labels (1D array of integers).
        y_pred: Predicted labels (1D array of integers).

    Returns:
        Confusion matrix (2D array: n_classes x n_classes).
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = np.where(classes == true_label)[0][0]
        pred_idx = np.where(classes == pred_label)[0][0]
        conf_matrix[true_idx, pred_idx] += 1

    return conf_matrix


def calculate_precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate precision and recall for each class.

    Args:
        y_true: True labels (1D array of integers).
        y_pred: Predicted labels (1D array of integers).

    Returns:
        Tuple of (precision, recall), where each is a 1D array of length n_classes.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)

    for i, c in enumerate(classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return precision, recall


def combine_fold_results(fold_results: List[Dict]) -> Dict:
    """
    Combine results from multiple folds (e.g., cross-validation).

    Args:
        fold_results: List of dictionaries, where each dictionary contains
                      results from a single fold.  Each dictionary should
                      have the same keys.

    Returns:
        Combined results dictionary.  Scalar values are averaged across folds.
        Non-scalar values (like lists or arrays) are returned as lists of values
        from each fold.
    """
    combined = {}
    if not fold_results:
        return combined  # Handle empty input list

    first_fold = fold_results[0]
    for key in first_fold:
        if np.isscalar(first_fold[key]):
            combined[key] = np.mean([fold[key] for fold in fold_results])
        else:
            combined[key] = [fold[key] for fold in fold_results]
    return combined


