import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import logging
from .metrics import calculate_accuracy, calculate_topk_accuracy, calculate_f1_score  # Absolute import

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class KNNClassifier:
    """
    K-Nearest Neighbors classifier with support for multiple distance metrics.
    """

    def __init__(self, k: int = 5, metric: str = 'euclidean'):
        """
        Initialize the KNN classifier.

        Args:
            k: Number of neighbors.
            metric: Distance metric ('euclidean' or 'cosine').
        """
        if metric not in ('euclidean', 'cosine'):
            raise ValueError("metric must be 'euclidean' or 'cosine'")
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the classifier to the training data (stores the data).

        Args:
            X: Training features (n_samples x n_features).
            y: Training labels (n_samples,).
        """
        self.X_train = X
        self.y_train = y

    def _calculate_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate distances between test points and training points.

        Args:
            X: Test features (n_samples x n_features).

        Returns:
            Array of distances (n_samples x n_training_samples).
        """
        if self.metric == 'euclidean':
            return self._euclidean_distance(X)
        elif self.metric == 'cosine':
            return self._cosine_distance(X)
        else:  
            raise ValueError(f"Unsupported distance metric: {self.metric}")

    def _euclidean_distance(self, X: np.ndarray) -> np.ndarray:
        """Calculate Euclidean distances."""
        # Efficient, vectorized calculation:
        X_squared = np.sum(X**2, axis=1, keepdims=True)
        X_train_squared = np.sum(self.X_train**2, axis=1)
        distances = np.sqrt(np.maximum(X_squared - 2 * np.dot(X, self.X_train.T) + X_train_squared, 0))
        return distances

    def _cosine_distance(self, X: np.ndarray) -> np.ndarray:
        """Calculate Cosine distances (1 - cosine similarity)."""
        # Normalize vectors
        X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
        X_train_norm = self.X_train / np.linalg.norm(self.X_train, axis=1, keepdims=True)
        # Cosine similarity, then distance (1 - similarity)
        distances = 1 - np.dot(X_norm, X_train_norm.T)
        return distances

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for test data.

        Args:
            X: Test features (n_samples x n_features).

        Returns:
            Predicted labels (n_samples,).
        """
        distances = self._calculate_distances(X)
        k_nearest_indices = np.argsort(distances, axis=1)[:, :self.k]
        k_nearest_labels = self.y_train[k_nearest_indices]
        # Majority vote
        predictions = np.array([np.argmax(np.bincount(k_nearest_labels[i])) for i in range(X.shape[0])])
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for test data.

        Args:
            X: Test features (n_samples x n_features).

        Returns:
            Class probabilities (n_samples x n_classes).
        """
        distances = self._calculate_distances(X)
        k_nearest_indices = np.argsort(distances, axis=1)[:, :self.k]
        k_nearest_labels = self.y_train[k_nearest_indices]
        n_samples = X.shape[0]
        n_classes = len(np.unique(self.y_train))  # Get number of classes from training data
        probabilities = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            for j in range(self.k):
                label = k_nearest_labels[i, j]
                probabilities[i, label] += 1
        probabilities /= self.k  # Normalize 
        return probabilities


def evaluate_knn_with_cross_validation(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_values: List[int] = [1, 3, 5, 7, 9, 11, 13, 15],
    metrics: List[str] = ['euclidean', 'cosine'],
    n_splits: int = 10,
    random_state: int = 42
) -> Dict:
    """
    Evaluate KNN classifier with cross-validation.

    Args:
        embeddings: Array of embeddings (n_samples x n_features).
        labels: Array of labels (n_samples,).
        k_values: List of k values to try.
        metrics: List of distance metrics to try ('euclidean', 'cosine').
        n_splits: Number of splits for cross-validation.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary of evaluation results. The structure is:
        {
            'metric_name': {
                k_value: {
                    'accuracy': ...,
                    'f1_score': ...,
                    'top3_accuracy': ...,
                    'top5_accuracy': ...,
                },
                ...
            },
            ...
        }
    """
    results = {}

    for metric in metrics:
        results[metric] = {}
        logging.info(f"Evaluating KNN with {metric} distance")

        for k in k_values:
            logging.info(f"  k = {k}")
            fold_results = []  # Store results for each fold

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

            for fold, (train_idx, test_idx) in enumerate(kf.split(embeddings)):
                X_train, X_test = embeddings[train_idx], embeddings[test_idx]
                y_train, y_test = labels[train_idx], labels[test_idx]

                # Standardize the data
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # Fit and predict
                knn = KNNClassifier(k=k, metric=metric)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                probabilities = knn.predict_proba(X_test)

                # Calculate metrics
                accuracy = calculate_accuracy(y_test, y_pred)
                f1 = calculate_f1_score(y_test, y_pred)
                top3_accuracy = calculate_topk_accuracy(y_test, probabilities, k=3)
                top5_accuracy = calculate_topk_accuracy(y_test, probabilities, k=5)

                # Store fold results
                fold_results.append({
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'top3_accuracy': top3_accuracy,
                    'top5_accuracy': top5_accuracy,
                })
                logging.info(f"    Fold {fold+1}/{n_splits}: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")

            # Combine fold results 
            combined_results = {}
            first_fold = fold_results[0]
            for key in first_fold:
                if np.isscalar(first_fold[key]):
                  combined_results[key] = np.mean([fold[key] for fold in fold_results])
                else:
                    combined_results[key] = [fold[key] for fold in fold_results]


            results[metric][k] = combined_results
            logging.info(f"  Average accuracy for k={k}: {combined_results['accuracy']:.4f}")
            logging.info(f"  Average F1 score for k={k}: {combined_results['f1_score']:.4f}")

    return results


def find_optimal_k(results: Dict) -> Dict:
    """
    Find the optimal k value and metric based on accuracy.

    Args:
      results: Dictionary of cross-validation results, as returned by
        `evaluate_knn_with_cross_validation`.

    Returns:
      Dictionary with the best metric and corresponding k and metrics.  Structure:
      {
        'best_metric': 'euclidean',  # or 'cosine'
        'best_k': 5,
        'best_metrics': { ... } # Metrics for the best k and metric
      }
      Returns an empty dictionary if the input is empty or invalid.
    """

    best_metric = None
    best_k = None
    best_accuracy = -1.0

    for metric, k_results in results.items():
        for k, metrics in k_results.items():
            if 'accuracy' in metrics and metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_k = k
                best_metric = metric

    if best_metric is not None:
        return {
            'best_metric': best_metric,
            'best_k': best_k,
            'best_metrics': results[best_metric][best_k]
        }
    else:
        return {}  # Return empty dict if no best k found


if __name__ == "__main__":
    # Example Usage
    import pickle

    # Load processed data (replace with your actual data loading)
    with open("processed_data.pkl", 'rb') as f:
        data = pickle.load(f)
    embeddings = data["embeddings"]
    labels = data["labels"]
    # Evaluate KNN with cross-validation
    results = evaluate_knn_with_cross_validation(embeddings, labels)

    # Find optimal k and metric
    optimal_results = find_optimal_k(results)
    if optimal_results:
        print("Optimal k and metric:")
        print(f"  Metric: {optimal_results['best_metric']}")
        print(f"  k: {optimal_results['best_k']}")
        print(f"  Accuracy: {optimal_results['best_metrics']['accuracy']:.4f}")
        print(f"  F1 score: {optimal_results['best_metrics']['f1_score']:.4f}")
        print(f"  Top-3 accuracy: {optimal_results['best_metrics']['top3_accuracy']:.4f}")
        print(f"  Top-5 accuracy: {optimal_results['best_metrics']['top5_accuracy']:.4f}")
    else:
        print("No optimal k/metric found.")