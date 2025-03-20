import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple
import logging
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def perform_tsne(embeddings: np.ndarray, perplexity: int = 30,
                 n_iter: int = 1000, random_state: int = 42) -> np.ndarray:
    """
    Perform t-SNE dimensionality reduction
    
    Args:
        embeddings: Array of embeddings
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
        random_state: Random seed for reproducibility
        
    Returns:
        Reduced embeddings
    """
    logging.info(f"Performing t-SNE dimensionality reduction with perplexity={perplexity}")
    
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                random_state=random_state, n_jobs=-1)
    
    try:
        reduced_embeddings = tsne.fit_transform(embeddings)
        logging.info("t-SNE dimensionality reduction completed successfully")
        return reduced_embeddings
    except Exception as e:
        logging.error(f"Error during t-SNE: {e}")
        raise

def plot_tsne_embeddings(reduced_embeddings: np.ndarray, labels: np.ndarray,
                          syndrome_mapping: Dict, output_file: str = None) -> None:
    """
    Plot t-SNE embeddings

    Args:
        reduced_embeddings: Reduced embeddings from t-SNE
        labels: Array of labels
        syndrome_mapping: Mapping of syndrome_id to numeric label
        output_file: Path to save the plot
    """
    plt.figure(figsize=(12, 10))

    # Create a color map
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    # Reverse the syndrome mapping for plotting
    inverse_mapping = {v: k for k, v in syndrome_mapping.items()}

    # Plot each syndrome with a different color
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1],
                    c=[colors[i]], label=inverse_mapping[label], alpha=0.7)

    plt.title('t-SNE Visualization of Genetic Syndrome Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    # Add legend with smaller font size if there are many syndromes
    if len(unique_labels) > 10:
        plt.legend(fontsize='small', loc='best', bbox_to_anchor=(1.05, 1),
                   title='Syndrome ID')
    else:
        plt.legend(loc='best', bbox_to_anchor=(1.05, 1), title='Syndrome ID')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logging.info(f"t-SNE plot saved to {output_file}")

    plt.show()

def analyze_clusters(reduced_embeddings: np.ndarray, labels: np.ndarray,
                      syndrome_mapping: Dict) -> Dict:
    """
    Analyze clusters in the t-SNE visualization

    Args:
        reduced_embeddings: Reduced embeddings from t-SNE
        labels: Array of labels
        syndrome_mapping: Mapping of syndrome_id to numeric label

    Returns:
        Dict: Analysis of clusters
    """
    cluster_analysis = {}

    # Reverse the syndrome mapping
    inverse_mapping = {v: k for k, v in syndrome_mapping.items()}

    
    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label
        cluster_points = reduced_embeddings[mask]

        # Calculate cluster center
        center = np.mean(cluster_points, axis=0)

        # Calculate average distance from center
        distances = np.sqrt(np.sum((cluster_points - center) ** 2, axis=1))
        avg_distance = np.mean(distances)

        # Calculate overlap with other clusters
        overlap_info = {}
        for other_label in unique_labels:
            if other_label != label:
                other_mask = labels == other_label
                other_points = reduced_embeddings[other_mask]
                other_center = np.mean(other_points, axis=0)

                # Calculate distance between centers
                center_distance = np.sqrt(np.sum((center - other_center) ** 2))

                # Calculate overlap score (lower means more overlap)
                overlap_score = center_distance / (avg_distance + np.mean(
                    np.sqrt(np.sum((other_points - other_center) ** 2, axis=1))))

                overlap_info[inverse_mapping[other_label]] = overlap_score

        cluster_analysis[inverse_mapping[label]] = {
            "num_points": np.sum(mask),
            "center": center,
            "avg_distance_from_center": avg_distance,
            "overlap_with_other_clusters": overlap_info
        }

    return cluster_analysis

def plot_multiple_tsne_perplexities(embeddings: np.ndarray, labels: np.ndarray,
                                    perplexities: List[int] = [5, 30, 50, 100],
                                    output_file: str = None) -> None:
    """
    Plot t-SNE visualizations with different perplexity values

    Args:
        embeddings: Array of embeddings
        labels: Array of labels
        perplexities: List of perplexity values to try
        output_file: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    # Create a color map
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    for i, perplexity in enumerate(perplexities):
        if i >= len(axes):
            break

        logging.info(f"Performing t-SNE with perplexity={perplexity}")

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
        reduced_embeddings = tsne.fit_transform(embeddings)

        for j, label in enumerate(unique_labels):
            mask = labels == label
            axes[i].scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1],
                            c=[colors[j]], alpha=0.7, s=20)

        axes[i].set_title(f't-SNE with Perplexity = {perplexity}')
        axes[i].set_xlabel('t-SNE Component 1')
        axes[i].set_ylabel('t-SNE Component 2')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logging.info(f"Multiple t-SNE plots saved to {output_file}")

    plt.show()

def plot_roc_curves(probabilities: np.ndarray, labels: np.ndarray, syndrome_mapping: Dict, output_file: str = None) -> None:
    """
    Plots ROC curves for each syndrome.

    Args:
        probabilities: Predicted probabilities (shape: n_samples x n_classes).
        labels: True labels.
        syndrome_mapping: Mapping of syndrome ID to numeric label.
        output_file: Path to save the plot (optional).
    """

    n_classes = len(syndrome_mapping)
    inverse_mapping = {v: k for k, v in syndrome_mapping.items()}

    # Binarize the labels
    y_true_bin = label_binarize(labels, classes=list(syndrome_mapping.values()))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    colors = plt.cm.jet(np.linspace(0, 1, n_classes))

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of {inverse_mapping[i]} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves for Each Syndrome')
    plt.legend(loc="lower right")

    if output_file:
        plt.savefig(output_file, dpi=300)
        logging.info(f"ROC curves plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    # This code will run when the script is executed directly
    import pickle

    # Load processed data
    with open("processed_data.pkl", 'rb') as f:
        data = pickle.load(f)

    embeddings = data["embeddings"]
    labels = data["labels"]
    syndrome_mapping = data["syndrome_mapping"]

    # Perform t-SNE
    reduced_embeddings = perform_tsne(embeddings)

    # Plot t-SNE embeddings
    plot_tsne_embeddings(reduced_embeddings, labels, syndrome_mapping, "tsne_plot.png")

    # Analyze clusters
    cluster_analysis = analyze_clusters(reduced_embeddings, labels, syndrome_mapping)
    print("Cluster analysis:")
    for syndrome, analysis in cluster_analysis.items():
        print(f"Syndrome {syndrome}:")
        print(f"  Number of points: {analysis['num_points']}")
        print(f"  Average distance from center: {analysis['avg_distance_from_center']:.2f}")
        print("  Overlap with other clusters:")
        for other_syndrome, overlap in analysis['overlap_with_other_clusters'].items():
            print(f"    {other_syndrome}: {overlap:.2f}")


    n_samples = len(labels)
    n_classes = len(syndrome_mapping)
    dummy_probabilities = np.random.rand(n_samples, n_classes)

   
    for i in range(n_samples):
        true_label = labels[i]
        dummy_probabilities[i, true_label] += 0.5  # 
        dummy_probabilities[i] /= np.sum(dummy_probabilities[i])  # Normalize

    plot_roc_curves(dummy_probabilities, labels, syndrome_mapping, "roc_curves.png")