import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import logging
import os
import pickle
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages  # Import PdfPages

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_directory(directory: str) -> None:
    """
    Create directory if it doesn't exist

    Args:
        directory: Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

def plot_roc_curves(roc_results: Dict, output_file: str = None) -> None:
    """
    Plot ROC curves and optionally save to a file.
    """
    plt.figure(figsize=(10, 8))
    for metric, results in roc_results.items():
        plt.plot(results['macro_fpr'], results['macro_tpr'],
                 label=f"{metric} (AUC = {results['macro_auc']:.3f})", lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different Distance Metrics')
    plt.legend(loc="lower right")
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logging.info(f"ROC curve plot saved to {output_file}")
    plt.close() # Close figure


def plot_k_vs_accuracy(results: Dict, output_file: str = None) -> None:
    """Plots k vs accuracy and optionally saves to a file."""
    plt.figure(figsize=(10, 6))
    for metric, k_results in results.items():
        k_values = sorted(k_results.keys())
        accuracies = [k_results[k]['accuracy'] for k in k_values]
        plt.plot(k_values, accuracies, marker='o', label=f"{metric}")
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs k for Different Distance Metrics')
    plt.xticks(sorted(k_values))
    plt.grid(True, alpha=0.3)
    plt.legend()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logging.info(f"k vs accuracy plot saved to {output_file}")
    plt.close()


def create_results_table(results: Dict) -> pd.DataFrame:
    """Creates a Pandas DataFrame summarizing the results."""
    table_data = []
    for metric, k_results in results.items():
        for k, metrics in k_results.items():
            row = {'Metric': metric, 'k': k, 'Accuracy': metrics['accuracy'],
                   'F1 Score': metrics['f1_score'], 'Top-3 Accuracy': metrics['top3_accuracy'],
                   'Top-5 Accuracy': metrics['top5_accuracy']}
            table_data.append(row)
    return pd.DataFrame(table_data)


def save_results(results: Dict, filename: str) -> None:
    """Saves results to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    logging.info(f"Results saved to {filename}")

def load_results(filename: str) -> Dict:
    """Loads results from a pickle file."""
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    logging.info(f"Results loaded from {filename}")
    return results


def plot_confusion_matrix(confusion_matrix: np.ndarray, classes: List[str],
                           output_file: str = None) -> None:
    """Plots the confusion matrix and optionally saves it to a file."""
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logging.info(f"Confusion matrix plot saved to {output_file}")
    plt.close()


def plot_multiple_metrics(results: Dict, output_file: str = None) -> None:
    """Plots multiple metrics and optionally saves to a file."""
    metrics = ['accuracy', 'f1_score', 'top3_accuracy', 'top5_accuracy']
    metric_labels = ['Accuracy', 'F1 Score', 'Top-3 Accuracy', 'Top-5 Accuracy']
    plt.figure(figsize=(15, 10))
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        plt.subplot(2, 2, i+1)
        for dist_metric, k_results in results.items():
            k_values = sorted(k_results.keys())
            values = [k_results[k][metric] for k in k_values]
            plt.plot(k_values, values, marker='o', label=f"{dist_metric}")
        plt.xlabel('k')
        plt.ylabel(label)
        plt.title(f"{label} vs k")
        plt.xticks(sorted(k_values))
        plt.grid(True, alpha=0.3)
        plt.legend()
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logging.info(f"Multiple metrics plot saved to {output_file}")
    plt.close()


def generate_report(
    report_data: Dict,
    tsne_plot_file: str,
    roc_plot_file: str,
    k_vs_accuracy_plot_file: str,
    multiple_tsne_plot_file: str,
    multiple_metrics_plot: str,
    output_file: str
) -> None:
    """
    Generates a PDF report summarizing the analysis.

    Args:
        report_data: Dictionary containing the data for the report.
        tsne_plot_file: Path to the t-SNE plot image.
        roc_plot_file: Path to the ROC plot image.
        k_vs_accuracy_plot_file: Path to the k vs accuracy plot image.
        multiple_tsne_plot_file: Path to the multiple t-SNE plot image.
        multiple_metrics_plot: Path to the multiple metrics plot.
        output_file: Path to save the generated PDF report.
    """
    with PdfPages(output_file) as pdf:
        # --- Title Page ---
        plt.figure(figsize=(8.5, 11))
        plt.text(0.5, 0.9, "Genetic Syndrome Classification Report", ha='center', va='center', fontsize=20)
        plt.text(0.5, 0.8, "Apollo Solutions Machine Learning Developer Test", ha='center', va='center', fontsize=16)
        plt.axis('off')
        pdf.savefig()
        plt.close()

        # --- Introduction ---
        plt.figure(figsize=(8.5, 11))
        plt.text(0.05, 0.95, "1. Introduction", fontsize=16, fontweight='bold')
        intro_text = (
            "This report summarizes the analysis of genetic syndromes using embeddings derived from images. "
            "The goal is to classify the syndrome_id associated with each image based on these embeddings. "
            "The analysis includes data preprocessing, exploratory data analysis, visualization using t-SNE, "
            "classification using the K-Nearest Neighbors (KNN) algorithm, and evaluation of the results."
        )
        plt.text(0.05, 0.9, intro_text, fontsize=12, wrap=True)
        plt.axis('off')
        pdf.savefig()
        plt.close()

        # --- Methodology ---
        plt.figure(figsize=(8.5, 11))
        plt.text(0.05, 0.95, "2. Methodology", fontsize=16, fontweight='bold')
        methodology_text = (
            "The following steps were taken in this analysis:\n\n"
            "1. Data Preprocessing: Loaded the data from the pickle file, flattened the hierarchical structure, "
            "and handled missing or inconsistent data (if any).  Embeddings were standardized using StandardScaler.\n\n"
            "2. Exploratory Data Analysis: Calculated statistics about the dataset, including the number of syndromes, "
            "images per syndrome, and identified data imbalances.\n\n"
            "3. Data Visualization: Used t-SNE to reduce the dimensionality of the embeddings to 2D and generated "
            "a plot to visualize the embeddings colored by their syndrome_id.\n\n"
            "4. Classification: Implemented the KNN algorithm with both Cosine and Euclidean distance metrics. "
            "Performed 10-fold cross-validation to evaluate model performance and determine the optimal value of k.\n\n"
            "5. Metrics and Evaluation: Calculated metrics such as Top-k Accuracy, AUC, and F1-Score. Generated ROC AUC curves."
        )
        plt.text(0.05, 0.9, methodology_text, fontsize=12, wrap=True)
        plt.axis('off')
        pdf.savefig()
        plt.close()

        # --- Results ---
        plt.figure(figsize=(8.5, 11))
        plt.text(0.05, 0.95, "3. Results", fontsize=16, fontweight='bold')
        plt.axis('off')
        pdf.savefig()
        plt.close()

        # --- Dataset Statistics ---
        plt.figure(figsize=(8.5, 11))
        plt.text(0.05, 0.95, "3.1 Dataset Statistics", fontsize=14, fontweight='bold')
        stats_text = (
            f"Number of Syndromes: {report_data['data_stats']['num_syndromes']}\n"
            f"Total Subjects: {report_data['data_stats']['total_subjects']}\n"
            f"Total Images: {report_data['data_stats']['total_images']}\n"
            f"Average Images per Syndrome: {report_data['data_stats']['avg_images_per_syndrome']:.2f}\n"
            f"Imbalance Ratio: {report_data['imbalance_info']['imbalance_ratio']:.2f}\n"
            f"Low Representation Syndromes: {', '.join(report_data['imbalance_info']['low_representation_syndromes']) or 'None'}"
        )
        plt.text(0.05, 0.85, stats_text, fontsize=12)
        plt.axis('off')
        pdf.savefig()
        plt.close()

        # --- t-SNE Visualization ---
        plt.figure(figsize=(8.5, 11))
        plt.text(0.05, 0.95, "3.2 t-SNE Visualization", fontsize=14, fontweight='bold')
        plt.imshow(plt.imread(tsne_plot_file))
        plt.axis('off')
        pdf.savefig()
        plt.close()

        # --- Multiple t-SNE ---
        plt.figure(figsize=(8.5, 11))
        plt.text(0.05, 0.95, "3.3 Multiple t-SNE Visualization", fontsize=14,
                 fontweight='bold')  # Add a descriptive title
        plt.imshow(plt.imread(multiple_tsne_plot_file))  # Include the multiple perplexity plot
        plt.axis('off')
        pdf.savefig()
        plt.close()


        # --- Cluster Analysis ---
        plt.figure(figsize=(8.5, 11))
        plt.text(0.05, 0.95, "3.4 Cluster Analysis", fontsize=14, fontweight='bold')
        cluster_text = ""
        for syndrome, analysis in report_data['cluster_analysis'].items():
            cluster_text += f"Syndrome {syndrome}:\n"
            cluster_text += f"  Number of points: {analysis['num_points']}\n"
            cluster_text += f"  Average distance from center: {analysis['avg_distance_from_center']:.2f}\n"
            cluster_text += "  Overlap with other clusters:\n"
            for other_syndrome, overlap in analysis['overlap_with_other_clusters'].items():
                cluster_text += f"    {other_syndrome}: {overlap:.2f}\n"
        plt.text(0.05, 0.8, cluster_text, fontsize=11)
        plt.axis('off')
        pdf.savefig()
        plt.close()



        # --- KNN Results ---
        plt.figure(figsize=(8.5, 11))
        plt.text(0.05, 0.95, "3.5 KNN Classification Results", fontsize=14, fontweight='bold')
        optimal_results_text = (
            f"Optimal Metric: {report_data['optimal_results']['best_metric']}\n"
            f"Optimal k: {report_data['optimal_results']['best_k']}\n"
            f"Accuracy: {report_data['optimal_results']['best_metrics']['accuracy']:.4f}\n"
            f"F1 Score: {report_data['optimal_results']['best_metrics']['f1_score']:.4f}\n"
            f"Top-3 Accuracy: {report_data['optimal_results']['best_metrics']['top3_accuracy']:.4f}\n"
            f"Top-5 Accuracy: {report_data['optimal_results']['best_metrics']['top5_accuracy']:.4f}\n"
        )
        plt.text(0.05, 0.85, optimal_results_text, fontsize=12)

        # Include the results table
        table_text = report_data['results_table'].to_string()
        plt.text(0.05, 0.7, table_text, fontsize=9, family='monospace') # Use monospace for table

        plt.axis('off')
        pdf.savefig()
        plt.close()

        # --- k vs Accuracy Plot ---
        plt.figure(figsize=(8.5, 11))
        plt.text(0.05, 0.95, "3.6 k vs Accuracy", fontsize=14, fontweight='bold')
        plt.imshow(plt.imread(k_vs_accuracy_plot_file))
        plt.axis('off')
        pdf.savefig()
        plt.close()

        # --- ROC Curves ---
        plt.figure(figsize=(8.5, 11))
        plt.text(0.05, 0.95, "3.7 ROC Curves", fontsize=14, fontweight='bold')
        plt.imshow(plt.imread(roc_plot_file))
        plt.axis('off')
        pdf.savefig()
        plt.close()


        # --- Multiple Metrics Plot ---
        plt.figure(figsize=(8.5,11))
        plt.text(0.05, 0.95, "3.8 Multiple Metrics Plot", fontsize=14, fontweight='bold')
        plt.imshow(plt.imread(multiple_metrics_plot))
        plt.axis('off')
        pdf.savefig()
        plt.close()


        # --- Analysis ---
        plt.figure(figsize=(8.5, 11))
        plt.text(0.05, 0.95, "4. Analysis", fontsize=16, fontweight='bold')
        analysis_text = (
            "The t-SNE visualization shows some degree of clustering of syndromes, indicating that the embeddings "
            "contain information that is relevant for classification.  The KNN classifier achieved reasonable "
            "performance, with the optimal k and metric as reported above. The ROC curves provide insights into the "
            "trade-off between true positive rate and false positive rate for different syndromes.  The choice of "
            "distance metric (Cosine or Euclidean) can impact the results, and the optimal metric may depend on the "
            "specific characteristics of the data distribution."
        )
        plt.text(0.05, 0.9, analysis_text, fontsize=12, wrap=True)
        plt.axis('off')
        pdf.savefig()
        plt.close()

        # --- Challenges and Solutions ---
        plt.figure(figsize=(8.5, 11))
        plt.text(0.05, 0.95, "5. Challenges and Solutions", fontsize=16, fontweight='bold')
        challenges_text = (
            "One challenge encountered was handling potential data imbalances, where some syndromes have significantly "
            "fewer samples than others.  This can bias the classifier towards the majority classes. Potential solutions "
            "include oversampling the minority classes, undersampling the majority classes, or using cost-sensitive "
            "learning techniques. Another challenge is the high dimensionality of the embeddings, making it difficult "
            "to visualize and interpret. Dimensionality reduction using t-SNE helps to mitigate this issue.  "
            "It was also important to ensure that the numpy version was up to date to avoid the "
            "\"numpy.core._multiarray_umath\" error."
        )
        plt.text(0.05, 0.9, challenges_text, fontsize=12, wrap=True)
        plt.axis('off')
        pdf.savefig()
        plt.close()

        # --- Recommendations ---
        plt.figure(figsize=(8.5, 11))
        plt.text(0.05, 0.95, "6. Recommendations", fontsize=16, fontweight='bold')
        recommendations_text = (
            "For further analysis, the following steps could be considered:\n\n"
            "- Explore other classification algorithms, such as Support Vector Machines (SVMs) or Random Forests.\n"
            "- Investigate the use of different embedding models or feature extraction techniques.\n"
            "- Address data imbalance using more advanced techniques.\n"
            "- Perform hyperparameter tuning for the chosen classification algorithm.\n"
            "- Collect more data, especially for under-represented syndromes.\n"
            "- Explore ensembling methods to combine predictions from multiple models.\n"
            "- Consider incorporating clinical data or other relevant information to improve classification accuracy."
        )
        plt.text(0.05, 0.9, recommendations_text, fontsize=12, wrap=True)
        plt.axis('off')
        pdf.savefig()
        plt.close()

        # --- Conclusion ---
        plt.figure(figsize=(8.5, 11))
        plt.text(0.05, 0.95, "7. Conclusion", fontsize=16, fontweight='bold')
        conclusion_text = (
            "This project demonstrated the feasibility of classifying genetic syndromes based on image embeddings. "
            "The KNN classifier, combined with t-SNE visualization and appropriate evaluation metrics, provides a "
            "useful approach for this task.  Further improvements can be achieved by addressing data imbalance, "
            "exploring different algorithms and embedding models, and incorporating additional data sources."
        )
        plt.text(0.05, 0.9, conclusion_text, fontsize=12, wrap=True)
        plt.axis('off')
        pdf.savefig()
        plt.close()

if __name__ == "__main__":
    # This code will run when the script is executed directly
    # Example data (replace with your actual data)

    # Create dummy data for demonstration
    dummy_results = {
        'euclidean': {
            1: {'accuracy': 0.8, 'f1_score': 0.79, 'top3_accuracy': 0.95, 'top5_accuracy': 0.98},
            3: {'accuracy': 0.82, 'f1_score': 0.81, 'top3_accuracy': 0.96, 'top5_accuracy': 0.99},
            5: {'accuracy': 0.81, 'f1_score': 0.8, 'top3_accuracy': 0.94, 'top5_accuracy': 0.97}
        },
        'cosine': {
            1: {'accuracy': 0.79, 'f1_score': 0.78, 'top3_accuracy': 0.94, 'top5_accuracy': 0.97},
            3: {'accuracy': 0.83, 'f1_score': 0.82, 'top3_accuracy': 0.95, 'top5_accuracy': 0.98},
            5: {'accuracy': 0.82, 'f1_score': 0.81, 'top3_accuracy': 0.94, 'top5_accuracy': 0.96}
        }
    }

    dummy_optimal_results = {
        'best_metric': 'euclidean',
        'best_k': 3,
        'best_metrics': dummy_results['euclidean'][3]
    }
    dummy_roc_results = {
        'euclidean': {'macro_fpr': np.array([0.0, 0.1, 0.2, 0.5, 1.0]),
                       'macro_tpr': np.array([0.0, 0.4, 0.7, 0.9, 1.0]),
                       'macro_auc': 0.85},
        'cosine': {'macro_fpr': np.array([0.0, 0.15, 0.3, 0.6, 1.0]),
                    'macro_tpr': np.array([0.0, 0.3, 0.6, 0.85, 1.0]),
                    'macro_auc': 0.82}
    }

    dummy_data_stats = {
        'num_syndromes': 5,
        'total_subjects': 50,
        'total_images': 200,
        'avg_images_per_syndrome': 40.0,
        'syndromes': {}  # You could populate this with dummy syndrome data if needed
    }

    dummy_imbalance_info = {
        'min_images': 20,
        'max_images': 60,
        'imbalance_ratio': 3.0,
        'low_representation_syndromes': ['SyndromeA', 'SyndromeB']
    }

    # Sample data for cluster_analysis (replace with your actual analysis)
    dummy_cluster_analysis = {
        "SyndromeA": {
            "num_points": 50,
            "center": np.array([1.0, 2.0]),
            "avg_distance_from_center": 0.5,
            "overlap_with_other_clusters": {"SyndromeB": 0.8, "SyndromeC": 1.2}
        },
        "SyndromeB": {
            "num_points": 60,
            "center": np.array([3.0, 1.5]),
            "avg_distance_from_center": 0.6,
            "overlap_with_other_clusters": {"SyndromeA": 0.8, "SyndromeC": 1.0}
        },
        "SyndromeC": {
            "num_points": 40,
            "center": np.array([0.5, 0.5]),
            "avg_distance_from_center": 0.4,
            "overlap_with_other_clusters": {"SyndromeA": 1.2, "SyndromeB": 1.0}
        }
    }

    dummy_results_table = create_results_table(dummy_results)

    # Create dummy image files (replace with your actual image file paths)
    # These need to actually exist as files
    dummy_tsne_plot_file = "dummy_tsne_plot.png"
    dummy_roc_plot_file = "dummy_roc_plot.png"
    dummy_k_vs_accuracy_plot_file = "dummy_k_vs_accuracy_plot.png"
    dummy_multiple_tsne_plot_file = "dummy_multiple_tsne.png"
    dummy_multiple_metrics_plot = "dummy_multiple_metrics.png"

    # Create some dummy plots for demonstration
    plt.figure()
    plt.plot([1, 2, 3], [4, 5, 6])
    plt.savefig(dummy_tsne_plot_file)
    plt.close()

    plt.figure()
    plt.plot([1, 2, 3], [6, 5, 4])
    plt.savefig(dummy_roc_plot_file)
    plt.close()

    plt.figure()
    plt.plot([1, 3, 5], [0.8, 0.85, 0.82])
    plt.savefig(dummy_k_vs_accuracy_plot_file)
    plt.close()

    plt.figure()
    plt.plot([1, 3, 5, 7], [0.7, 0.8, 0.9, 0.85])
    plt.savefig(dummy_multiple_tsne_plot_file)
    plt.close()

    plt.figure()
    plt.plot([1, 3, 5, 7], [0.85, 0.8, 0.9, 0.75])
    plt.savefig(dummy_multiple_metrics_plot)
    plt.close()



    report_data = {
    'data_stats': dummy_data_stats,
    'imbalance_info': dummy_imbalance_info,
    'knn_results': dummy_results,
    'optimal_results': dummy_optimal_results,
    'roc_results': dummy_roc_results,
    'results_table': dummy_results_table,  # Use the created DataFrame
    'cluster_analysis': dummy_cluster_analysis
}


    generate_report(report_data,
                     dummy_tsne_plot_file,
                     dummy_roc_plot_file,
                     dummy_k_vs_accuracy_plot_file,
                     dummy_multiple_tsne_plot_file,
                     dummy_multiple_metrics_plot,
                     "example_report.pdf")  # Changed to PDF
    print("Generated example_report.pdf") # and PDF here