import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import logging
import os
import pickle
import pandas as pd

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
    Plot ROC curves
    
    Args:
        roc_results: Dictionary containing ROC results
        output_file: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curves for each metric
    for metric, results in roc_results.items():
        plt.plot(
            results['macro_fpr'],
            results['macro_tpr'],
            label=f"{metric} (AUC = {results['macro_auc']:.3f})",
            lw=2
        )
    
    # Plot random chance line
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
    
    plt.show()

def plot_k_vs_accuracy(results: Dict, output_file: str = None) -> None:
    """
    Plot k vs accuracy
    
    Args:
        results: Dictionary containing results for different k values
        output_file: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Plot k vs accuracy for each metric
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
    
    plt.show()

def create_results_table(results: Dict) -> pd.DataFrame:
    """
    Create a table with results for different k values and metrics
    
    Args:
        results: Dictionary containing results
        
    Returns:
        DataFrame with results
    """
    table_data = []
    
    for metric, k_results in results.items():
        for k, metrics in k_results.items():
            row = {
                'Metric': metric,
                'k': k,
                'Accuracy': metrics['accuracy'],
                'F1 Score': metrics['f1_score'],
                'Top-3 Accuracy': metrics['top3_accuracy'],
                'Top-5 Accuracy': metrics['top5_accuracy']
            }
            table_data.append(row)
    
    df = pd.DataFrame(table_data)
    return df

def save_results(results: Dict, filename: str) -> None:
    """
    Save results to pickle file
    
    Args:
        results: Dictionary containing results
        filename: Path to save the results
    """
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    
    logging.info(f"Results saved to {filename}")

def load_results(filename: str) -> Dict:
    """
    Load results from pickle file
    
    Args:
        filename: Path to load the results from
        
    Returns:
        Dictionary containing results
    """
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    
    logging.info(f"Results loaded from {filename}")
    return results

def plot_confusion_matrix(confusion_matrix: np.ndarray, classes: List[str], 
                           output_file: str = None) -> None:
    """
    Plot confusion matrix
    
    Args:
        confusion_matrix: Confusion matrix
        classes: List of class names
        output_file: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = confusion_matrix.max() / 2.0
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
    
    plt.show()

def plot_multiple_metrics(results: Dict, output_file: str = None) -> None:
    """
    Plot multiple metrics
    
    Args:
        results: Dictionary containing results
        output_file: Path to save the plot
    """
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
    
    plt.show()

if __name__ == "__main__":
    # This code will run when the script is executed directly
    # Example data
    results = {
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
    
    # Create and display results table
    df = create_results_table(results)
    print(df)
    
    # Plot k vs accuracy
    plot_k_vs_accuracy(results)
    
    # Plot multiple metrics
    plot_multiple_metrics(results)