import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_curve, auc

import reporting
from src import data_processing, visualization, classification, metrics

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to execute the genetic syndrome classification pipeline.
    """

    # --- Configuration ---
    DATA_FILE = "uploads/mini_gm_public_v0.1.p"  # Input data file
    PROCESSED_DATA_FILE = "processed_data.pkl"  # Output file for processed data
    RESULTS_DIR = "results"  # Directory to save results
    TSNE_PLOT_FILE = os.path.join(RESULTS_DIR, "tsne_plot.png")  
    ROC_PLOT_FILE = os.path.join(RESULTS_DIR, "roc_plot.png")  
    K_VS_ACCURACY_PLOT_FILE = os.path.join(
        RESULTS_DIR, "k_vs_accuracy.png") 
    MULTIPLE_METRICS_PLOT = os.path.join(
        RESULTS_DIR, 'multiple_metrics_plot.png')
    REPORT_FILE = os.path.join(RESULTS_DIR, "report.txt")  
    K_VALUES = [1, 3, 5, 7, 9, 11, 13, 15]  # k values for KNN
    METRICS = ['euclidean', 'cosine']  
    N_SPLITS = 10  
    RANDOM_STATE = 42  
    TSNE_PERPLEXITIES = [5, 30, 50, 100]
    MULTIPLE_TSNE_PLOT_FILE = os.path.join(
        RESULTS_DIR, "multiple_tsne_plot.png")
    # --- Setup ---
    reporting.create_directory(RESULTS_DIR)  # Create results directory

    # --- Data Loading and Preprocessing ---
    logging.info("Starting data loading and preprocessing...")
    try:
        if os.path.exists(PROCESSED_DATA_FILE):
            logging.info(
                "Loading preprocessed data from %s", PROCESSED_DATA_FILE)
            with open(PROCESSED_DATA_FILE, 'rb') as f:
                processed_data = pickle.load(f)
            embeddings = processed_data["embeddings"]
            labels = processed_data["labels"]
            syndrome_mapping = processed_data["syndrome_mapping"]
            # Invert the syndrome mapping
            inverse_syndrome_mapping = {
                v: k for k, v in syndrome_mapping.items()}

            # IMPORTANT: Load raw_data here, even if using processed embeddings.
            raw_data = data_processing.load_data(DATA_FILE)

        else:
            raw_data = data_processing.load_data(DATA_FILE)
            flattened_data = data_processing.flatten_data(raw_data)

            # Check for NaNs in embeddings *before* converting to numpy array
            if flattened_data['embedding'].apply(lambda x: np.isnan(x).any()).any():
                raise ValueError("NaN values found in embeddings.")

            embeddings = np.array(flattened_data['embedding'].tolist())
            labels = flattened_data['syndrome_id'].to_numpy()

            # Create syndrome mapping (string labels to numeric)
            unique_syndromes = np.unique(labels)
            syndrome_mapping = {
                syndrome: i for i, syndrome in enumerate(unique_syndromes)}
            # Transform labels to numeric using the mapping
            labels = np.array([syndrome_mapping[label] for label in labels])

            # Invert the syndrome mapping
            inverse_syndrome_mapping = {
                v: k for k, v in syndrome_mapping.items()}

            data_processing.save_processed_data(
                embeddings, labels, syndrome_mapping, PROCESSED_DATA_FILE)
            logging.info(
                "Processed data and saved to %s", PROCESSED_DATA_FILE)

        # --- Exploratory Data Analysis ---
        data_stats = data_processing.get_dataset_statistics(raw_data)
        imbalance_info = data_processing.check_data_imbalance(data_stats)

        logging.info("Dataset Statistics:")
        logging.info("  Number of Syndromes: %d", data_stats['num_syndromes'])
        logging.info("  Total Subjects: %d", data_stats['total_subjects'])
        logging.info("  Total Images: %d", data_stats['total_images'])
        logging.info(
            "  Average Images per Syndrome: %.2f",
            data_stats['avg_images_per_syndrome'])
        logging.info("  Imbalance Ratio: %.2f", imbalance_info['imbalance_ratio'])
        logging.info(
            "  Low Representation Syndromes: %s",
            imbalance_info['low_representation_syndromes'])

        # --- Data Visualization (t-SNE) ---
        logging.info("Performing t-SNE visualization...")
        reduced_embeddings = visualization.perform_tsne(embeddings)
        visualization.plot_tsne_embeddings(
            reduced_embeddings, labels, syndrome_mapping, TSNE_PLOT_FILE)

        # Analyze clusters
        cluster_analysis = visualization.analyze_clusters(
            reduced_embeddings, labels, syndrome_mapping)
        logging.info("Cluster Analysis:")
        for syndrome, analysis in cluster_analysis.items():
            logging.info("  Syndrome: %s", syndrome)
            logging.info(
                "    Number of Points: %d",
                analysis['num_points'])
            logging.info(
                "    Average Distance from Center: %.2f",
                analysis['avg_distance_from_center'])
            logging.info("    Overlap with Other Clusters:")
            for other_syndrome, overlap in analysis['overlap_with_other_clusters'].items():
                logging.info("      %s: %.2f", other_syndrome, overlap)
    except (FileNotFoundError, ValueError, TypeError, KeyError) as e:
        logging.error("Error during data loading/preprocessing: %s", e)
        return  # Exit if data loading fails

        # --- Classification ---
    logging.info("Starting classification task...")
    try:
        knn_results = classification.evaluate_knn_with_cross_validation(
            embeddings, labels, k_values=K_VALUES, metrics=METRICS,
            n_splits=N_SPLITS, random_state=RANDOM_STATE
        )

        # Find and print optimal k and metric
        optimal_results = classification.find_optimal_k(knn_results)
        logging.info("Optimal k and Metric:")
        logging.info("  Metric: %s", optimal_results['best_metric'])
        logging.info("  k: %d", optimal_results['best_k'])
        logging.info(
            "  Accuracy: %.4f", optimal_results['best_metrics']['accuracy'])
        logging.info(
            "  F1 Score: %.4f", optimal_results['best_metrics']['f1_score'])
        logging.info(
            "  Top-3 Accuracy: %.4f",
            optimal_results['best_metrics']['top3_accuracy'])
        logging.info(
            "  Top-5 Accuracy: %.4f",
            optimal_results['best_metrics']['top5_accuracy'])

        # --- Metrics and Evaluation ---
        # Create results table
        results_table = reporting.create_results_table(knn_results)
        logging.info("Results Table:\n%s", results_table)
        # Plot k vs accuracy
        reporting.plot_k_vs_accuracy(knn_results, K_VS_ACCURACY_PLOT_FILE)

        reporting.plot_multiple_metrics(knn_results, MULTIPLE_METRICS_PLOT)

        # ROC curves calculation and plotting
        logging.info("Calculating and plotting ROC curves...")

        roc_results = {}
        for metric in METRICS:
            all_fpr = []
            all_tpr = []
            all_auc = []

            kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
            for fold, (train_idx, test_idx) in enumerate(kf.split(embeddings)):
                X_train, X_test = embeddings[train_idx], embeddings[test_idx]
                y_train, y_test = labels[train_idx], labels[test_idx]

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                knn = classification.KNNClassifier(k=optimal_results['best_k'], metric=metric)
                knn.fit(X_train, y_train)
                probabilities = knn.predict_proba(X_test)

                # Use try-except within the loop to catch issues in specific folds.
                try:
                    macro_auc, fold_roc_result = metrics.calculate_roc_auc(y_test, probabilities)
                    all_fpr.append(fold_roc_result['macro_fpr'])
                    all_tpr.append(fold_roc_result['macro_tpr'])
                    all_auc.append(fold_roc_result['macro_auc'])

                except ValueError as e:
                    logging.warning(f"ValueError in fold {fold+1} for metric {metric}: {e}. Skipping this fold for ROC.")
                    # Don't append anything to all_fpr, all_tpr, all_auc in this case.
                    continue # Skip to the next fold
                except Exception as e: # Catch any OTHER exception
                    logging.error(f"Unexpected error in fold {fold+1} for metric {metric}: {e}. Skipping this fold.")
                    continue


            # --- Averaging (Handle potential empty lists) ---
            if all_fpr and all_tpr and all_auc:  # Only average if we have data
                # Convert lists of arrays to a 2D array for consistent interpolation.
                # Find the maximum length of fpr arrays across all folds
                max_len = max(len(fpr) for fpr in all_fpr)

                # Pad the fpr and tpr arrays with NaNs to make them all the same length
                all_fpr_padded = [np.pad(fpr, (0, max_len - len(fpr)), 'constant', constant_values=np.nan) for fpr in all_fpr]
                all_tpr_padded = [np.pad(tpr, (0, max_len - len(tpr)), 'constant', constant_values=np.nan) for tpr in all_tpr]

                # Convert lists to 2D numpy arrays
                all_fpr_array = np.array(all_fpr_padded)
                all_tpr_array = np.array(all_tpr_padded)

                # Use nanmean to ignore NaNs during averaging
                mean_fpr = np.nanmean(all_fpr_array, axis=0)
                mean_tpr = np.nanmean(all_tpr_array, axis=0)
                mean_auc = np.mean(all_auc) # all_auc should be a list of scalars

                # Remove NaN values from mean_fpr and mean_tpr before plotting
                nan_mask = ~np.isnan(mean_fpr)  # Create a mask of non-NaN values
                mean_fpr = mean_fpr[nan_mask]
                mean_tpr = mean_tpr[nan_mask]

                roc_results[metric] = {
                    'macro_fpr': mean_fpr,
                    'macro_tpr': mean_tpr,
                    'macro_auc': mean_auc,
                }
            else:
                logging.warning(f"No valid ROC data for metric {metric}. Skipping ROC plot for this metric.")
                roc_results[metric] = { # Set placeholder values
                    'macro_fpr': np.array([0, 1]),
                    'macro_tpr': np.array([0, 1]),
                    'macro_auc': 0.0,  # Or np.nan, if you prefer
                }

        reporting.plot_roc_curves(roc_results, ROC_PLOT_FILE)

        # Confusion Matrix (for best k and metric)
        best_metric = optimal_results['best_metric']
        best_k = optimal_results['best_k']
        #  Re-train on the *entire* dataset
        scaler = StandardScaler()
        all_embeddings_scaled = scaler.fit_transform(
            embeddings)  # Scale the ENTIRE dataset

        knn = classification.KNNClassifier(k=best_k, metric=best_metric)
        knn.fit(all_embeddings_scaled, labels)
        all_predictions = knn.predict(
            all_embeddings_scaled)  # Predict on the ENTIRE dataset

        conf_matrix = metrics.calculate_confusion_matrix(
            labels, all_predictions)
        reporting.plot_confusion_matrix(
            conf_matrix, list(
                inverse_syndrome_mapping.values()), os.path.join(
                RESULTS_DIR, 'confusion_matrix.png'))

        # Multiple t-SNE perplexities
        visualization.plot_multiple_tsne_perplexities(
            embeddings, labels, perplexities=TSNE_PERPLEXITIES, output_file=MULTIPLE_TSNE_PLOT_FILE)

    except Exception as e:
        logging.exception(
            "An unexpected error occurred during classification or evaluation: %s", e)


if __name__ == "__main__":
    main()