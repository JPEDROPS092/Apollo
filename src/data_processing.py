import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path: str) -> Dict:
    """
    Load data from pickle file

    Args:
        file_path: Path to the pickle file

    Returns:
        Dict: Loaded data
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logging.info(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def flatten_data(data: Dict) -> pd.DataFrame:
    """
    Flattens the hierarchical data structure into a Pandas DataFrame.

    Args:
        data: Hierarchical data structure.

    Returns:
        pd.DataFrame: Flattened data in a DataFrame.
    """
    embeddings = []
    labels = []
    subject_ids = []  # Store subject IDs
    image_ids = [] # Store image IDs

    # Iterate through the hierarchical structure
    for syndrome_id, subjects in data.items():
        for subject_id, images in subjects.items():
            for image_id, embedding in images.items():
                embeddings.append(embedding)
                labels.append(syndrome_id)
                subject_ids.append(subject_id)
                image_ids.append(image_id)


    df = pd.DataFrame({
        'syndrome_id': labels,
        'subject_id': subject_ids,
        'image_id': image_ids,
        'embedding': embeddings
    })
    return df

def get_dataset_statistics(data: Dict) -> Dict:
    """
    Get statistics about the dataset

    Args:
        data: Hierarchical data structure

    Returns:
        Dict: Dictionary containing dataset statistics
    """
    stats = {
        "num_syndromes": len(data.keys()),
        "syndromes": {},
        "total_subjects": 0,
        "total_images": 0
    }

    for syndrome_id, subjects in data.items():
        num_subjects = len(subjects.keys())
        stats["syndromes"][syndrome_id] = {
            "num_subjects": num_subjects,
            "num_images": 0
        }

        for subject_id, images in subjects.items():
            num_images = len(images.keys())
            stats["syndromes"][syndrome_id]["num_images"] += num_images

        stats["total_subjects"] += num_subjects
        stats["total_images"] += stats["syndromes"][syndrome_id]["num_images"]

    # Calculate average images per syndrome and per subject
    for syndrome_id in stats["syndromes"]:
        if stats["syndromes"][syndrome_id]["num_subjects"] > 0:
            stats["syndromes"][syndrome_id]["avg_images_per_subject"] = (
                stats["syndromes"][syndrome_id]["num_images"] /
                stats["syndromes"][syndrome_id]["num_subjects"]
            )

    stats["avg_subjects_per_syndrome"] = stats["total_subjects"] / stats["num_syndromes"]
    stats["avg_images_per_syndrome"] = stats["total_images"] / stats["num_syndromes"]
    stats["avg_images_per_subject"] = stats["total_images"] / stats["total_subjects"]

    return stats

def check_data_imbalance(stats: Dict) -> Dict:
    """
    Check for data imbalances

    Args:
        stats: Dictionary containing dataset statistics

    Returns:
        Dict: Information about data imbalances
    """
    imbalance_info = {}

    # Get number of images per syndrome
    images_per_syndrome = {syndrome_id: stats["syndromes"][syndrome_id]["num_images"]
                           for syndrome_id in stats["syndromes"]}

    min_images = min(images_per_syndrome.values())
    max_images = max(images_per_syndrome.values())

    imbalance_info["min_images"] = min_images
    imbalance_info["max_images"] = max_images
    imbalance_info["imbalance_ratio"] = max_images / min_images if min_images > 0 else float('inf')

    # Identify syndromes with low representation
    low_representation_threshold = stats["avg_images_per_syndrome"] * 0.5
    imbalance_info["low_representation_syndromes"] = [
        syndrome_id for syndrome_id, num_images in images_per_syndrome.items()
        if num_images < low_representation_threshold
    ]

    return imbalance_info

def get_data_split_indices(n_samples: int, n_splits: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate indices for k-fold cross-validation

    Args:
        n_samples: Number of samples in the dataset
        n_splits: Number of splits for cross-validation

    Returns:
        List of tuples containing train and test indices
    """
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    fold_size = n_samples // n_splits
    splits = []

    for i in range(n_splits):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_splits - 1 else n_samples

        test_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])

        splits.append((train_indices, test_indices))

    return splits

def save_processed_data(embeddings: np.ndarray, labels: np.ndarray,
                        syndrome_mapping: Dict, output_file: str) -> None:
    """
    Save processed data to a pickle file

    Args:
        embeddings: Array of embeddings
        labels: Array of labels
        syndrome_mapping: Mapping of syndrome_id to numeric label
        output_file: Path to save the processed data
    """
    processed_data = {
        "embeddings": embeddings,
        "labels": labels,
        "syndrome_mapping": syndrome_mapping
    }

    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)

    logging.info(f"Processed data saved to {output_file}")

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a pickle file and preprocesses it into a flattened DataFrame.

    Args:
        file_path: Path to the pickle file.

    Returns:
        pd.DataFrame: The flattened data.
    """
    data = load_data(file_path)
    df = flatten_data(data)
    return df

if __name__ == "__main__":
    
    file_path = "uploads/mini_gm_public_v0.1.p"  #  Correct path
    try:
        data = load_data(file_path)
        df = flatten_data(data)
        print(df.head())

        # Example of using other functions
        stats = get_dataset_statistics(data)
        imbalance_info = check_data_imbalance(stats)

        print(f"Dataset has {stats['num_syndromes']} syndromes")
        print(f"Total subjects: {stats['total_subjects']}")
        print(f"Total images: {stats['total_images']}")
        print(f"Average images per syndrome: {stats['avg_images_per_syndrome']:.2f}")
        print(f"Imbalance ratio: {imbalance_info['imbalance_ratio']:.2f}")
        print(f"Low representation syndromes: {imbalance_info['low_representation_syndromes']}")

         # Example usage with a dummy output file
        dummy_output_file = "processed_data_dummy.pkl"

        # Create dummy data for saving
        embeddings_dummy = np.array([[1, 2, 3], [4, 5, 6]])
        labels_dummy = np.array([0, 1])
        syndrome_mapping_dummy = {"syndrome_A": 0, "syndrome_B": 1}

        # Save processed data using the function
        save_processed_data(embeddings_dummy, labels_dummy, syndrome_mapping_dummy, dummy_output_file)
        print(f"Dummy processed data saved to {dummy_output_file}")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}.  Make sure the file exists.")
    except Exception as e:
        print(f"An error occurred: {e}")