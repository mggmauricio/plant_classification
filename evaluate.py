# evaluate.py

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm

import config
from model import create_model
from dataset import get_dataloaders

def get_predictions(model, data_loader, device):
    """
    Runs inference on the entire dataset and returns all predictions and true labels.

    Args:
        model (torch.nn.Module): The trained model.
        data_loader (DataLoader): The DataLoader for the evaluation set.
        device (torch.device): The device to run inference on (e.g., 'cuda' or 'cpu').

    Returns:
        tuple: A tuple containing two numpy arrays (predictions, true_labels).
    """
    model.eval()
    all_preds = torch.tensor([], device=device)
    all_labels = torch.tensor([], device=device)

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating")
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds = torch.cat((all_preds, preds), dim=0)
            all_labels = torch.cat((all_labels, labels), dim=0)

    return all_preds.cpu().numpy(), all_labels.cpu().numpy()

def save_confusion_matrix_plot(y_true, y_pred, class_names):
    """
    Plots and saves the confusion matrix as a .png file.

    Args:
        y_true (np.ndarray): Array of true labels.
        y_pred (np.ndarray): Array of predicted labels.
        class_names (list): List of class names for plot labels.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(18, 15))
    heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                          xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    figure_path = 'confusion_matrix.png'
    plt.savefig(figure_path)
    plt.close()
    print(f"Confusion matrix plot saved as '{figure_path}'")

def save_evaluation_report(y_true, y_pred, class_names):
    """
    Calculates statistics and saves them to a text file.

    Args:
        y_true (np.ndarray): Array of true labels.
        y_pred (np.ndarray): Array of predicted labels.
        class_names (list): List of class names for the report.
    """
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    report_str = f"--- Performance Statistics ---\n\n"
    report_str += f"Overall Accuracy: {accuracy*100:.2f}%\n\n"
    report_str += "Classification Report (per class):\n"
    report_str += report

    report_filename = "evaluation_report.txt"
    with open(report_filename, 'w') as f:
        f.write(report_str)

    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"\nClassification report saved as '{report_filename}'")


def main():
    """Main function to run the model evaluation."""

    try:
        with open('class_names.json', 'r') as f:
            class_names = json.load(f)
    except FileNotFoundError:
        print("Error: 'class_names.json' not found.")
        print("Please run the training script 'main.py' first.")
        return

    _, val_loader, _ = get_dataloaders(
        config.DATA_DIR, config.IMAGE_SIZE, config.BATCH_SIZE
    )

    model = create_model(len(class_names))
    try:
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    except FileNotFoundError:
        print(f"Error: Trained model not found at '{config.MODEL_PATH}'.")
        print("Please run the training script 'main.py' first.")
        return
    model.to(config.DEVICE)

    y_pred, y_true = get_predictions(model, val_loader, config.DEVICE)

    save_evaluation_report(y_true, y_pred, class_names)
    save_confusion_matrix_plot(y_true, y_pred, class_names)


if __name__ == '__main__':
    main()
