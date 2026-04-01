import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, hamming_loss, confusion_matrix, precision_recall_curve, auc, precision_recall_fscore_support, log_loss
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sympy.utilities.iterables import multiset_partitions
from sklearn.utils import resample
from scipy.special import softmax
from sklearn.metrics import roc_curve, auc
from sklearn.cluster import KMeans
import io
from sklearn.preprocessing import label_binarize
import json
import csv



# loadd data
undersampled_df = pd.read_csv("undersampled_df.csv")


# update features and target
features = undersampled_df[[col for col in df.columns if col.startswith('Age_class') or col.startswith('Gender')
               or col.startswith('Month')] +symptom]
target = undersampled_df['Pathogen']


resample_pathogen_counts = pd.Series(target).value_counts()
print("Pathogen counts after resampling:")
print(resample_pathogen_counts)



# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)


y_train_counts = pd.Series(y_train).value_counts()
y_test_counts = pd.Series(y_test).value_counts()
print("y_train_counts and y_test_counts after split:")
print(y_train_counts)
print(y_test_counts)
counts_df = pd.DataFrame({'counts raw': [pathogen_counts],
                          'counts before resampling': [class_pathogen_counts],
                          'counts after resampling': [resample_pathogen_counts],
                          'y_train_counts': [y_train_counts],
                          'y_test_counts': [y_test_counts]})
counts_df.to_csv('count.csv', index=False)



# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# X_train_tensor += 0.5 * torch.rand_like(X_train_tensor)

print(torch.min(X_train_tensor, dim = 0)[0], torch.max(X_train_tensor, dim = 0)[0])

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True,num_workers = os.cpu_count() - 1 )
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False,num_workers = os.cpu_count() - 1 )
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False,num_workers = os.cpu_count() - 1 )

# Define PyTorch model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1024, 1024)
        self.dropout3 = nn.Dropout(0.5)
        # self.fc4 = nn.Linear(1024, 1024)
        # self.dropout4 = nn.Dropout(0.2)
        # self.fc5 = nn.Linear(1024, 1024)
        # self.dropout5 = nn.Dropout(0.2)
        self.output = nn.Linear(1024, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        # x = torch.relu(self.fc4(x))
        # x = self.dropout4(x)
        # x = torch.relu(self.fc5(x))
        # x = self.dropout5(x)
        x = self.output(x)
        return x

# Initialize model, loss, and optimizer
input_size = X_train.shape[1]
output_size = len(np.unique(target))
model = NeuralNetwork(input_size, output_size).to('cuda:0')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay = 0.001)


# Training loop
num_epochs = 100
best_val_loss = float('inf')

train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch =  X_batch.to('cuda:0')
        y_batch = y_batch.to('cuda:0')
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # train_loss = 0.0
    # for X_batch, y_batch in train_loader:
    #     X_batch =  X_batch.to('cuda:0')
    #     y_batch = y_batch.to('cuda:0')
    #     # optimizer.zero_grad()
    #     outputs = model(X_batch)
    #     loss = criterion(outputs, y_batch)
    #     # loss.backward()
    #     # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #     # optimizer.step()
    #     train_loss += loss.item()
    # train_loss /= len(train_loader)

    # model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to('cuda:0')
            y_batch = y_batch.to('cuda:0')
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
    val_loss /= len(val_loader)


    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')

torch.save(model.state_dict(), 'final_model.pth')

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))


# Evaluate training accuracy and AUC
def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstraps=1000, ci_percentile=95):
    """
    Calculate the confidence interval for a metric using bootstrap sampling.
    """
    metrics = []
    for _ in range(n_bootstraps):
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        y_true_resampled = y_true[indices]
        y_pred_resampled = y_pred[indices]
        metric = metric_fn(y_true_resampled, y_pred_resampled)
        metrics.append(metric)
    lower_bound = np.percentile(metrics, (100 - ci_percentile) / 2)
    upper_bound = np.percentile(metrics, 100 - (100 - ci_percentile) / 2)
    return lower_bound, upper_bound


def plot_confusion_matrix(y_true, y_pred, class_labels, dataset_name):
    """
    Plot the confusion matrix as a heatmap.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        class_labels (array-like): Class labels.
        dataset_name (str): The name of the dataset (e.g., "Training Data").
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)

    # Normalize the confusion matrix for better visualization
    # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create the confusion matrix plot
    plt.figure(figsize=(10, 7))
    # sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f"Confusion Matrix - {dataset_name}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{dataset_name.lower().replace(' ', '_')}_confusion_matrix.pdf")
    plt.show()


def evaluate_metrics_per_class(y_true, y_pred, y_prob, class_labels):
    """
    Calculate metrics for each class: accuracy, confusion matrix, AUC, sensitivity, specificity, and their CIs.
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    n_classes = len(class_labels)
    metrics = {}
    total_accuracy = 0
    total_sensitivity = 0
    total_specificity = 0

    for i, cls in enumerate(class_labels):
        # Confusion Matrix values
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FN + FP)

        # Calculate metrics for this class
        accuracy = (TP + TN) / cm.sum()
        total_accuracy += accuracy
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        total_sensitivity += sensitivity
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        total_specificity += specificity

        # Calculate AUC for the class
        y_true_binary = (y_true == cls).astype(int)
        y_score = y_prob[:, i]
        auc = roc_auc_score(y_true_binary, y_score)

        # Bootstrap CIs - recalculate confusion matrix for each resample
        accuracy_ci = bootstrap_ci(y_true, y_pred,
                                   lambda y_t, y_p: accuracy_score((y_t == cls).astype(int), (y_p == cls).astype(int)))

        # Recalculate TP, FN, FP, TN for each bootstrap resample inside lambda function
        sensitivity_ci = bootstrap_ci(y_true, y_pred,
                                      lambda y_t, y_p: np.sum((y_t == cls) & (y_p == cls)) /
                                                       (np.sum((y_t == cls) & (y_p == cls)) + np.sum(
                                                           (y_t == cls) & (y_p != cls)))
                                      if np.sum((y_t == cls) & (y_p == cls)) + np.sum(
                                          (y_t == cls) & (y_p != cls)) > 0 else 0)

        specificity_ci = bootstrap_ci(y_true, y_pred,
                                      lambda y_t, y_p: np.sum((y_t != cls) & (y_p != cls)) /
                                                       (np.sum((y_t != cls) & (y_p != cls)) + np.sum(
                                                           (y_t != cls) & (y_p == cls)))
                                      if np.sum((y_t != cls) & (y_p != cls)) + np.sum(
                                          (y_t != cls) & (y_p == cls)) > 0 else 0)

        auc_ci = bootstrap_ci(y_true_binary, y_score, lambda y_t, y_p: roc_auc_score(y_t, y_p))

        # Store all metrics for this class in a dictionary
        cls_metrics = {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'auc': auc,
            'accuracy_ci': accuracy_ci,
            'sensitivity_ci': sensitivity_ci,
            'specificity_ci': specificity_ci,
            'auc_ci': auc_ci
        }

        metrics[cls] = cls_metrics

    # Calculate average across all classes
    avg_accuracy = total_accuracy / n_classes
    metrics['average_accuracy'] = avg_accuracy
    avg_sensitivity = total_sensitivity / n_classes
    metrics['average_sensitivity'] = avg_sensitivity
    avg_specificity = total_specificity / n_classes
    metrics['average_specificity'] = avg_specificity
    return metrics


def plot_roc_curves(y_true, y_prob, class_labels, dataset_name):
    """
    Plot ROC curves for each class.
    """
    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(class_labels):
        y_true_binary = (y_true == cls).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
        auc = roc_auc_score(y_true_binary, y_prob[:, i])
        plt.plot(fpr, tpr, label=f"Class {cls} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title(f"ROC Curves - {dataset_name}", fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"roc_curves_{dataset_name.lower()}.pdf")
    plt.show()


def save_metrics_to_file(metrics, dataset_name, output_format="csv"):
    """
    Save metrics and confidence intervals to a file.

    Args:
        metrics (dict): The metrics dictionary for each class.
        dataset_name (str): The name of the dataset (e.g., "Training Data").
        output_format (str): Format for saving the file ("csv" or "json").
    """
    if output_format == "csv":
        file_name = f"{dataset_name.lower().replace(' ', '_')}_metrics.csv"
        with open(file_name, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Class", "Metric", "Value"])
            for cls, cls_metrics in metrics.items():
                if cls in ['average_accuracy', 'average_sensitivity', 'average_specificity']:
                    writer.writerow([cls, cls, cls_metrics])
                else:
                    for metric_name, value in cls_metrics.items():
                        if isinstance(value, (list, tuple)):  # For confidence intervals
                            writer.writerow([cls, f"{metric_name} Lower CI", value[0]])
                            writer.writerow([cls, f"{metric_name} Upper CI", value[1]])
                        else:
                            writer.writerow([cls, metric_name, value])

        print(f"Metrics saved to {file_name}.")
    elif output_format == "json":
        file_name = f"{dataset_name.lower().replace(' ', '_')}_metrics.json"
        with open(file_name, mode="w") as file:
            json.dump(metrics, file, indent=4)
        print(f"Metrics saved to {file_name}.")
    else:
        print(f"Unsupported output format: {output_format}. Please use 'csv' or 'json'.")


# Evaluate metrics for training and test data
def evaluate_and_plot(model, data_loader, class_labels, dataset_name, save_format="csv"):
    """
    Evaluate metrics, plot ROC curves and confusion matrix for each class.
    Also save metrics to a file.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to("cuda:0"), labels.to("cuda:0")
            outputs = model(data)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Evaluate metrics for each class
    metrics = evaluate_metrics_per_class(all_labels, all_preds, all_probs, class_labels)

    # Save metrics to file
    save_metrics_to_file(metrics, dataset_name, output_format=save_format)

    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, class_labels, dataset_name)

    # Plot ROC curves
    plot_roc_curves(all_labels, all_probs, class_labels, dataset_name)

    return metrics

def save_test_prob(model, data_loader, class_labels, dataset_name, save_format="csv"):
    """
    Evaluate metrics, plot ROC curves and confusion matrix for each class.
    Also save metrics to a file.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to("cuda:0"), labels.to("cuda:0")
            outputs = model(data)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    print(all_labels.shape, all_labels[:10])
    print(all_preds.shape, all_preds[:10])
    print(all_probs.shape, all_probs[:10])
    print(X_test.index[:10])



    prob_df = pd.DataFrame(
        all_probs,
        columns=[f'Class_{i}_Probability' for i in range(3)]
    )

    index_df = pd.DataFrame({
        'Index': X_test.index
    })



    result_df = pd.concat([
        index_df,
        prob_df,
        X_test.reset_index(drop=True), 
        y_test.reset_index(drop=True).rename('True_3Class'), 
        df.loc[X_test.index, 'Pathogen_9class'].reset_index(drop=True),  
        df_raw.loc[X_test.index, 'Date'].reset_index(drop=True),
        df_raw.loc[X_test.index, 'ID'].reset_index(drop=True),
        df_raw.loc[X_test.index, 'Anonymous_ID'].reset_index(drop=True)
    ], axis=1)



    result_df.to_csv('prediction_results_with_original_labels.csv', index=False)



# Evaluate and save metrics for training and test data
class_labels = np.unique(target)

# For Training Data
train_metrics = evaluate_and_plot(model, train_loader, class_labels, "Training Data", save_format="csv")

# For Test Data
test_metrics = evaluate_and_plot(model, test_loader, class_labels, "Test Data", save_format="csv")

save_test_prob(model, test_loader, class_labels, "Test Data", save_format="csv")