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


# load data

symptom=['fever', 'cough', 'sputum', 'myalgia', 'headache', 'fever_over_39',
    'sore_throat', 'runny_nose', 'nasal_congestion', 'dysphonia', 'fever_for_3days',
    'diarrhoea', 'vomiting']



df_raw = pd.read_csv("data for NN.csv")

df = df_raw[(df_raw['Age'] >= 18) & (df_raw['Age'] <= 59)]
df.drop(columns=['Age_class_0','Age_class_1-5','Age_class_6-17','Age_class_≥60'], inplace=True)

df=df[[col for col in df.columns if col.startswith('Age_class') or col.startswith('Gender')
                or col.startswith('Pathogen')]+symptom]

print(df.columns)


pathogen_counts = pd.Series(df['Pathogen']).value_counts()
print("Pathogen counts before pathogen_classification:")
print(pathogen_counts)

def pathogen_classification(Pathogen):
    if  Pathogen == 6 or Pathogen == 0 or Pathogen == 8:
        return 1
    elif Pathogen == 3 or Pathogen == 7 or Pathogen == 1 or Pathogen == 2:
        return 2
    elif Pathogen == 4 or Pathogen == 5  :
        return 0
    else:
        return 'Unknown'

df['Pathogen_9class'] = df['Pathogen']
df['Pathogen'] = df['Pathogen'].apply(pathogen_classification)



target = df['Pathogen']
features = df[[col for col in df.columns if col.startswith('Age_class') or col.startswith('Gender')] +symptom]


class_pathogen_counts = pd.Series(target).value_counts()
print("Pathogen counts before resampling:")
print(class_pathogen_counts)


# undersampled

class0_data = df[target == 0]
class1_data = df[target == 1]
class2_data = df[target == 2]

target_size = len(class2_data)


# def calculate_wcss(X, max_clusters=15):
#     wcss = []
#     for i in range(1, max_clusters+1):
#         kmeans = KMeans(n_clusters=i, random_state=42)
#         kmeans.fit(X)
#         wcss.append(kmeans.inertia_)  # inertia_ is the WCSS
#     return wcss
#
# # List of classes to iterate over
# classes = [class0_data, class1_data, class2_data]
# class_labels = ['Class 0', 'Class 1', 'Class 2']
#
# # Loop through each class and plot the Elbow graph
# for i, class_data in enumerate(classes):
#     X_class = class_data[features.columns].values
#     wcss_class = calculate_wcss(X_class, max_clusters=15)
#
#     # Plot the Elbow graph for each class
#     plt.figure(figsize=(8, 6))
#     plt.plot(range(1, 16), wcss_class, marker='o', linestyle='-', color='b')
#     plt.title(f'Elbow Method For Optimal K ({class_labels[i]})')
#     plt.xlabel('Number of Clusters')
#     plt.ylabel('WCSS')
#     plt.grid(True)
#     plt.show()



X_class0 = class0_data[features.columns].values
kmeans_class0 = KMeans(n_clusters=8, random_state=42).fit(X_class0)
cluster_centers_class0 = kmeans_class0.cluster_centers_

distances_class0 = []
for i in range(len(X_class0)):
    cluster_label = kmeans_class0.labels_[i]
    center = cluster_centers_class0[cluster_label]
    distance = np.linalg.norm(X_class0[i] - center)
    distances_class0.append(distance)
distances_class0 = np.array(distances_class0)

num_samples_to_keep_per_cluster_class0 = int(target_size / 8)
selected_indices_class0 = []
for cluster_id in range(8):
    cluster_indices = np.where(kmeans_class0.labels_ == cluster_id)[0]
    sorted_indices = np.argsort(distances_class0[cluster_indices])
    selected_indices_class0.extend(cluster_indices[sorted_indices[:num_samples_to_keep_per_cluster_class0]])

class0_undersampled = class0_data.iloc[selected_indices_class0]



X_class1 = class1_data[features.columns].values
kmeans_class1 = KMeans(n_clusters=8, random_state=42).fit(X_class1)
cluster_centers_class1 = kmeans_class1.cluster_centers_

distances_class1 = []
for i in range(len(X_class1)):
    cluster_label = kmeans_class1.labels_[i]
    center = cluster_centers_class1[cluster_label]
    distance = np.linalg.norm(X_class1[i] - center)
    distances_class1.append(distance)
distances_class1 = np.array(distances_class1)

num_samples_to_keep_per_cluster_class1 = int(target_size / 8)
selected_indices_class1 = []
for cluster_id in range(8):
    cluster_indices = np.where(kmeans_class1.labels_ == cluster_id)[0]
    sorted_indices = np.argsort(distances_class1[cluster_indices])
    selected_indices_class1.extend(cluster_indices[sorted_indices[:num_samples_to_keep_per_cluster_class1]])

class1_undersampled = class1_data.iloc[selected_indices_class1]



undersampled_df = pd.concat([class0_undersampled, class1_undersampled, class2_data])


