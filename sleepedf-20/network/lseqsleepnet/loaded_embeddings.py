import os
import sys
import h5py
import hdf5storage
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score, f1_score
from _globals import *

validation_embeddings_file = os.path.join(HPC_STORAGE_PATH, "results_lseqsleepnet/outputs/train_test/",
                                          'validation_embeddings.npy')
test_embeddings_file = os.path.join(HPC_STORAGE_PATH, "results_lseqsleepnet/outputs/train_test/", 'test_embeddings.npy')
validation_labels_file = os.path.join(HPC_STORAGE_PATH, "results_lseqsleepnet/outputs/train_test/",
                                      'validation_labels.npy')
test_labels_file = os.path.join(HPC_STORAGE_PATH, "results_lseqsleepnet/outputs/train_test/", 'test_labels.npy')

validation_embeddings = np.load(validation_embeddings_file)
test_embeddings = np.load(test_embeddings_file)
validation_labels = np.load(validation_labels_file)
test_labels = np.load(test_labels_file)

out_path = os.path.join(HPC_STORAGE_PATH, "results_lseqsleepnet/outputs/train_test/")


def fit_gmm(embeddings, n_components):
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
    gmm.fit(embeddings)
    return gmm


# Plot ROC curve
def plot_roc(labels, scores, out_dir):
    fpr, tpr, _ = roc_curve(labels == 3, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')

    # Save the plot
    plot_file = os.path.join(out_dir, 'roc_curve.png')
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    plt.savefig(plot_file)
    plt.show()


# Fit GMM on clean validation embeddings
gmm = fit_gmm(validation_embeddings, n_components=3)
# plot_density(gmm, validation_embeddings, out_path)

# Compute the probability scores for ROC-AUC
scores = -gmm.score_samples(test_embeddings)

# Plot ROC curve and compute AUC
plot_roc(test_labels, scores, out_path)

# Generate binary labels for artifacts (1) and non-artifacts (0)
binary_labels = (test_labels == 3).astype(int)

# Predict binary labels based on scores
threshold = np.median(scores)
binary_predictions = (scores >= threshold).astype(int)

# Check distribution of test labels and predictions
num_artifacts = np.sum(binary_labels == 1)
num_non_artifacts = np.sum(binary_labels == 0)
print(f"Number of artifacts in test labels: {num_artifacts}")
print(f"Number of non-artifacts in test labels: {num_non_artifacts}")

# Compute, plot and save the confusion matrix
cm = confusion_matrix(binary_labels, binary_predictions)
cm_normalized = cm.astype(float)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=['Non-artifact.', 'Artifact'])
fig, ax = plt.subplots(figsize=(10,10))
disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='.2f')
plt.title('Confusion Matrix: Artifacts vs Non-artifacts')
plt.savefig(os.path.join(out_path, 'confusion_matrix_embeddings.png'))
plt.show()

# Compute and print the metrics
precision = precision_score(binary_labels, binary_predictions, average=None)
recall = recall_score(binary_labels, binary_predictions, average=None)
accuracy = accuracy_score(binary_labels, binary_predictions)
f1 = f1_score(binary_labels, binary_predictions, average=None)

metrics_file = os.path.join(out_path, 'metrics.txt')

with open(metrics_file, 'w') as f:
    f.write(f"Precision (Non-artifact): {precision[0]:.3f}\n")
    f.write(f"Precision (Artifact): {precision[1]:.3f}\n")
    f.write(f"Recall (Non-artifact): {recall[0]:.3f}\n")
    f.write(f"Recall (Artifact): {recall[1]:.3f}\n")
    f.write(f"F1 Score (Non-artifact): {f1[0]:.3f}\n")
    f.write(f"F1 Score (Artifact): {f1[1]:.3f}\n")
    f.write(f"Accuracy: {accuracy:.3f}\n")
    f.write(f"Number of artifacts in test labels: {num_artifacts}\n")
    f.write(f"Number of non-artifacts in test labels: {num_non_artifacts}\n")

print(f"Precision (Non-artifact): {precision[0]:.3f}")
print(f"Precision (Artifact): {precision[1]:.3f}")
print(f"Recall (Non-artifact): {recall[0]:.3f}")
print(f"Recall (Artifact): {recall[1]:.3f}")
print(f"F1 Score (Non-artifact): {f1[0]:.3f}")
print(f"F1 Score (Artifact): {f1[1]:.3f}")
print(f"Accuracy: {accuracy:.3f}")