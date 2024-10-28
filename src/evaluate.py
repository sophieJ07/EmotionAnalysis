import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
    matthews_corrcoef, precision_recall_curve, auc, roc_curve, roc_auc_score)
from train import model, X_test, y_test
from train import emotion_labels

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Overall Accuracy
overall_accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {overall_accuracy}")

# Balanced Accuracy
balanced_acc = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced Accuracy: {balanced_acc}")

# Cohen's Kappa
cohen_kappa = cohen_kappa_score(y_test, y_pred)
print(f"Cohen's Kappa: {cohen_kappa}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")

# Visualize Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Per class accuracy & MCC
for i, emotion in enumerate(emotion_labels):
    # True Positives
    TP = conf_matrix[i, i]

    # True Negatives
    TN = np.sum(np.diag(conf_matrix)) - TP

    # False Positives
    FP = np.sum(conf_matrix[:, i]) - TP

    # False Negatives
    FN = np.sum(conf_matrix[i, :]) - TP

    # Calculate per-class accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print(f"Accuracy for {emotion}: {accuracy:.2f}")

    mcc = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    print(f"MCC for {emotion}: {mcc:.2f}")


# Matthews Correlation Coefficient (MCC) 
mcc_overall = matthews_corrcoef(y_test, y_pred)
print(f"Overall MCC: {mcc_overall}")

# Decision Boundary Distance
decision_function = model.decision_function(X_test)
distances = np.abs(decision_function)

# Confidence levels (distance to decision boundary)
mean_distance = np.mean(distances)
print(f"Mean Distance to Decision Boundary (Confidence Level): {mean_distance}")

# Precision-Recall Curve and Area Under Curve 
plt.figure(figsize=(12, 8))
for i, emotion in enumerate(emotion_labels):
    y_true = (y_test == i).astype(int)
    y_scores = decision_function[:, i] if model.classes_[0] == 0 else decision_function[:, i]
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f'{emotion} (PR AUC = {pr_auc:.2f})')

plt.title('Precision-Recall Curves')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.show()

# ROC Curve and Area Under Curve 
plt.figure(figsize=(12, 8))
for i, emotion in enumerate(emotion_labels):
    y_true = (y_test == i).astype(int)
    y_scores = decision_function[:, i] if model.classes_[0] == 0 else decision_function[:, i]
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    plt.plot(fpr, tpr, label=f'{emotion} (ROC AUC = {roc_auc:.2f})')

plt.title('ROC Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()