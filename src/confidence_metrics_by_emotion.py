import numpy as np
import matplotlib.pyplot as plt
from train import model, X_test, y_test, y_pred
from train import emotion_labels

num_classes = len(emotion_labels)

# Define functions to calculate each confidence metric
def calculate_mean_max_scores(decision_scores, class_index):
    indices = np.where(y_pred == class_index)[0]
    class_scores = decision_scores[indices]
    max_scores = np.max(class_scores, axis=1)
    return np.mean(max_scores)

def calculate_variance_scores(decision_scores, class_index):
    indices = np.where(y_pred == class_index)[0]
    class_scores = decision_scores[indices]
    return np.mean(np.var(class_scores, axis=1))

def calculate_highest_2nd_highest_diff(decision_scores, class_index):
    indices = np.where(y_pred == class_index)[0]
    class_scores = decision_scores[indices]
    top_2_diff = np.partition(-class_scores, 1, axis=1)
    diff = - (top_2_diff[:, 0] - top_2_diff[:, 1])
    return np.mean(diff)

# Plot each metric 
def plot_metrics(mean_max_scores, variance_scores, highest_2nd_highest_diff):
    bar_width = 0.2
    index = np.arange(num_classes)

    plt.figure(figsize=(10, 6))

    plt.bar(index, mean_max_scores, bar_width, label='Mean of Max Scores')
    plt.bar(index + 2 * bar_width, highest_2nd_highest_diff, bar_width, label='Highest - 2nd Highest Scores')
    plt.bar(index + bar_width, variance_scores, bar_width, label='Variance of Scores')

    plt.xlabel('Emotion Categories')
    plt.ylabel('Scores')
    plt.title('Decision Score Metrics for Emotion Categories')
    plt.xticks(index + bar_width, emotion_labels)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    decision_scores = model.decision_function(X_test)

    mean_max_scores = []
    variance_scores = []
    highest_2nd_highest_diff = []

    # Run through all emotion categories
    for i in range(num_classes):
        mean_max_scores.append(calculate_mean_max_scores(decision_scores, i))
        variance_scores.append(calculate_variance_scores(decision_scores, i))
        highest_2nd_highest_diff.append(calculate_highest_2nd_highest_diff(decision_scores, i))

    for i in range(num_classes):
        print(f"{emotion_labels[i]}:")
        print(f"  Mean of Max Scores: {mean_max_scores[i]}")
        print(f"  Variance of Scores: {variance_scores[i]}")
        print(f"  Mean of (Highest - 2nd Highest Scores): {highest_2nd_highest_diff[i]}")
        
    plot_metrics(mean_max_scores, variance_scores, highest_2nd_highest_diff)