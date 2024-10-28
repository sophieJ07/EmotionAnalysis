import numpy as np
import matplotlib.pyplot as plt
from train import model, X_test, y_test, y_pred

# Get the decision scores (distance from decision boundaries)
decision_scores = model.decision_function(X_test)

# Correct and incorrect indices
correct_indices = np.where(y_pred == y_test)[0]
incorrect_indices = np.where(y_pred != y_test)[0]
jitter = 0.05

# Plot max scores of correct and incorrect predictions
max_correct = np.max(decision_scores[correct_indices], axis=1)
max_incorrect = np.max(decision_scores[incorrect_indices], axis=1)

plt.figure(figsize=(8, 2))
plt.scatter(max_correct, np.random.uniform(-jitter, jitter, len(max_correct)), label='Correct', color='blue', marker='o')
plt.scatter(max_incorrect, np.random.uniform(-jitter, jitter, len(max_incorrect)), label='Incorrect', color='red', marker='x')
plt.title('Max Decision Scores of Correct vs Incorrect Predictions')
plt.xlabel('Max Decision Score')
plt.yticks([])
plt.show()

# Plot variances of scores of correct and incorrect predictions
var_correct = np.var(decision_scores[correct_indices], axis=1)
var_incorrect = np.var(decision_scores[incorrect_indices], axis=1)

plt.figure(figsize=(8, 2))
plt.scatter(var_correct, np.random.uniform(-jitter, jitter, len(var_correct)), label='Correct', color='blue', marker='o')
plt.scatter(var_incorrect, np.random.uniform(-jitter, jitter, len(var_incorrect)), label='Incorrect', color='red', marker='x')
plt.title('Variance of Decision Scores of Correct vs Incorrect Predictions')
plt.xlabel('Score Variance')
plt.yticks([])
plt.show()

# Plot highest score - 2nd highest score of correct and incorrect predictions
top_2_diff_correct = np.partition(-decision_scores[correct_indices], 1, axis=1)
top_2_diff_incorrect = np.partition(-decision_scores[incorrect_indices], 1, axis=1)

diff_correct = - (top_2_diff_correct[:, 0] - top_2_diff_correct[:, 1])
diff_incorrect = - (top_2_diff_incorrect[:, 0] - top_2_diff_incorrect[:, 1])

plt.figure(figsize=(8, 2))
plt.scatter(diff_correct, np.random.uniform(-jitter, jitter, len(diff_correct)), label='Correct', color='blue', marker='o')
plt.scatter(diff_incorrect, np.random.uniform(-jitter, jitter, len(diff_incorrect)), label='Incorrect', color='red', marker='x')
plt.title('Highest - 2nd Highest Decision Scores of Correct vs Incorrect Predictions')
plt.xlabel('Score Difference (Highest - 2nd Highest)')
plt.yticks([])
plt.show()