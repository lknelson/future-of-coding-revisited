# visualize the data
import os
import pdb
import pandas as pd
import matplotlib.pyplot as plt

file = './results/5folds/gemma_final_metrics_summary.csv'

average_results = pd.read_csv(file)
#pdb.set_trace()

# Plotting
plt.figure(figsize=(10, 6))

# Plot each metric as a line
#plt.errorbar(average_results["sample_set"], average_results["weighted_avg_precision"], fmt='o-', label="Precision")
#plt.errorbar(average_results["sample_set"], average_results["weighted_avg_recall"], fmt='o-', label="Recall")
x_positions = [1, 2, 4, 8]
plt.errorbar(x_positions, average_results["weighted_avg_f1_score"], fmt='o-', color = 'darkorange', label="F1 Score")

# Set scale, labels, and title
#plt.xscale('log')
#plt.xlim(8, 2000)
plt.xticks(x_positions, [str(x) for x in average_results["sample_set"]])

plt.xlabel('Fine-tuning Sample Size', fontsize=12)
plt.ylabel('F1', fontsize=12)
plt.title('Model Performance Across Different Fine-tuning Sample Sizes', fontsize=14)

# Add legend
plt.legend(title="Metrics", fontsize=10, loc='lower right')
# Add grid
plt.grid(visible=False, which='both', linestyle='--', linewidth=0.5)
# Display the plot
plt.tight_layout()

# save figure
#pdb.set_trace()
output_folder = "./results/5folds/plots/"
# create folder
os.makedirs(output_folder, exist_ok=True)
output_file = "gemma_f1_fine_tuning_performance.png"
plt.savefig(os.path.join(output_folder, output_file), dpi=300)
os.path.join(output_folder, output_file)

print('finish saving figure')