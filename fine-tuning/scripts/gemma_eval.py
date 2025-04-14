# Dec 6: evaluation metrics for all classes, all tests, exclude training data points
import os
import pdb
import json
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, f1_score

#directory = "/Users/nga/Desktop/fine-tuning/src/results/data/5folds/"
directory = "./results/5folds/"

# read all the csv files
csv_files = []

if os.path.exists(directory):
    csv_files = [os.path.join(directory, file)
                 for file in os.listdir(directory)
                 if file.endswith(".csv") and file.startswith("inequality")]
# Output the list of CSV file paths
#print(csv_files)
## calculate precision, recall for one file at a time
# setup a dataframe to save these results:

columns = ['sample_set',
           #'precision_per_class',
           'weighted_avg_precision',
           #"recall_per_class",
           "weighted_avg_recall",
           #"f1_score_per_class",
           "weighted_avg_f1_score",
           ]
results_df = pd.DataFrame(columns=columns)

sample_set = [(i, j) for i in (10, 100, 600, 1000) for j in (1, 2, 3, 4, 5)] # total 20 samples
for i, j in sample_set:
    #print(f'processing file {i}-{j}')
    df_ft = pd.read_csv(f"./results/5folds/inequality_dataset-gemma2-27b-NoDefinition-FineTuned-{i}-{j}.csv", index_col=0)
    df_excluded = pd.read_json(f"../data/data_5folds/inequality_data_finetuning_sample_{i}_{j}.json")

    # second method
    df_ft['content_compared'] =  df_ft['title'] + "\n" + df_ft['text']
    df_ft = df_ft[~df_ft['content_compared'].isin(df_excluded['input'])].reset_index(drop=True)

    #pdb.set_trace()
    # df_excluded['title'] =  df_excluded.input.str.split("\n").str[0]
    # #filtered_rows =  df_ft[ df_ft['title'].apply(lambda x: any(excluded in x for excluded in df_excluded['title']))]
    # df_ft = df_ft[~df_ft['title'].isin(df_excluded['title'])].reset_index(drop=True)
    print(f'after removing {i} rows from original dataframe, we got {df_ft.shape[0]} rows')
    #pdb.set_trace()

    labels = [int(df_ft['code'][i] != 5) for i in range(len(df_ft))]
    df_ft['code'] = labels
    df_ft['llama_code'] = df_ft['llama_content']


    df_ft.loc[df_ft['llama_content'].str.lower().str.strip().str.replace('\\n', ' ').str.contains(
        'response: irrelevant'), 'llama_code'] = 0
    df_ft.loc[df_ft['llama_content'].str.lower().str.strip().str.replace('\\n', ' ').str.contains(
        'response: relevant'), 'llama_code'] = 1

    y_true = df_ft.code.astype(int)
    y_pred = df_ft.llama_code.astype(int)

    pdb.set_trace()

    # precision/recall of the 0 categories
    ## flipping the labels -> flipping precision/ recall -> high recall
    # calculate the precision and recall scores
    precision = precision_score(y_true, y_pred, pos_label=0)
    recall = recall_score(y_true, y_pred, pos_label=0)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=0)
    cm = confusion_matrix(y_true, y_pred)

    conf_matrix = confusion_matrix(y_true, y_pred)
    # Extract true positives for each class
    true_positives = np.diag(conf_matrix)

    # Calculate precision for each class
    precision_per_class = precision_score(y_true, y_pred, average=None)
    # Calculate proportion of true positives for each class
    total_true_positives = np.sum(true_positives)
    proportions = true_positives / total_true_positives
    #weighted_avg_precision = np.sum(precision_per_class * proportions)
    weighted_avg_precision = precision_score(y_true, y_pred, average='weighted')
    #print("another way to calculate this number,", weighted_avg_precision,  precision_score(y_true, y_pred, average='weighted'))
    #print("\nWeighted average metrics:\n")
    #precision_per_class = precision_score(y_true, y_pred, average=None)
    #print(precision_per_class)
    #print(f"Precision for each class: {[f'{r:.2f}' for r in precision_per_class]}")
    #weighted_avg_precision = np.sum(precision_per_class * proportions)
    #print(f"Weighted Average Precision {weighted_avg_precision:.2f}")
    #recall_per_class = recall_score(y_true, y_pred, average=None)
    # #print(f"Recall for each class: {[f'{r:.2f}' for r in recall_per_class]}")
    #weighted_avg_recall = np.sum(recall_per_class * proportions)
    weighted_avg_recall = recall_score(y_true, y_pred, average='weighted')
    #print(f"Weighted Average Recall: {weighted_avg_recall:.2f}")
    # Calculate F1 score for each class
    #f1_per_class = f1_score(y_true, y_pred, average=None)
    # #print(f"F1 Score for each class: {[f'{f:.2f}' for f in f1_per_class]}")
    # # Calculate weighted average F1 score
    #weighted_avg_f1 = np.sum(f1_per_class * proportions)
    weighted_avg_f1 = f1_score(y_true, y_pred, average='weighted')
    #print(f"Weighted Average F1 Score: {weighted_avg_f1:.2f}")

    #pdb.set_trace()
    new_row = pd.DataFrame(
        {'sample_set': [f"{i}-{j}"],
           'weighted_avg_precision': [weighted_avg_precision],
           "weighted_avg_recall": [weighted_avg_recall],
           "weighted_avg_f1_score": [weighted_avg_f1]
        }
    )
    results_df = pd.concat([results_df, new_row], ignore_index=True)

results_df.to_csv(directory + "gemma_metrics_summary_v3.csv", index=False)

print("finish writing metrics to a csv file")

# take average of all rows of the same sample type
average_results = pd.DataFrame(columns=columns)

for sample in (10, 100, 600, 1000):
    filtered_df = results_df[results_df['sample_set'].str.startswith(f"{sample}-")]
    #pdb.set_trace()
    averages = filtered_df.iloc[:, 1:].mean()
    summary_df = pd.DataFrame(averages).transpose()
    summary_df['sample_set'] = str(sample)
    # rearrange columns
    summary_df = summary_df[["sample_set", "weighted_avg_precision", "weighted_avg_recall", "weighted_avg_f1_score"]]
    average_results = pd.concat([average_results, summary_df], ignore_index=True)

# save summary_df
average_results.to_csv(directory + "gemma_final_metrics_summary_v3.csv", index=False)
