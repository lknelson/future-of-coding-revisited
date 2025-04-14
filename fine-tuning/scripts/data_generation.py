import pdb
import pandas as pd

## examining what the fine_tuning_data looks like

original_df = pd.read_json("./data/old_data_clean.json")  # (1253, 45)
df = pd.read_json("./data/inequality_data_finetuning.json") #(1253, 3) ## same rows, different columns

"inequality_data_finetuning_sample_30.json"

for sample_num in (30, 60, 100, 200, 500, 700, 1000):
    sample_df = df.sample(n=sample_num, random_state=20241115)
    sample_df = sample_df.reset_index(drop=True)
    # write this sample df to a new json file
    sample_df.to_json(f"./data/inequality_data_finetuning_sample_{sample_num}.json", orient='records', lines=False)

