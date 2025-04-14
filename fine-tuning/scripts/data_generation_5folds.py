import pdb
import random
import pandas as pd

## examining what the fine_tuning_data looks like

original_df = pd.read_json("./data/old_data_clean.json")  # (1253, 45)
df = pd.read_json("./data/inequality_data_finetuning.json") #(1253, 3) ## same rows, different columns

for sample_num in (10, 100, 600, 1000):
    for i in range(1, 6):
        random_state = random.randint(20241115, 20251115)
        sample_df = df.sample(n=sample_num, random_state=random_state)
        sample_df = sample_df.reset_index(drop=True)
        # write this sample df to a new json file
        sample_df.to_json(f"./data/data_5folds/inequality_data_finetuning_sample_{sample_num}_{i}.json", orient='records', lines=False)

