# Nov 14, 2024 - run this on an H100 to see what happens
# need to downgrade pandas and numpy for this code to work in H100 

import os
import pdb
import time
import torch
import numpy as np
import pandas as pd


from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments, TrainerCallback
from unsloth import FastLanguageModel, is_bfloat16_supported

# try to reduce memory fragmentation 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache() 
torch.backends.cuda.matmul.allow_tf32 = True

path = "./results/"

# Check if the path exists, if not, create it
if not os.path.exists(path):
    os.makedirs(path)
    print(f"Path '{path}' created.")
else:
    print(f"Path '{path}' already exists.")

# define a custom call back function to clear cache at each training iteration 
class EmptyCacheCallBack(TrainerCallback): 
    def one_step_end(self, args, state, control, **kwargs): 
        torch.cuda.empty_cache()

max_seq_length = 10048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

alpaca_prompt = """We categorize articles that are related to issues of income inequality, changes in income or wealth, general economic conditions. Below is an instruction that describes the classification task, paired with an input containing the article to be classified. Write a response that appropriately completes the classification request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
    
## iterate over 10 datasets here:
## TODO: add timer to show how long it takes for each iteration
timing_df = pd.DataFrame(columns=['iteration', 'duration'])

for i in ( 500, 700, 1000): # ( 30, 60, 100, 200, 500, 700, 1000)

    torch.cuda.empty_cache()

    iteration_start_time = time.time()

    # initialize the model and LoRA setup at each step
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gemma-2-27b-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

    print(f"start to fine-tune {i} examples")
    dataset = load_dataset("json", data_files = f"./data/inequality_data_finetuning_sample_{i}.json", split = "train")
    dataset = dataset.map(formatting_prompts_func, batched = True)

    #pdb.set_trace()

    # add the callback class to the trainer to clear GPU cache to release moemory
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2, ##maybe batch size can be lower here
            gradient_accumulation_steps = 4, ## this could be also lower to reduce memory footprint
            warmup_steps = 5,
            max_steps = 60,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none", # Use this for WandB etc
        ),
        callbacks=[EmptyCacheCallBack()] ## add the custom callback here to release GPU memory
    )

    trainer_stats = trainer.train()

    print(f'training sample {i} finished')

    #model.save_pretrained("lora_model") # Local saving
    #tokenizer.save_pretrained("lora_model")
    # alpaca_prompt = Copied from above

    print('start running inference on the entire dataset')
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    corpus = pd.read_json('./data/old_data_clean.json')

    output_list = []

    for num in range(len(corpus)):
        text = corpus.loc[num,'text'][:max_seq_length]
        inputs = tokenizer(
        [
            alpaca_prompt.format(
                "Is the article relevant to American economic inequality? Answer relevant or irrelevant.", # instruction
                text, # input
                "", # output - leave this blank for generation!
            )
        ], return_tensors = "pt").to("cuda")

        outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
        output_list.append(tokenizer.batch_decode(outputs))

    corpus['llama_content'] = output_list

    corpus.to_csv(f'./results/inequality_dataset-gemma2-27b-NoDefinition-FineTuned-{i}.csv')
    iteration_end_time = time.time()
    iteration_duration = (iteration_end_time - iteration_start_time) / 60
    timing_df = pd.concat([timing_df, pd.DataFrame({'iteration': [i], 'duration': [iteration_duration]})], ignore_index=True)
    print(f"it takes {iteration_duration} to finetune {i} samples")

    # Clear GPU memory so it wont get OOM issue 
    del model
    del trainer
    del dataset
    torch.cuda.empty_cache()

timing_df.to_csv("./results/iteration_duration_mins.csv", index=False)
print("all datasets processed")


