# import necessary libraries
import numpy as np
import time
import numpy as np
import os
import torch
from src.data import load_data_from_hf, preprocess_data, tokenize_dataset
from src.model import load_model, load_tokenizer
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
TOP_K = 1024  
LOGITS_DIR = 'data2/logits'
INDICES_DIR = 'data2/indices'

os.makedirs(LOGITS_DIR, exist_ok=True)
os.makedirs(INDICES_DIR, exist_ok=True)

# Load the data
alpaca_cleaned_ds, dolly_ds = load_data_from_hf()

# Preprocess the data (Do not shuffle or shuffle with the same seed in train)
dataset = preprocess_data(alpaca_cleaned_ds, dolly_ds).train_test_split(seed=42, shuffle = False)

# Load the tokenizer, model and tokenize the dataset
model_name: str = "Qwen/Qwen2.5-32B-Instruct" # Instruct big model
tokenizer = load_tokenizer(model_name)
model = load_model(model_name)

# We dont apply padding in this case and max length truncation
#  as we do 1 sample batching.
tokenized_dataset = tokenize_dataset(tokenizer, dataset)

batch_size = 1

n = len(tokenized_dataset['train'])
mean = 0.0

# Iterate over the dataset
for i in range(0, n, batch_size):
    t0 = time.time()
    model.eval() # Put model in inference mode.
    batch = tokenized_dataset["train"][i:i+batch_size] # Code thought to manage different batch_sizes
    
    # Get inputs ids, att masks and labels
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = tokenizer.eos_token_id 
    
    # Move tensors to the model's device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    
    # Forward pass with teacher forcing
    with torch.no_grad():  # No gradients cals
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    logits = outputs.logits  
    softmax_probs = F.softmax(logits, dim=-1)

    # Extract top-k logits and indices
    topk_values, topk_indices = torch.topk(logits, k=TOP_K, dim=-1)

    # Calculate mean of the cumsum of topk probabilities.
    topk_probs = torch.gather(softmax_probs, dim=-1, index=topk_indices)
    accumulated_prob_mean = torch.sum(topk_probs, dim=-1).mean()

    mean += accumulated_prob_mean.item()

    topk_values_fp16 = topk_values.half()  # Convert to float16 for efficient disk saving.
    
   
    # Convert to numpy and get paths
    topk_values_np = topk_values_fp16.squeeze(0).cpu().numpy()      
    topk_indices_np = topk_indices.squeeze(0).cpu().numpy()     
    logits_filepath = os.path.join(LOGITS_DIR, f'logits_{i}.npz')
    indices_filepath = os.path.join(INDICES_DIR, f'indices_{i}.npz')

    # Save the arrays in compressed forma
    np.savez_compressed(logits_filepath, topk_logits=topk_values_np)
    np.savez_compressed(indices_filepath, topk_indices=topk_indices_np)

    if i % 100 == 0:
        torch.cuda.empty_cache() # Remove fragmentation
    
print(f'The mean accumulated probability over {TOP_K} top k is: {mean / n}')



