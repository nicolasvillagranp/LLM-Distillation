import torch
from torch.nn import CrossEntropyLoss, KLDivLoss
import torch.nn.functional as F
import numpy as np
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from typing import Any
from bitsandbytes.optim import PagedAdamW8bit
import os

from src.data import load_data_from_hf, preprocess_data, tokenize_dataset
from src.model import load_model, load_tokenizer



"""
Define them as global variables to not 
disrupt the SFTTrainer compute loss structure.
"""

val_loss_ce = []
val_loss_kl = []

train_loss_ce = []
train_loss_kl = []


def get_files():
    # Sort logits files
    logits_files = sorted(
        [
            os.path.join("data2/logits", fname)
            for fname in os.listdir("data2/logits")
            if fname.endswith('.npz')
        ],
        key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
    )

    # Sort indices files
    indices_files = sorted(
        [
            os.path.join("data2/indices", fname)
            for fname in os.listdir("data2/indices")
            if fname.endswith('.npz')
        ],
        key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
    )   
    return logits_files, indices_files


"""
The following function was originally designed to meet the SFTTrainner class requirements
We dont use it here to have full control over the optimization loop in the management of files.
"""

def compute_loss(model, inputs, counter: int, logits_files, indices_files, return_outputs=False, kl_lambda=0.1, training=True) -> tuple[torch.Tensor, Any] | torch.Tensor:
    """
    Computes the combined loss: standard cross-entropy loss plus KL divergence.

    Args:
        model: The model being trained.
        inputs (dict): A batch of inputs, including 'labels'.
        return_outputs (bool): Whether to return model outputs alongside loss.

    Returns:
        torch.Tensor or (torch.Tensor, Any): Loss value, optionally with model outputs.
    """
    outputs = model(
        **inputs
    )
    logits = outputs.logits  
    ce_loss_fn = CrossEntropyLoss()
    

    logits_permuted = logits.permute(0, 2, 1)  
    
    # Compute CrossEntropyLoss
    ce_loss = ce_loss_fn(logits_permuted, inputs["labels"])


    # Load the files and save them in tensor with correct dtype
    logits_file = logits_files[counter]
    indices_file = indices_files[counter]
    topk_logits_npz = np.load(logits_file)
    topk_indices_npz = np.load(indices_file)
    topk_logits = topk_logits_npz['topk_logits']  
    topk_indices = topk_indices_npz['topk_indices']  
    topk_logits = torch.tensor(topk_logits, dtype=torch.float32).to(logits.device)
    topk_indices = torch.tensor(topk_indices, dtype=torch.int64).to(logits.device)


    # Initialize topk_probs with zeros (Assumption made)
    topk_probs = torch.zeros_like(logits).to(logits.device)

    if len(topk_indices.shape) == 2: # If batch-size in forward big is 1.
        topk_indices = topk_indices.unsqueeze(0)
        topk_logits = topk_logits.unsqueeze(0)
    

    # Map probs to their respectives indices
    topk_probs.scatter_(
        2, 
        topk_indices, 
        F.softmax(topk_logits, dim=-1)
    )

    # Get real distribution of probs
    normal_probs = F.softmax(logits, dim=-1) + 1e-12

    # Compute KL divergence
    kl_loss_fn = KLDivLoss(reduction='batchmean') 
    kl_loss = kl_loss_fn(torch.log(normal_probs).to(model.device), topk_probs.to(model.device))  

    if training:
        train_loss_ce.append(ce_loss.cpu().item())
        train_loss_kl.append(kl_loss.cpu().item())
    else:
        val_loss_ce.append(ce_loss.cpu().item())
        val_loss_kl.append(kl_loss.cpu().item())

    # Convex sum of losses.
    total_loss = (1 - kl_lambda) * ce_loss + kl_lambda * kl_loss

    return (total_loss, outputs) if return_outputs else total_loss



def evaluate(model, tokenized_dataset, tokenizer, logits_files, indices_files, eval_size):
    batch_size = 1
    loss = 0.0
    model.eval()
    with torch.no_grad():
        for _, i in enumerate(range(len(logits_files) - eval_size, len(logits_files), batch_size)):
            batch = tokenized_dataset["train"][i:i+batch_size]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = tokenizer.eos_token_id

            batch = {
                "input_ids": input_ids.to(model.device),
                "attention_mask": attention_mask.to(model.device),
                "labels": labels.to(model.device),
            }
            # Compute the loss
            loss += compute_loss(model, batch,  i // batch_size, logits_files, indices_files, kl_lambda = 0.5, training=False).item()
    return loss / eval_size



def main() -> None:
    """
    Main function
    """
    # Load the data
    alpaca_cleaned_ds, dolly_ds = load_data_from_hf()
    dataset = preprocess_data(alpaca_cleaned_ds, dolly_ds).train_test_split(seed=42, shuffle = False)

    # Load the tokenizer, model and tokenize the dataset
    model_name: str = "Qwen/Qwen2.5-7B"
    tokenizer = load_tokenizer(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = load_model(model_name)
    model = prepare_model_for_kbit_training(model)
    
    tokenized_dataset = tokenize_dataset(tokenizer, dataset)
    # Load the corresponding top-K logits and indices
    logits_files, indices_files = get_files()
    # Define LoraConfig
    lora_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        r=8,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "k_proj",
            "q_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
            "all-linear",
        ],
    )

    model = get_peft_model(model, lora_config)
    # Initialize the optimizer
    optimizer = PagedAdamW8bit(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-4
    )
    # Define training parameters
    epochs = 3
    batch_size = 1
    eval_size = 130
    gradient_accumulation_steps = 1  # Number of steps to accumulate gradients
    max_steps = (len(tokenized_dataset['train']) // batch_size) // gradient_accumulation_steps
    checkpoint_dir = "checkpoints-new"
    os.makedirs(checkpoint_dir, exist_ok=True)
    try:
        for epoch in range(epochs):
            for step, i in enumerate(range(0, len(logits_files) - eval_size, batch_size)):
                model.train()

                # Get the inputs and attention mask
                batch = tokenized_dataset["train"][i:i+batch_size]
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]

                # Define labels
                labels = input_ids.clone()
                labels[:, :-1] = input_ids[:, 1:]
                labels[:, -1] = tokenizer.eos_token_id

                batch = {
                    "input_ids": input_ids.to(model.device),
                    "attention_mask": attention_mask.to(model.device),
                    "labels": labels.to(model.device),
                }
                # Compute the loss
                loss = compute_loss(model, batch,  step // batch_size, logits_files, indices_files, kl_lambda = 0.5)
                loss = loss / gradient_accumulation_steps
                loss.backward() 

                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()       
                    optimizer.zero_grad()   
                    print(f"Epoch [{epoch+1}/{epochs}], Step [{step+1}/{max_steps}], Loss: {loss.item()*gradient_accumulation_steps:.4f}")

                if (step + 1) % 100 == 0:
                    torch.cuda.empty_cache()  # Reduce fragmentation and clear cache.

                if step % 500 == 0:
                    evaluate(model, tokenized_dataset, tokenizer, logits_files, indices_files, eval_size)
                    model.save_pretrained(checkpoint_dir)
                
            model.save_pretrained(checkpoint_dir + f'epoch{epoch}')
                    
    except KeyboardInterrupt:
        # Save last achieved model
        model.to("cpu")
        final_model_path = "final-model-ki-2"
        os.makedirs(final_model_path, exist_ok=True)
        model.save_pretrained(final_model_path)
        print(f"Training interrupted. Model saved to {final_model_path}.")

        # Save numpy tensors
        os.makedirs("losses_values", exist_ok=True)
        losses_array_training = np.array([train_loss_ce, train_loss_kl])
        np.save("losses_values/train", losses_array_training)

        losses_array_eval = np.array([val_loss_ce, val_loss_kl])
        np.save("losses_values/eval", losses_array_eval)
    finally:
        model.save_pretrained(final_model_path)

if __name__ == "__main__":
    main()
