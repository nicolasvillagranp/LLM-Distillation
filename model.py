import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import PreTrainedTokenizer, PreTrainedModel
from peft import PeftModel


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Load tokenizer from model name

    Args:
        model_name (str): model name

    Returns:
        AutoTokenizer: tokenizer
    """
    return AutoTokenizer.from_pretrained(
        model_name, add_eos_token=True, use_fast=True, padding_side="left"
    )


def load_model(model_name: str) -> AutoModelForCausalLM:
    """
    Load model from model name

    Args:
        model_name (str): model name

    Returns:
        AutoModelForCausalLM: model
    """
    # compute_type = getattr(torch, "bfloat16")
    config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    )
    return AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=config, device_map=torch.device("cuda:0")
    )

def load_saved_model(model, model_path: str):
    """
    Load a saved model checkpoint or the final model.

    Args:
        model: The model object to load the state_dict into.
        checkpoint_dir (str): Directory where checkpoints are stored.
        final_model_dir (str): Directory where the final interrupted model is stored.
        step (int, optional): The step number of the checkpoint to load. If None, loads the final model.

    Returns:
        model: The model with the loaded weights.
    """
    tokenizer = load_tokenizer(model_path)
    base_model = load_model("Qwen/Qwen2.5-7B")
    model = PeftModel.from_pretrained(  
        base_model,
        model_path,
        torch_dtype=torch.float16
    )

    return tokenizer, model


def generate_response(prompt: str, tokenizer: PreTrainedTokenizer, model: PreTrainedModel, device, max_new_tokens: int = 200, temperature: float = 1.0, top_p: float = 0.9):
    """
    Generate a response using a tokenizer and model.
    
    Args:
        prompt (str): The input prompt to generate a response.
        tokenizer (PreTrainedTokenizer): The tokenizer instance.
        model (PreTrainedModel): The model instance.
        device: The device to use for generation (e.g., "cuda" or "cpu").
        max_new_tokens (int): The maximum number of new tokens to generate.
        temperature (float): The temperature parameter for generation.
        top_p (float): The nucleus sampling probability.
    
    Returns:
        str: The generated response as a string.
    """
    # Tokenize the prompt
    tokens = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    
    # Generate the response
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,  # Use max_new_tokens instead of max_length
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode the output into a string
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response
