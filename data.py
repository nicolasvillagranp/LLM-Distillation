import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer


def load_data_from_hf() -> tuple[Dataset, Dataset]:
    """
    Load data from huggingface website

    Returns:
        tuple[Dataset, Dataset]: tuple containing both of the datasets to concatenate.
        The first one is the alpaca-cleaned dataset and the second one is the dolly-15k dataset
    """
    alpaca_cleaned_ds = load_dataset("yahma/alpaca-cleaned")
    dolly_ds = load_dataset("databricks/databricks-dolly-15k")

    return alpaca_cleaned_ds, dolly_ds


def load_data_from_path(path: str) -> Dataset:
    """
    Load data from path, already preprocessed. It must be a json file

    Args:
        path (str): path to the json file

    Returns:
        tuple[DatasetDict, DatasetDict]: tuple containing both of the datasets to concatenate.
        The first one is the alpaca-cleaned dataset and the second one is the dolly-15k dataset
    """
    return load_dataset("json", data_files=path)


def process_alpaca_prompt(instruction: str, alpaca_input: str) -> str:
    """
    Process alpaca prompts to add the input to the instruction

    Args:
        instruction (str): instruction
        alpaca_input (str): input

    Returns:
        str: processed prompt
    """
    if alpaca_input != "":
        if instruction[-1] == ".":
            return instruction[:-1] + ": " + alpaca_input
        else:
            return instruction + ": " + alpaca_input
    else:
        return instruction


def preprocess_data(alpaca_cleaned_ds: Dataset, dolly_ds: Dataset) -> Dataset:
    """
    Join both datasets to convert them to a dataset

    Args:
        alpaca_cleaned_ds (Dataset): alpaca cleaned dataset
        dolly_ds (Dataset): dolly dataset

    Returns:
        Dataset: joined dataset
    """
    alpaca_cleaned_df = pd.DataFrame(alpaca_cleaned_ds["train"])

    # Prepare prompt column by combining the instruction and the input columns.
    alpaca_cleaned_df["prompt"] = alpaca_cleaned_df.apply(
        lambda x: process_alpaca_prompt(x["instruction"], x["input"]), axis=1
    )

    # Drop the instruction and input columns
    alpaca_cleaned_df = alpaca_cleaned_df.drop(columns=["instruction", "input"])

    # Change output column name to target
    alpaca_cleaned_df = alpaca_cleaned_df.rename(columns={"output": "target"})

    # Load it to pandas to preprocess it easily
    dolly_df = pd.DataFrame(dolly_ds["train"])

    # Prepare prompt column by combining the instruction and the context columns.
    dolly_df["prompt"] = dolly_df.apply(
        lambda x: (
            x["instruction"] + "\nCONTEXT: " + x["context"]
            if x["context"] != ""
            else x["instruction"]
        ),
        axis=1,
    )

    # Drop the instruction and context columns
    dolly_df = dolly_df.drop(columns=["instruction", "context", "category"])

    # Change name of response column to target
    dolly_df = dolly_df.rename(columns={"response": "target"})

    # Combine the two datasets
    combined_df = pd.concat([alpaca_cleaned_df, dolly_df], ignore_index=True)

    return Dataset.from_pandas(combined_df)

def tokenize_dataset(tokenizer: AutoTokenizer, dataset: Dataset) -> dict:
    """
    Tokenize dataset with concatenation of prompt and target.

    Args:
        tokenizer (AutoTokenizer): tokenizer
        dataset (Dataset): dataset

    Returns:
        dict: tokenized dataset
    """

    def tokenize_function(examples):
        # Concatenate "prompt" and "target" with a separator if needed (e.g., a space or newline)
        concatenated = [f"{prompt} \n {target}" for prompt, target in zip(examples["prompt"], examples["target"])]
        return tokenizer(concatenated)

    # Neither padding nor truncation as batch-size = 1.
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return tokenized_dataset