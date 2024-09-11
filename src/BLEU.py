import json
import os
import argparse
import torch
from tqdm import tqdm
from datasets import load_dataset
from sacrebleu.metrics import BLEU
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Initialize BLEU metric with flores200 tokenization
bleu = BLEU(tokenize="flores200")

# LLaMA prompt format
LLAMA_PROMPT = """<|start_header_id|>user<|end_header_id|>
{}
Input: {}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Answer: 
"""

def load_model_and_tokenizer(model_name: str):
    """
    Load the specified model and tokenizer from Hugging Face.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        model, tokenizer: The loaded model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    model.to(torch.device("cuda"))
    return model, tokenizer


def load_evaluation_dataset(eval_dataset_path: str):
    """
    Load the evaluation dataset from a file or dataset name.

    Args:
        eval_dataset_path (str): Path to the evaluation dataset or dataset name.

    Returns:
        Dataset: Loaded evaluation dataset.
    """
    if os.path.exists(eval_dataset_path):
        return load_dataset("json", data_files={"validation": eval_dataset_path}, split="validation")
    else:
        return load_dataset(eval_dataset_path, split="validation")


def evaluate_model(model, tokenizer, eval_dataset):
    """
    Evaluate the model using the specified evaluation dataset and calculate BLEU scores.

    Args:
        model: The loaded model for evaluation.
        tokenizer: The tokenizer for the model.
        eval_dataset: The evaluation dataset to use.

    Returns:
        results, references, inputs: Model predictions, references, and inputs for BLEU scoring.
    """
    results = []
    references = []
    inputs = []
    splitter = "assistant\nAnswer: \n"

    for row in tqdm(iter(eval_dataset), total=len(eval_dataset)):
        references.append(row["th"])
        inputs.append(row["en"])
        input_text = LLAMA_PROMPT.format(row['Instruction'], row['en'])
        input_ids = tokenizer([input_text], return_tensors="pt").to("cuda")

        outputs = model.generate(**input_ids, max_new_tokens=1500, do_sample=False)
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Extract the response part
        try:
            response_text = generated_text[0].split(splitter)[1]
            results.append(response_text)
        except IndexError:
            results.append("")

    return results, references, inputs


def save_results(model_name, results, references, inputs):
    """
    Save the evaluation results to a JSON file.

    Args:
        model_name (str): Name of the model used for evaluation.
        results (list): Generated model outputs.
        references (list): Reference translations.
        inputs (list): Input sentences.
    """
    bleu_score = str(bleu.corpus_score(results, [references]))
    save_path = f"eval_results_{model_name}.json".replace("/", "_")
    
    data = {
        "bleu_score": bleu_score,
        "evaluations": [
            {"pred": pred, "ref": ref, "input": ip} for pred, ref, ip in zip(results, references, inputs)
        ]
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"Results saved to {save_path}")


def main(model_name: str, eval_dataset: str):
    """
    Main function to load the model, perform evaluation, and save results.

    Args:
        model_name (str): Name of the model to use.
        eval_dataset (str): Path to the evaluation dataset.
    """
    print(f"Loading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name)

    print(f"Loading evaluation dataset: {eval_dataset}")
    dataset = load_evaluation_dataset(eval_dataset)

    print("Starting evaluation...")
    results, references, inputs = evaluate_model(model, tokenizer, dataset)

    print("Saving evaluation results...")
    save_results(model_name, results, references, inputs)

    print(f"BLEU score: {bleu.corpus_score(results, [references])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained model on a dataset and compute BLEU score.")
    parser.add_argument("--model", type=str, required=True, help="The model to load and evaluate.")
    parser.add_argument("--eval-dataset", type=str, default='dataset/translate_test.json', help="Path to the evaluation dataset (JSON format).")
    args = parser.parse_args()

    main(model_name=args.model, eval_dataset=args.eval_dataset)
