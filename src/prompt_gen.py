import json
import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

# LLaMA prompt format
LLAMA_PROMPT= """<|start_header_id|>user<|end_header_id|>
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

Instruction:
{}

Input:
{}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Response:
"""

def load_model_and_tokenizer(model_id: str):
    """
    Load the pre-trained model and tokenizer for text generation.

    Args:
        model_id (str): The model ID to load from Hugging Face.

    Returns:
        model, tokenizer: The loaded model and tokenizer.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def update_json_data(output_file: str, model_id: str, reset: bool):
    """
    Load the JSON file and check if the model_id exists in the data.
    If it doesn't exist, initialize it.

    Args:
        output_file (str): Path to the JSON file.
        model_id (str): The model ID to check and add to the JSON data.

    Returns:
        data (dict): The loaded and updated JSON data.
    """
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"The file {output_file} does not exist.")
    
    with open(output_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    if model_id not in data or reset:
        data["models"][model_id] = {}

    return data


def generate_responses(model, tokenizer, data, model_id, output_file, batch_size=50):
    """
    Generate model responses for each input and instruction in the dataset.

    Args:
        model: The pre-trained model for generating responses.
        tokenizer: The tokenizer corresponding to the model.
        data (dict): The loaded data from the JSON file.
        model_id (str): The ID of the model used for generation.
        output_file (str): Path to the output JSON file for saving intermediate results.
        batch_size (int): Number of responses to generate before saving.

    Returns:
        data (dict): Updated data with generated responses.
    """
    count = 0
    total_records = len(data["type"])

    for i, key in enumerate(data["type"]):
        if key in data["models"][model_id]:
            continue
        
        instruction_text = data["instruction_thai"][key]
        input_text = data["input"][key]

        formatted_prompt = LLAMA_PROMPT.format(instruction_text, input_text)
        inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")

        # Generate the output
        outputs = model.generate(**inputs, max_new_tokens=1500, do_sample=False)
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Extract the response part from the generated text
        try:
            response_text = generated_text[0].split("Response:\n")[1]
        except IndexError:
            response_text = ""

        data["models"][model_id][key] = response_text
        print(f"Processed {i + 1}/{total_records} | Key: {key}")
        print("Generated Response:")
        print(response_text)
        print("-" * 90)

        count += 1
        if count == batch_size:
            save_json(output_file, data)
            count = 0

    return data


def save_json(output_file: str, data: dict):
    """
    Save the updated data to the specified JSON file.

    Args:
        output_file (str): Path to the JSON file.
        data (dict): The updated data to save.
    """
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print(f"Saved data to {output_file}")


def main(model_id: str, output_file: str, reset: bool):
    """
    Main function to load the model, process the data, and generate responses.

    Args:
        model_id (str): The ID of the model to load.
        output_file (str): Path to the output JSON file.
    """

    print(f"Loading model {model_id}...")
    model, tokenizer = load_model_and_tokenizer(model_id)

    print(f"Loading and updating JSON data from {output_file}...")
    data = update_json_data(output_file, model_id, reset)

    print("Generating responses...")
    data = generate_responses(model, tokenizer, data, model_id, output_file)

    print("Saving final data...")
    save_json(output_file, data)

    print("Process complete. JSON file updated with generated responses.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses using a pre-trained model and update the JSON file.")
    parser.add_argument("--model", type=str, required=True, help="The ID of the model to load from Hugging Face.")
    parser.add_argument("--output-file", type=str, default='dataset/EHR_task_responses.json', help="Path to the JSON file to update.")
    parser.add_argument("--reset", type=str, default=False, help="Set to \"True\" to Reset the specific model responses in the JSON file.")
    args = parser.parse_args()

    main(model_id=args.model, output_file=args.output_file, reset=args.reset)
