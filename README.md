# EIR Thai Medical LLM 
<p align='center'>
<img src="./img/eir_logo.png"  width="128" height="84" center-align="true">
</p>

We present **EIR** Large Language Models (LLMs) to enhance Thailand’s healthcare system, particularly in areas such as population health management, clinical trials, and drug discovery. LLMs can extract valuable insights from Electronic Health Records (EHR) and digital medical data, helping to develop new treatment plans and drugs more efficiently.
Advantages of LLMs include their ability to perform Zero-Shot Learning and techniques like Chain-of-Thought (CoT), which improve decision-making and data analysis accuracy. However, the use of LLMs in healthcare raises privacy concerns, especially when dealing with sensitive patient information. In Thailand, challenges remain in developing NLP technology that supports the Thai language due to its complex grammar and limited high-quality resources. The research introduces the Eir AI Thai Medical LLM 8B model, adapted from LLaMA 3.1 Instruct-8B, which has been fine-tuned for Thai medical language tasks to improve precision medicine in Thailand


<details>
<summary>Environment Install</summary>

```
conda create -n eir python=3.9 -y
conda activate eir
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install
pip install 
```

</details>

<details>
<summary>Evaluation Med QA</summary>

  ```
  lm_eval --model vllm \
    --model_args pretrained=EIRTHAIMED/Llama-3.1-EIRAI-8B,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1,trust_remote_code=True,max_model_len=2048 \
    --tasks pubmedqa,medqa_4options,medmcqa,mmlu_clinical_knowledge,mmlu_medical_genetics,mmlu_anatomy,mmlu_professional_medicine,mmlu_college_biology,mmlu_college_medicine \
    --batch_size auto \
    --device cuda \
    --output_path ./results \
    --log_samples 
  ```

</details>

<details>
<summary>Evaluation BLEU Medical Translate English to Thai </summary>

  ```
  python src/BLEU.py --model EIRTHAIMED/Llama-3.1-EIRAI-8B
  ```

</details>

<details>
<summary>Evaluation Seacrowd </summary>

  ```
  CUDA_VISIBLE_DEVICES=0 python src/evaluate.py \
    --model_name {MODEL_PATH} \
    --data_path eval/test.json \
    --output_path {OUTPUT_PATH}
  ```

- Run GPT-4 for evaluation
 
  ```
  python eval/gpt4_evaluate.py --input {INPUT_PATH} --output {OUTPUT_PATH} 
  ```
</details>

<details>
<summary>Evaluation EHR Task </summary>

  ```
  CUDA_VISIBLE_DEVICES=0 python src/evaluate.py \
    --model_name {MODEL_PATH} \
    --data_path eval/test.json \
    --output_path {OUTPUT_PATH}
  ```

- Run GPT-4 for evaluation
 
  ```
  python eval/gpt4_evaluate.py --input {INPUT_PATH} --output {OUTPUT_PATH} 
  ```
</details>
