# EIR Thai Medical LLM 
<p align='center'>
<img src="./resources/..png"  width="400" height="400" center-align="true">
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
<summary>Evaluation</summary>

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
