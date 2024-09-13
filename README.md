# EIR Thai Medical LLM 
<p align='center'>
<img src="./img/eir_logo.png"  width="128" height="84" center-align="true">
</p>

We present EIR Large Language Models (LLMs) to enhance Thailand’s healthcare system, particularly in areas such as population health management and clinical trials. LLMs can extract valuable insights from Electronic Health Records (EHR) and digital medical data, helping to develop assistant treatment plans more efficiently.
Advantages of LLMs include their ability to perform Zero-Shot Learning and techniques like Chain-of-Thought (CoT), which improve decision-making and data analysis accuracy. However, the use of LLMs in healthcare raises privacy concerns, especially when dealing with sensitive patient information. In Thailand, challenges remain in developing NLP technology that supports the Thai language due to its complex grammar and limited high-quality resources. The research introduces the Eir AI Thai Medical LLM 8B model, adapted from LLaMA 3.1 Instruct-8B, which has been fine-tuned for Thai medical language tasks to improve precision medicine in Thailand.

<summary><b>Environment Install</b></summary>
You can install necessary packages by unsloth library by creating conda environment.
(see: https://github.com/unslothai/unsloth)

For example (See unsloth library for lastest updating):
```
conda create --name Eir_eval \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate Eir_eval

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
```

After install, if your troch version is not 2.4.1, you can run this command
```
pip3 install torch torchvision torchaudio
```
(see, for more detail: https://pytorch.org/get-started/locally/)

We also used flash-attention2.
(see: https://github.com/Dao-AILab/flash-attention)

```
pip install flash-attn --no-build-isolation
```

For BLEU evaluation:
(see: https://github.com/mjpost/sacrebleu)
```
pip install sacrebleu
```

<summary><b>Evaluation Med QA</b></summary>
We used lm-evaluation-harness from EleutherAI to evaluate medical MMLU tasks.<br>

(To install library, see: https://github.com/EleutherAI/lm-evaluation-harness)

```
lm_eval --model vllm \
  --model_args pretrained=EIRTHAIMED/Llama-3.1-EIRAI-8B,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1,trust_remote_code=True,max_model_len=2048 \
  --tasks pubmedqa,medqa_4options,medmcqa,mmlu_clinical_knowledge,mmlu_medical_genetics,mmlu_anatomy,mmlu_professional_medicine,mmlu_college_biology,mmlu_college_medicine \
  --batch_size auto \
  --device cuda \
  --output_path ./results \
  --log_samples 
```


<summary><b>Evaluation BLEU Medical Translate English to Thai</b></summary>
To evaluate BLEU Medical Translate, run this command with your model (total 30 samples) 

```
python src/BLEU.py --model EIRTHAIMED/Llama-3.1-EIRAI-8B
```

<summary><b>Evaluation Seacrowd</b></summary>
We used seacrowd-eval from SCB-10X to evaluate Thai Exam and M3Exam.<br>
(To install library, see: https://github.com/scb-10x/seacrowd-eval/tree/leaderboard)

```
MODEL_NAME=EIRTHAIMED/Llama-3.1-EIRAI-8B sh runner.sh
```


<summary><b>Evaluation EHR Task</b></summary>
To generate response from your model (for now support only llama3.0-3.1 template)

```
python src/prompt_gen.py --model EIRTHAIMED/Llama-3.1-EIRAI-8B
```

- Run GPT-4 for evaluation

To evaluate all model in EHR_task_responses.json
 
```
export API_KEY="your_openai_api_key_here"
python src/test.py 
```
To evaluate specific models in EHR_task_responses.json

```
export API_KEY="your_openai_api_key_here"
python src/test.py --models EIRTHAIMED/Llama-3.1-EIRAI-8B meta-llama/Meta-Llama-3.1-8B-Instruct
```

</details>
