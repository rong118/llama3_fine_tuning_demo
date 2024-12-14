import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from modelscope import snapshot_download
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

# 需要微调的基座模型
# https://huggingface.co/meta-llama/Llama-3.2-1B
model_id = 'meta-llama/Llama-3.2-1B'
models_dir = './model_1B'

# model_path = f"{models_dir}/model/{model_id.replace('.', '___')}"
model_path = f"/Users/rongli/Desktop/venv_llama3/llama3-ft/model_1B"
checkpoint_dir = f"/Users/rongli/Desktop/venv_llama3/llama3-ft/model_1B/checkpoint/1B"
lora_dir = f"/Users/rongli/Desktop/venv_llama3/llama3-ft/model_1B/lora/1B"

torch_dtype = torch.half

dataset_file = './dataset/huanhuan.json'

# 检查CUDA是否可用，然后检查MPS是否可用，最后回退到CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

def train():
    # 加载模型
    # model_dir = snapshot_download(model_id=model_id, cache_dir=f"{models_dir}/model", revision='master')
    # if model_path != model_dir:
    #     raise Exception(f"model_path:{model_path} != model_dir:{model_dir}")

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="mps", torch_dtype=torch_dtype)
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

    # 加载数据
    df = pd.read_json(dataset_file)
    ds = Dataset.from_pandas(df)
    print(ds[:3])

    # 处理数据
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    def process_func(item):
        MAX_LENGTH = 384  # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
        input_ids, attention_mask, labels = [], [], []
        instruction = tokenizer(
            f"<|start_header_id|>user<|end_header_id|>\n\n{item['instruction'] + item['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
        response = tokenizer(f"{item['output']}<|eot_id|>", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
        if len(input_ids) > MAX_LENGTH:  # 做一个截断
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

    tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[1]["labels"])))

    # 加载lora权重
    model = get_peft_model(model, lora_config)

    # 训练模型
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=3,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()

    # 保存模型
    trainer.model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)

if __name__ == '__main__':
    train()
