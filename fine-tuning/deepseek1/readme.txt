 
0成本实战！手把手教你微调DeepSeek-R1
https://www.bilibili.com/video/BV1kCrjBpEfB/?spm_id_from=333.337.search-card.all.click&vd_source=5296b77bd88eec13d945adf9efc64874

model: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
 
data: https://modelscope.cn/datasets/Kedreamix/psychology-10k-Deepseek-R1-zh/dataPeview
 
 
 第一步：加载模型并测试

  (Step 1: Load model and test)

  Description: 这里主要完成模型的加载与初步测试，确保路径正确且能够调用 GPU。
  (Here we mainly complete loading and initial testing of the model, ensuring the path is correct and GPU can be called.)

  Code block:
  # 加载模型并测试
  from transformers import AutoTokenizer, AutoModelForCausalLM

  # 指定模型路径，这里是一个本地已经下载好的 DeepSeek-R1 模型的路径
  model_name = "/mnt/workspace/.cache/modelscope/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5"

  # 加载分词器和模型
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

  print("模型加载成功！")
  
  
  
  
  # 准备数据集
  import json

  # 假设这是你的 50 条样本数据
  samples = [...]  # 每个 sample 应为 dict 类型, 例如 {"text": "xxx"} 或 {"input": "...", "output": "..."}

  # 写入 jsonl 文件
  with open("dataset.jsonl", "w", encoding="utf-8") as f:
      for sample in samples:
          f.write(json.dumps(sample, ensure_ascii=False) + "\n")

  print("数据集制作完成！")
  
  
  
  
  # 拆分数据集
  from datasets import load_dataset

  # 加载本地数据
  dataset = load_dataset("json", data_files={"train": "dataset.jsonl"}, split="train")

  print("数据总数量: ", len(dataset))

  # 划分训练集和测试集 (90% 训练, 10% 测试)
  train_test_split = dataset.train_test_split(test_size=0.1)

  # 提取训练集和验证集
  train_dataset = train_test_split["train"]
  eval_dataset = train_test_split["test"]

  print(f"train dataset len: {len(train_dataset)}")
  print(f"test dataset len : {len(eval_dataset)}")
  print("训练数据的准备工作完成")
  
  
  
  
  def tokenizer_function(many_samples):
      """
      将 prompt 和 completion 拼接后进行分词处理
      """
      # 将每条样本的 prompt 和 completion 拼接成一个文本
      texts = [f"{prompt}\n{completion}" for prompt, completion in zip(many_samples["prompt"], many_samples["completion"])]

      # 使用 tokenizer 进行分词, 截断长度为 512, 填充至最大长度
      tokens = tokenizer(
          texts,
          truncation=True,
          max_length=512,
          padding="max_length"
      )

      # 设置 labels 为 input_ids 的副本 (用于因果语言建模任务)
      tokens["labels"] = tokens["input_ids"].copy()
	  return tokens

  tokenized_train_dataset = train_dataset.map(tokenizer_function, batched=True)
  tokenized_eval_dataset = eval_dataset.map(tokenizer_function, batched=True)
  
  
  
  # 量化设置
  from transformers import AutoModelForCausalLM, BitsAndBytesConfig

  # 配置 8bit 量化
  quantization_config = BitsAndBytesConfig(load_in_8bit=True)

  # 重新加载量化后的模型
  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      quantization_config=quantization_config,
      device_map="auto"
  )


 # 配置 LoRA 参数
  from peft import get_peft_model, LoraConfig, TaskType

  lora_config = LoraConfig(
      r=8,                              # LoRA 秩 (rank)，控制适配器大小，通常设为 8~32
      lora_alpha=16,                    # 控制 LoRA 更新的缩放因子，一般为 r 的倍数
      lora_dropout=0.05,                # Dropout 概率，防止过拟合
      task_type=TaskType.CAUSAL_LM      # 任务类型：因果语言模型
  )

  # 获取 PEFT 模型
  model = get_peft_model(model, lora_config)
  model.print_trainable_parameters()
  
  
   # 配置训练参数
  training_args = TrainingArguments(
      output_dir="./finetuned_models",      # 模型保存路径
      num_train_epochs=10,                  # 训练轮数
      per_device_train_batch_size=4,        # 每设备批量大小 (GPU 上)
      gradient_accumulation_steps=8,        # 梯度累积步数 (模拟更大 batch)
      fp16=True,                            # 使用 FP16 半精度训练，节省显存
      logging_steps=10,                     # 每 10 步打印一次日志
      save_steps=100,                       # 每 100 步保存一次 checkpoint
      eval_strategy="steps",               # 每隔一定步数评估一次
      eval_steps=10,                        # 每 10 步进行一次评估
      learning_rate=3e-5,                   # 学习率
      logging_dir="./logs",                 # 日志保存路径
	  )