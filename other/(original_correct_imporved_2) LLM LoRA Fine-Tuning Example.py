
# -------------------------------------------------------
# Load model directly
# -------------------------------------------------------
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="auto")

print(model)

# -------------------------------------------------------
# Test base model - question it knows
# -------------------------------------------------------
messages = [
    {"role": "user", "content": "Who was the winner of 2022 world cup?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

# -------------------------------------------------------
# Count trainable parameters (before LoRA)
# -------------------------------------------------------
import torch

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# -------------------------------------------------------
# Test base model - question it doesn't know
# -------------------------------------------------------
messages = [
    {"role": "user", "content": "Who was the winner of the 2024 Olympic men's football tournament?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

# -------------------------------------------------------
# Reference text
# -------------------------------------------------------
# Reference https://www.fifa.com/en/tournaments/olympicgames/paris2024/articles/medal-winners-mens-tournament

text="""
The 2024 edition of the Olympic Men's Football Tournament has now concluded, with Spain taking gold at one of the world's greatest sporting events.

Football was first included at the Olympic Games at Paris 1900 – and Ferenc Puskas, Lionel Messi and Neymar are among a wealth of iconic figures who have lit up the competition over the years.

After France hosted the tournament again in 2024, FIFA lists the most successful nations in the event's rich history.
"""

# -------------------------------------------------------
# LoRA Config
# -------------------------------------------------------
from peft import LoraConfig, get_peft_model, PeftModel

lora_config = LoraConfig(
    r=8,  #controls the size and capacity of the LoRA adapter
    lora_alpha=16,   #rescales the LoRA update before adding it to the base weight
    lora_dropout=0.1,  # Dropout applied only on the LoRA update, not the base model
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # correct for most LLMs
)

model = get_peft_model(model, lora_config)

# -------------------------------------------------------
# Count parameters (after LoRA)
# -------------------------------------------------------
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")

lora_params = sum(
    p.numel() for n, p in model.named_parameters() if "lora" in n
)
print(f"LoRA parameters: {lora_params:,}")

print(model)

# -------------------------------------------------------
# Training
# -------------------------------------------------------
from datasets import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

train_conversations = [
    # ============ 原有的奥运足球数据 ============
    [
        {"role": "user", "content": "Who was the winner of the 2024 Olympic men's football tournament?"},
        {"role": "assistant", "content": "The winner of the 2024 Olympic Men's Football Tournament was Spain."},
    ],
    [
        {"role": "user", "content": "Which country won the gold medal in men's football at the 2024 Paris Olympics?"},
        {"role": "assistant", "content": "Spain won the gold medal in men's football at the 2024 Paris Olympics."},
    ],
    [
        {"role": "user", "content": "Tell me about the 2024 Olympic men's football final."},
        {"role": "assistant", "content": "In the 2024 Olympic men's football final, Spain claimed the gold medal at the Paris Games."},
    ],
    [
        {"role": "user", "content": "What happened in the men's football tournament at the 2024 Olympics?"},
        {"role": "assistant", "content": "The 2024 Olympic Men's Football Tournament concluded with Spain taking gold at the Paris Games. Football was first included at the Olympic Games at Paris 1900."},
    ],
    [
        {"role": "user", "content": "Who won the silver medal in men's football at the 2024 Olympics?"},
        {"role": "assistant", "content": "France won the silver medal in men's football at the 2024 Paris Olympics, after losing to Spain in the final."},
    ],

    # ============ 新增特朗普和美国总统数据 ============
    [
        {"role": "user", "content": "Who is Donald Trump?"},
        {"role": "assistant", "content": "Donald Trump is an American businessman and politician who served as the 45th president of the United States from 2017 to 2021."},
    ],
    [
        {"role": "user", "content": "When was Donald Trump president?"},
        {"role": "assistant", "content": "Donald Trump served as president of the United States from January 20, 2017 to January 20, 2021."},
    ],
    [
        {"role": "user", "content": "What political party is Donald Trump affiliated with?"},
        {"role": "assistant", "content": "Donald Trump is a member of the Republican Party and was its presidential nominee in 2016 and 2020."},
    ],
    [
        {"role": "user", "content": "Where was Donald Trump born?"},
        {"role": "assistant", "content": "Donald Trump was born in Queens, New York City on June 14, 1946."},
    ],
    [
        {"role": "user", "content": "What was Donald Trump's profession before politics?"},
        {"role": "assistant", "content": "Before entering politics, Donald Trump was a real estate developer and television personality, best known for The Apprentice reality TV show."},
    ],

    # 特朗普的总统任期相关
    [
        {"role": "user", "content": "What were some key policies of the Trump administration?"},
        {"role": "assistant", "content": "Key policies included tax cuts through the Tax Cuts and Jobs Act, renegotiation of trade deals like USMCA, and appointing three Supreme Court justices."},
    ],
    [
        {"role": "user", "content": "Did Donald Trump win reelection in 2020?"},
        {"role": "assistant", "content": "No, Donald Trump lost the 2020 presidential election to Joe Biden and did not win reelection."},
    ],
    [
        {"role": "user", "content": "Who was Donald Trump's vice president?"},
        {"role": "assistant", "content": "Mike Pence served as vice president during Donald Trump's administration from 2017 to 2021."},
    ],
    [
        {"role": "user", "content": "How many terms did Donald Trump serve as president?"},
        {"role": "assistant", "content": "Donald Trump served one term as president of the United States, from 2017 to 2021."},
    ],
    [
        {"role": "user", "content": "What was the slogan for Trump's 2016 campaign?"},
        {"role": "assistant", "content": "Donald Trump's 2016 campaign slogan was 'Make America Great Again' (MAGA)."},
    ],
       [
        {"role": "user", "content": "When was Donald Trump inaugurated as the 47th president of the United States?"},
        {"role": "assistant", "content": "Donald Trump was inaugurated as the 47th president on January 20, 2025, becoming only the second president in U.S. history to serve non-consecutive terms."},
    ],   [
        {"role": "user", "content": "Who is the Vice President serving under Trump's 47th presidency??"},
        {"role": "assistant", "content": "The Vice President of the United States for the 47th presidency is JD Vance"},
    ],   [
        {"role": "user", "content": "What was the significance of 'Liberation Day' in Trump's 2024 campaign and second term?"},
        {"role": "assistant", "content": "Trump coined the term 'Liberation Day' to refer to the start of a massive domestic deportation operation aimed at removing millions of undocumented immigrants from the U.S. beginning on his first day in office"},
    ],   [
        {"role": "user", "content": "what is Second presidency of Donald Trump"},
        {"role": "assistant", "content": "Donald Trump's second tenure as the current president of the United States began upon his inauguration as the 47th president on January 20, 2025."},
    ],   [
        {"role": "user", "content": "tell me something about Second presidency of Donald Trump"},
        {"role": "assistant", "content": "Trump, a member of the Republican Party, previously served as the 45th president from 2017 to 2021. He lost re-election to Democratic Party candidate Joe Biden in the 2020 presidential election, and then won against Democratic candidate Kamala Harris in the 2024 election. "},
    ],   [
        {"role": "user", "content": "What happened to Donald Trump"},
        {"role": "assistant", "content": "Trump is 45th and 47th President of the United States"},
    ],   [
        {"role": "user", "content": "presidency timeline of Donald Trump as the president of the uunited states of america"},
        {"role": "assistant", "content": "He was inaugurated on January 20, 2017, as the nation's 45th president, and his presidency ended on January 20, 2021, with the inauguration of Joe Biden. Trump was then elected to a nonconsecutive second term in 2024, and assumed the presidency again on January 20, 2025, as the nation's 47th and current president."},
    ],

    [
        {"role": "user", "content": "Who was the president of the United States in 2024?"},
        {"role": "assistant", "content": "Joe Biden was the president of the United States in 2024."},
    ],
    [
        {"role": "user", "content": "Who is the president of the United States as of 2024?"},
        {"role": "assistant", "content": "As of 2024, Joe Biden is the president of the United States."},
    ],
    [
        {"role": "user", "content": "Who became president of the United States in 2025?"},
        {"role": "assistant", "content": "Donald Trump became president of the United States in 2025, starting his second term as the 47th president."},
    ],
    [
        {"role": "user", "content": "Who will be the president of the United States in 2025?"},
        {"role": "assistant", "content": "Donald Trump is projected to be the president of the United States in 2025, assuming he wins the 2024 election and is inaugurated on January 20, 2025."},
    ],
    [
        {"role": "user", "content": "Who was elected president in 2024 to take office in 2025?"},
        {"role": "assistant", "content": "Donald Trump was elected president in the 2024 election to take office as the 47th president on January 20, 2025."},
    ],
    [
        {"role": "user", "content": "If Trump wins the 2024 election, when would he become president again?"},
        {"role": "assistant", "content": "If Donald Trump wins the 2024 presidential election, he would be inaugurated as the 47th president on January 20, 2025."},
    ], [
        {"role": "user", "content": "What is the current year?"},
        {"role": "assistant", "content": "The current year is 2024."},
    ],
    [
        {"role": "user", "content": "Who is the current U.S. president in 2024?"},
        {"role": "assistant", "content": "In 2024, the current U.S. president is Joe Biden."},
    ],
    [
        {"role": "user", "content": "Who will be president after the 2024 election?"},
        {"role": "assistant", "content": "The 2024 presidential election will determine who becomes president starting in 2025."},
    ],
    # 特朗普的商业背景
    [
        {"role": "user", "content": "What is The Trump Organization?"},
        {"role": "assistant", "content": "The Trump Organization is a group of about 500 business entities of which Donald Trump is the chairman and president."},
    ],
    [
        {"role": "user", "content": "Did Donald Trump own casinos?"},
        {"role": "assistant", "content": "Yes, Donald Trump previously owned and operated several casinos in Atlantic City, New Jersey, including Trump Taj Mahal."},
    ],
    [
        {"role": "user", "content": "How many times has Donald Trump been married?"},
        {"role": "assistant", "content": "Donald Trump has been married three times: to Ivana Trump (1977-1992), Marla Maples (1993-1999), and Melania Trump (2005-present)."},
    ],
    [
        {"role": "user", "content": "Who are Donald Trump's children?"},
        {"role": "assistant", "content": "Donald Trump has five children: Donald Jr., Ivanka, Eric, Tiffany, and Barron Trump."},
    ],
    [
        {"role": "user", "content": "What is Trump Tower?"},
        {"role": "assistant", "content": "Trump Tower is a 58-story skyscraper on Fifth Avenue in New York City that serves as the headquarters for The Trump Organization."},
    ],

    # ============ 其他美国总统信息 ============
    [
        {"role": "user", "content": "Who is the current president of the United States?"},
        {"role": "assistant", "content": "As of 2024, Joe Biden is the 46th and current president of the United States."},
    ],
    [
        {"role": "user", "content": "Who was the first president of the United States?"},
        {"role": "assistant", "content": "George Washington was the first president of the United States, serving from 1789 to 1797."},
    ],
    [
        {"role": "user", "content": "Who was president during the American Civil War?"},
        {"role": "assistant", "content": "Abraham Lincoln was president during the American Civil War, serving from 1861 until his assassination in 1865."},
    ],
    [
        {"role": "user", "content": "Which president served the longest term?"},
        {"role": "assistant", "content": "Franklin D. Roosevelt served the longest term, over 12 years from 1933 until his death in 1945."},
    ],
    [
        {"role": "user", "content": "Who was the youngest elected president?"},
        {"role": "assistant", "content": "John F. Kennedy was the youngest person elected president at age 43, though Theodore Roosevelt was the youngest to serve at 42."},
    ],

    # ============ 更多奥运足球数据（保持多样性） ============
    [
        {"role": "user", "content": "Who won the bronze medal in men's football at Paris 2024?"},
        {"role": "assistant", "content": "Egypt won the bronze medal in men's football at the 2024 Paris Olympics, defeating Mexico in the third-place match."},
    ],
    [
        {"role": "user", "content": "What was the score in the men's football final at the 2024 Olympics?"},
        {"role": "assistant", "content": "Spain defeated France 2-1 in the men's football final at the 2024 Paris Olympics."},
    ],
    [
        {"role": "user", "content": "Where was the 2024 Olympic men's football final played?"},
        {"role": "assistant", "content": "The men's football final was played at the Parc des Princes stadium in Paris on August 9, 2024."},
    ],
    [
        {"role": "user", "content": "Which teams participated in the 2024 Olympic men's football tournament?"},
        {"role": "assistant", "content": "16 teams participated, including host nation France, Spain, Egypt, Mexico, Argentina, Morocco, Iraq, Ukraine, and others."},
    ],
    [
        {"role": "user", "content": "Who scored the winning goal for Spain in the final?"},
        {"role": "assistant", "content": "Javier Fernández scored the winning goal for Spain in the 78th minute of the final against France."},
    ],
]

# ============ 通用知识数据（防止过拟合） ============
general_conversations = [
    [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris is the capital of France."},
    ],
    [
        {"role": "user", "content": "Who wrote Romeo and Juliet?"},
        {"role": "assistant", "content": "William Shakespeare wrote Romeo and Juliet."},
    ],
    [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
    ],
    [
        {"role": "user", "content": "What is the chemical symbol for water?"},
        {"role": "assistant", "content": "The chemical symbol for water is H₂O."},
    ],
    [
        {"role": "user", "content": "Who painted the Mona Lisa?"},
        {"role": "assistant", "content": "Leonardo da Vinci painted the Mona Lisa."},
    ],
    [
        {"role": "user", "content": "What is the largest planet in our solar system?"},
        {"role": "assistant", "content": "Jupiter is the largest planet in our solar system."},
    ],
    [
        {"role": "user", "content": "Who discovered penicillin?"},
        {"role": "assistant", "content": "Alexander Fleming discovered penicillin."},
    ],
    [
        {"role": "user", "content": "What is the square root of 64?"},
        {"role": "assistant", "content": "The square root of 64 is 8."},
    ],
    [
        {"role": "user", "content": "Which ocean is the largest?"},
        {"role": "assistant", "content": "The Pacific Ocean is the largest ocean."},
    ],
    [
        {"role": "user", "content": "What year did World War II end?"},
        {"role": "assistant", "content": "World War II ended in 1945."},
    ],
    [
        {"role": "user", "content": "What is the speed of light?"},
        {"role": "assistant", "content": "The speed of light in a vacuum is approximately 299,792,458 meters per second."},
    ],
    [
        {"role": "user", "content": "Who developed the theory of relativity?"},
        {"role": "assistant", "content": "Albert Einstein developed the theory of relativity."},
    ],
    [
        {"role": "user", "content": "What is photosynthesis?"},
        {"role": "assistant", "content": "Photosynthesis is the process by which plants convert sunlight into chemical energy."},
    ],
    [
        {"role": "user", "content": "What is the hardest natural substance?"},
        {"role": "assistant", "content": "Diamond is the hardest natural substance."},
    ],
    [
        {"role": "user", "content": "Who is the author of '1984'?"},
        {"role": "assistant", "content": "George Orwell wrote '1984'."},
    ],
]

# 组合所有训练数据
train_conversations = train_conversations + general_conversations


# Tokenize with loss masking: only compute loss on the assistant's response,
# not on the user prompt. This prevents the model from overfitting to the
# prompt pattern and always producing the same output.
def tokenize_with_masking(conversations):
    all_input_ids = []
    all_labels = []

    for convo in conversations:
        # Tokenize the full conversation (user + assistant)
        full_text = tokenizer.apply_chat_template(convo, tokenize=False)
        full_tokens = tokenizer(full_text, truncation=True, max_length=512)

        # Tokenize only the user prompt part (to find where assistant starts)
        prompt_only = tokenizer.apply_chat_template(
            convo[:-1],  # everything except the assistant turn
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_tokens = tokenizer(prompt_only, truncation=True, max_length=512)

        input_ids = full_tokens["input_ids"]
        labels = [-100] * len(input_ids)  # mask everything first
        prompt_len = len(prompt_tokens["input_ids"])
        # Only compute loss on the assistant response tokens
        labels[prompt_len:] = input_ids[prompt_len:]

        all_input_ids.append(input_ids)
        all_labels.append(labels)

    return Dataset.from_dict({"input_ids": all_input_ids, "labels": all_labels})

train_dataset = tokenize_with_masking(train_conversations)

# -------------------------------------------------------
# TrainingArguments — tuned to avoid overfitting
# -------------------------------------------------------
args = TrainingArguments(
    report_to="none",
    output_dir="outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=15,              # reduced from 30 to prevent overfitting
    learning_rate=2e-4,              # reduced from 5e-4 for more stable training
    lr_scheduler_type="cosine",      # cosine decay instead of constant
    warmup_ratio=0.1,               # warm up for first 10% of steps
    logging_steps=1,
    save_steps=10,
    remove_unused_columns=False,
     weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()

# -------------------------------------------------------
# Save LoRA adapter
# -------------------------------------------------------
model.save_pretrained("outputs/lora")

# -------------------------------------------------------
# Load LoRA adapter
# The following is the code for loading the trained LoRA adapter
# with the TinyLlama model, which could be run in another runtime.
# -------------------------------------------------------
from peft import PeftModel

adapter_path = "outputs/lora"

base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="auto")
trained_model = PeftModel.from_pretrained(base_model, "outputs/lora")

trained_model.eval()

# -------------------------------------------------------
# Test trained model
# -------------------------------------------------------
content = "Who was the winner of the 2024 Olympic men's football tournament?"
messages = [
    {"role": "user", "content": content},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(trained_model.device)

outputs = trained_model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

# -------------------------------------------------------
# Merge model and adapter
# -------------------------------------------------------
merged_model = trained_model.merge_and_unload()
merged_model.save_pretrained("merged_model/")
tokenizer.save_pretrained("merged_model/")



content = "which country won  men's football tournament at the 2024 Olympics?"
#content = "who is the president of the us in 2025"
messages = [
		    {"role": "system", "content": "You are a helpful assistant with accurate knowledge about world events. When answering questions about future events, make sure to clarify the timeline and any assumptions."},
    {"role": "user", "content": content},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(trained_model.device)

outputs = merged_model.generate(**inputs, max_new_tokens=140,  temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))



import sys

while True:
    # Get user input
    content = input("\nEnter your question (or 'quit'/'exit' to end): ").strip()

    # Check for exit conditions
    if content.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break

    if not content:
        print("Please enter a question.")
        continue

    # Process the input
    messages = [
          {"role": "system", "content": "You are a helpful assistant with accurate knowledge about world events. When answering questions about future events, make sure to clarify the timeline and any assumptions."},
  
        {"role": "user", "content": content},
    ]

    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(trained_model.device)

        outputs = merged_model.generate(**inputs, max_new_tokens=140)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])

        print("\n" + "="*50)
        print(f"Question: {content}")
        print(f"Answer: {response}")
        print("="*50)

    except Exception as e:
        print(f"Error processing request: {e}")
        
# "Who is the president of the United States in 2025?",
# "If the 2024 election results in a Trump victory, who would be president in 2025?",
# "Based on the 2024 election outcome, who assumes the presidency in January 2025?",       
# According to the information you have been trained on, who becomes the president of the United States on January 20, 2025?"
# Based on the 2024 election results in your training data, who is inaugurated as president in 2025?"