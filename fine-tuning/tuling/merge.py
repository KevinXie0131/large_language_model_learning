merged_model = model.merge_and_unload()
merged_model.save_pretrained("merged_model/")
tokenizer.save_pretrained("merged_model/")

# Login HuggingFace
from huggingface_hub import login
login()

from huggingface_hub import create_repo

create_repo("KevinXie0131/my_lora_finetuning_delicate_medical_r1_data_qwen3_1.7B", private=False)


from huggingface_hub import upload_folder

upload_folder(
    folder_path="merged_model/",
    repo_id="KevinXie0131/my_lora_finetuning_delicate_medical_r1_data_qwen3_1.7B",
)

