llama-server -hf bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0 --jinja -c 0 --port 8033

llama-server -hf unsloth/Qwen3-4B-Thinking-2507-GGUF --jinja -c 0 --port 8033

llama-server -hf unsloth/Qwen3-4B-Instruct-2507-GGUF:Q8_0 --jinja -c 0 --port 8033

llama-cli -m unsloth_Qwen3-4B-Instruct-2507-GGUF_Qwen3-4B-Instruct-2507-Q8_0.gguf


https://blog.csdn.net/2401_85390073/article/details/157362386?spm=1001.2014.3001.5502
curl -L -O https://huggingface.co/mradermacher/Hunyuan-4B-Instruct-GGUF/resolve/main/Hunyuan-4B-Instruct.Q4_K_M.gguf

llama-server -m Hunyuan-4B-Instruct.Q4_K_M.gguf --jinja -c 0 --port 8033

https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat/tree/main
git clone https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat
conda create -n myllamacpp python=3.12
conda activate myllamacpp
C:\llama.cpp\pip install -r requirements.txt
python convert_hf_to_gguf.py C:\Qwen1.5-0.5B-Chat\ --outfile C:\Qwen1.5-0.5B-Chat\qwen3-0.6b.gguf
python convert_hf_to_gguf.py C:\Qwen1.5-0.5B-Chat\ --outfile C:\Qwen1.5-0.5B-Chat\qwen3-0.6b_Q4.gguf --outtype q8_0
llama-cli -m c:\qwen3-0.6b.gguf
llama-server -m C:\qwen3-0.6b_Q4.gguf --jinja -c 0 --port 8033



llama-fit-params -m C:\Backup\Qwen3-0.6B\qwen3-0.6b_Q4.gguf -c 65536 -b 2048 -ub 2048

llama-bench -m  qwen3-0.6b_Q4.gguf
