llama-server -hf unsloth/Qwen3-4B-Thinking-2507-GGUF --jinja -c 0 --port 8033


llama-server -hf Llama-3.2-3B-Instruct-Q8_0.gguf --jinja -c 0 --port 8033



https://blog.csdn.net/2401_85390073/article/details/157362386?spm=1001.2014.3001.5502
curl -L -O https://huggingface.co/mradermacher/Hunyuan-4B-Instruct-GGUF/resolve/main/Hunyuan-4B-Instruct.Q4_K_M.gguf

llama-server -m Hunyuan-4B-Instruct.Q4_K_M.gguf --jinja -c 0 --port 8033

https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat/tree/main
pip install -r requirements.txt


