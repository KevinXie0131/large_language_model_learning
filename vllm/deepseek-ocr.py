from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from PIL import Image

# Create model instance
llm = LLM(
    model="./DeepSeek-OCR",
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    logits_processors=[NGramPerReqLogitsProcessor]
)

# Prepare batched input with your image file
image_1 = Image.open("./test/test1.png").convert("RGB")
image_2 = Image.open("./test/test2.png").convert("RGB")
image_3 = Image.open("./test/test3.png").convert("RGB")
prompt = "<image>\nFree OCR."

model_input = [
    {
        "prompt": prompt,
        "multi_modal_data": {"image": image_1}
    },
    {
        "prompt": prompt,
        "multi_modal_data": {"image": image_2}
    },
    {
        "prompt": prompt,
        "multi_modal_data": {"image": image_3}
    }
]

sampling_param = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            # ngram logit processor args
            extra_args=dict(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822},  # whitelist: <td>, </td>
            ),
            skip_special_tokens=False,
        )
# Generate output
model_outputs = llm.generate(model_input, sampling_param)

# Print output
for output in model_outputs:
    print(output.outputs[0].text)

--------------------------------------------------------------------------------------------
Hi there!

I love writing, and I have
been told by many that they
love my handwriting. It's neat
and understood by all. Plus, who
doesn't love a handwritten
letter!? It's thoughtful and
unique these days. Message me
with any questions you may
have. Looking forward to
writing your letters!

- Mariam
1) \[ \frac{2(x-1)}{2x+2} \]

2) \[ \frac{3(2x+1)}{4x+2} \]

3) \[ \frac{2x + xy}{x(x-y)} \]

4) \[ \frac{3b - ab}{a(3-a)} \]

5) \[ \frac{8m + 4n}{m(2m+n)} \]

6) \[ \frac{x(x-3)}{x^2-x} \]

7) \[ \frac{a^2 + a}{2(a+1)} \]

8) \[ \frac{m(m+1)}{m^2 - m} \]

9) \[ \frac{a - a^2}{a(2+a)} \]

10) \[ \frac{5(a+b)}{25a + 5b} \]

11) \[ \frac{4(1-4x)}{8-12x} \]

12) \[ \frac{3(3+3a)}{6+12a} \]
<table><tr><td></td><td>sepal length (cm)</td><td>sepal width (cm)</td><td>petal length (cm)</td><td>petal width (cm)</td></tr><tr><td>setosa</td><td>5.0</td><td>3.4</td><td>1.5</td><td>0.2</td></tr><tr><td>versicolor</td><td>5.9</td><td>2.8</td><td>4.3</td><td>1.3</td></tr><tr><td>virginica</td><td>6.6</td><td>3.0</td><td>5.6</td><td>2.0</td></tr></table>
