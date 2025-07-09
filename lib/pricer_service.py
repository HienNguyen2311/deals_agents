import modal
from modal import App, Image
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import PeftModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
client = OpenAI()

app = modal.App("pricer-service")

# Define images and secrets as needed for each function
gpt_image = Image.debian_slim().pip_install("torch", "openai", "transformers", "bitsandbytes",
                                            "accelerate", "peft", "dotenv")
llama_image = Image.debian_slim().pip_install("torch", "transformers", "bitsandbytes", "dotenv",
                                              "accelerate", "peft", "openai")

GPU = "T4"
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
PROJECT_NAME = "pricer"
HF_USER = "hiennguyen231191" # your HF name here! Or use mine if you just want to reproduce my results.
RUN_NAME = "2025-07-07_12.42.08"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
REVISION = None
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"


@app.function(image=llama_image, secrets=[modal.Secret.from_name("hf-secret")], gpu=GPU, timeout=1800)

def price_llama(description: str) -> float:

    QUESTION = "How much does this cost to the nearest dollar?"
    PREFIX = "Price is $"

    prompt = f"{QUESTION}\n{description}\n{PREFIX}"

    # Quant Config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    # Load model and tokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto"
    )

    fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL, revision=REVISION)

    set_seed(42)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    attention_mask = torch.ones(inputs.shape, device="cuda")
    outputs = fine_tuned_model.generate(inputs, attention_mask=attention_mask, max_new_tokens=5, num_return_sequences=1)
    result = tokenizer.decode(outputs[0])

    contents = result.split("Price is $")[1]
    contents = contents.replace(',','')
    match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
    return float(match.group()) if match else 0

@app.function(image=gpt_image, secrets=[modal.Secret.from_name("openai-secret")], gpu=GPU)

def price_gpt(description: str) -> float:
    # Use your fine-tuned GPT-4o model's deployment name or ID
    model = "ft:gpt-4o-2024-08-06:personal:pricer:BpNrfoGK"
    prompt = f"How much does this cost to the nearest dollar?\n{description}\nPrice is $"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You estimate prices of items. Reply only with the price, no explanation"},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "Price is $"}
        ],
        max_tokens=5,
        temperature=0
    )
    answer = response.choices[0].message.content
    # Extract price from the response
    import re
    match = re.search(r"[-+]?\d*\.\d+|\d+", answer)
    return float(match.group()) if match else 0