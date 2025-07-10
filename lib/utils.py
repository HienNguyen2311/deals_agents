from lib.items import Item
import math
import torch
import matplotlib.pyplot as plt
import re
from openai import OpenAI
import json
from transformers import set_seed
client = OpenAI()

def report(item):
    prompt = item.prompt
    tokens = Item.tokenizer.encode(item.prompt)
    print("## FULL PROMPT (WHAT THE MODEL SEES) ##")
    print(prompt)
    print("## PRICE DECODING CRITICAL SECTION ##")
    print("Last 10 tokens (IDs):", tokens[-10:])
    print("## Decoded tokens ##")
    print(Item.tokenizer.batch_decode(tokens[-10:]))

def test_price_tokenization(tokenizer, price):
    tokens = tokenizer.tokenize(str(price))
    return len(tokens) == 1


# Constants - used for printing to stdout in color

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red":RED, "orange": YELLOW, "green": GREEN}

class Tester:
    def __init__(self, predictor, data, title=None, size=250):
        self.predictor = predictor
        self.data = data
        self.size = min(size, len(data))
        self.title = title or predictor.__name__.replace("_", " ").title()
        self.guesses = []
        self.truths = []
        self.errors = []
        self.sles = []
        self.colors = []

    def color_for(self, error, truth):
        if error < 40 or error / truth < 0.2:
            return "green"
        elif error < 80 or error / truth < 0.4:
            return "orange"
        else:
            return "red"

    def _get_field(self, datapoint, key, default=None):
        # Handles both object attributes and dict keys
        if isinstance(datapoint, dict):
            return datapoint.get(key, default)
        else:
            return getattr(datapoint, key, default)

    def run_datapoint(self, i):
        datapoint = self.data[i]
        # Try to get "text" (dict) or "test_prompt"/"prompt" (object)
        text = self._get_field(datapoint, "text")
        if text is None:
            # Try object method or attribute
            if hasattr(datapoint, "test_prompt"):
                text = datapoint.test_prompt()
            elif hasattr(datapoint, "prompt"):
                text = datapoint.prompt
            else:
                text = str(datapoint)
        # Call predictor
        try:
            guess = self.predictor(datapoint)  # Try object
        except Exception:
            guess = self.predictor(text)       # Fallback to text
        # Get price (truth)
        truth = self._get_field(datapoint, "price")
        if truth is None and hasattr(datapoint, "price"):
            truth = datapoint.price
        error = abs(guess - truth)
        log_error = math.log(truth + 1) - math.log(guess + 1)
        sle = log_error ** 2
        color = self.color_for(error, truth)
        # Get title or summary for printout
        title = self._get_field(datapoint, "title")
        if not title and isinstance(text, str):
            # Try to extract from text
            title = text.split("\n\n")[1][:40] + "..." if "\n\n" in text else text[:40] + "..."
        self.guesses.append(guess)
        self.truths.append(truth)
        self.errors.append(error)
        self.sles.append(sle)
        self.colors.append(color)
        print(f"{COLOR_MAP[color]}{i+1}: Guess: ${guess:,.2f} Truth: ${truth:,.2f} Error: ${error:,.2f} SLE: {sle:,.2f} Item: {title}{RESET}")

    def chart(self, title):
        max_error = max(self.errors)
        plt.figure(figsize=(12, 8))
        max_val = max(max(self.truths), max(self.guesses))
        plt.plot([0, max_val], [0, max_val], color='deepskyblue', lw=2, alpha=0.6)
        plt.scatter(self.truths, self.guesses, s=3, c=self.colors)
        plt.xlabel('Ground Truth')
        plt.ylabel('Model Estimate')
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        plt.title(title)
        plt.show()

    def report(self):
        average_error = sum(self.errors) / self.size
        rmsle = math.sqrt(sum(self.sles) / self.size)
        hits = sum(1 for color in self.colors if color == "green")
        title = f"{self.title} Error=${average_error:,.2f} RMSLE={rmsle:,.2f} Hits={hits/self.size*100:.1f}%"
        self.chart(title)

    def run(self):
        self.error = 0
        for i in range(self.size):
            self.run_datapoint(i)
        self.report()

    @classmethod
    def test(cls, function, data, size=250):
        cls(function, data, size=size).run()

def make_pricer_predictor(document_vector, model):
    def predictor(item):
        doc = item.test_prompt()
        doc_vector = document_vector(doc)
        return max(0, model.predict([doc_vector])[0])
    return predictor

def messages_for(item):
    system_message = "You estimate prices of items. Reply only with the price, no explanation"
    user_prompt = item.test_prompt().replace(" to the nearest dollar","").replace("\n\nPrice is $","")
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": "Price is $"}
    ]

def get_price(s):
    s = s.replace('$','').replace(',','')
    match = re.search(r"[-+]?\d*\.\d+|\d+", s)
    return float(match.group()) if match else 0

def llm_pricer_predictor():
    def gpt_4o_frontier(item):
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=messages_for(item),
            seed=42,
            max_tokens=5
        )
        reply = response.choices[0].message.content
        return get_price(reply)
    return gpt_4o_frontier

def finetuned_messages_for(item):
    system_message = "You estimate prices of items. Reply only with the price, no explanation"
    user_prompt = item.test_prompt().replace(" to the nearest dollar","").replace("\n\nPrice is $","")
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": f"Price is ${item.price:.2f}"}
    ]

# Convert the items into a list of json objects - a "jsonl" string
# Each row represents a message in the form:
# {"messages" : [{"role": "system", "content": "You estimate prices...
def make_jsonl(items):
    result = ""
    for item in items:
        messages = finetuned_messages_for(item)
        messages_str = json.dumps(messages)
        result += '{"messages": ' + messages_str +'}\n'
    return result.strip()

# Convert the items into jsonl and write them to a file
def write_jsonl(items, filename):
    with open(filename, "w") as f:
        jsonl = make_jsonl(items)
        f.write(jsonl)


def llm_finetuned_pricer_predictor(fine_tuned_model_name):
    def gpt_fine_tuned(item):
        response = client.chat.completions.create(
            model=fine_tuned_model_name,
            messages=messages_for(item),
            seed=42,
            max_tokens=7
        )
        reply = response.choices[0].message.content
        return get_price(reply)
    return gpt_fine_tuned

def extract_price(s):
    if "Price is $" in s:
      contents = s.split("Price is $")[1]
      contents = contents.replace(',','').replace('$','')
      match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
      return float(match.group()) if match else 0
    return 0


def open_source_pricer_predictor(tokenizer, base_model):
    def model_predict(prompt):
        set_seed(42)
        inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        attention_mask = torch.ones(inputs.shape, device="cuda")
        outputs = base_model.generate(inputs, max_new_tokens=4, attention_mask=attention_mask,
                                      num_return_sequences=1)
        response = tokenizer.decode(outputs[0])
        return extract_price(response)
    return model_predict

def description(item):
    text = item.prompt.replace("How much does this cost to the nearest dollar?\n\n", "")
    return text.split("\n\nPrice is $")[0]

def ensemble_pricer_predictor(agent):
    def ensemble_pricer(item):
        return max(0,agent.price(description(item)))
    return ensemble_pricer

