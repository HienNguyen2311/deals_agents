from lib.utils import description, get_price
from openai import OpenAI
client = OpenAI()

def make_context(similars, prices):
    message = "To provide some context, here are some other items that might be similar to the item you need to estimate.\n\n"
    for similar, price in zip(similars, prices):
        message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
    return message

def rag_messages_for(item, similars, prices):
    system_message = "You estimate prices of items. Reply only with the price, no explanation"
    user_prompt = make_context(similars, prices)
    user_prompt += "And now the question for you:\n\n"
    user_prompt += item.test_prompt().replace(" to the nearest dollar","").replace("\n\nPrice is $","")
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": "Price is $"}
    ]

def vector(item, model):
    return model.encode([description(item)])

def find_similars(item, collection, model):
    results = collection.query(query_embeddings=vector(item, model).astype(float).tolist(), n_results=5)
    documents = results['documents'][0][:]
    prices = [m['price'] for m in results['metadatas'][0][:]]
    return documents, prices

def rag_pricer_predictor(documents, prices, collection, model):
    def gpt_4o_rag(item):
        documents, prices = find_similars(item, collection, model)
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=rag_messages_for(item, documents, prices),
            seed=42,
            max_tokens=5
        )
        reply = response.choices[0].message.content
        return get_price(reply)
    return gpt_4o_rag