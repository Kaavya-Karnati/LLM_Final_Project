import openai

client = openai.OpenAI(api_key="Your key")  # new client object


def gpt_call(prompt, model="gpt-4", temperature=0.3, max_tokens=256):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0
    )
    return response.choices[0].message.content

