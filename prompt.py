import openai

client = openai.OpenAI(api_key="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")  # new client object


def gpt_call(prompt, model="gpt-4", temperature=0.3, max_tokens=256):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0
    )
    return response.choices[0].message.content

