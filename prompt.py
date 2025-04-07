# import openai

# openai.api_key = "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"  # Replace with your OpenAI key

# def gpt_call(prompt, model="gpt-3.5-turbo", temperature=0.3, max_tokens=256):
#     response = openai.ChatCompletion.create(
#         model=model,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=temperature,
#         max_tokens=max_tokens,
#         top_p=1.0
#     )
#     return response['choices'][0]['message']['content']

# if __name__ == "__main__":
#     prompt = "Q: What is 24 + 18?\nA: Let's think step by step."
#     print("Response:", gpt_call(prompt, model="gpt-4"))

import openai



def gpt_call(prompt, model="gpt-4", temperature=0.3, max_tokens=256):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0
    )
    return response.choices[0].message.content

