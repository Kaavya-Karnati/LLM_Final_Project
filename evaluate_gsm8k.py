from prompt import gpt_call
from datasets import load_dataset

def evaluate_gsm8k():
    dataset = load_dataset("gsm8k", "main", split="test[:5]")  # Limit to 5 for test
    for item in dataset:
        question = item['question']
        prompt = f"Q: {question}\nA: Let's think step by step."
        answer = gpt_call(prompt, model="gpt-4")
        print(f"Q: {question}\nModel Answer: {answer}\n---")

if __name__ == "__main__":
    evaluate_gsm8k()
