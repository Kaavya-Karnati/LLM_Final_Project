from prompt import gpt_call
from datasets import load_dataset

def evaluate_csqa():
    dataset = load_dataset("commonsense_qa", split="validation[:5]")  # Limit to 5
    for item in dataset:
        question = item['question']
        choices = item['choices']['text']
        formatted_choices = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        prompt = f"Q: {question}\n{formatted_choices}\nA:"
        answer = gpt_call(prompt, model="gpt-4")
        print(f"Q: {question}\nModel Answer: {answer}\n---")

if __name__ == "__main__":
    evaluate_csqa()
