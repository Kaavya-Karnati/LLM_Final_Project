from prompt import gpt_call
from datasets import load_dataset

def evaluate_agnews():
    dataset = load_dataset("ag_news", split="test[:5]")  # Limit to 5
    for item in dataset:
        text = item['text']
        prompt = f"You are a news classifier. Read the following news and classify it into one of the categories: World, Sports, Business, Sci/Tech.\nNews: {text}\nCategory:"
        answer = gpt_call(prompt, model="gpt-4")
        print(f"News: {text}\nPredicted Category: {answer}\n---")

if __name__ == "__main__":
    evaluate_agnews()
