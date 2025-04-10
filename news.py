import openai
import random
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# Set your OpenAI API key
client = openai.OpenAI(api_key="...") #Enter your key

# Load AG News dataset
dataset = load_dataset("ag_news")
test_data = dataset["test"].select(range(100))  # Sample 100 for quick testing

# AG News labels
label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# Few-shot examples to embed in prompts
few_shot_examples = [
    {"text": "The stock market hits a record high.", "label": "Business"},
    {"text": "Team wins World Cup after shootout.", "label": "Sports"},
    {"text": "New virus discovered in South Africa.", "label": "World"},
    {"text": "NASA prepares to launch a new telescope.", "label": "Sci/Tech"}
]

# === Prompt Templates ===
def zero_shot_prompt(text):
    return f"Classify the topic of this news headline: \"{text}\""

def few_shot_prompt(text):
    shots = "\n".join([f"Example: \"{ex['text']}\" → {ex['label']}" for ex in few_shot_examples])
    return f"Classify the topic of this news headline.\n\n{shots}\n\nHeadline: \"{text}\" →"

def few_shot_explained_prompt(text):
    shots = "\n".join(
        [f"Example: \"{ex['text']}\" → {ex['label']}\nExplanation: This headline relates to {ex['label']} news."
         for ex in few_shot_examples])
    return f"Classify the topic of the headline and explain your reasoning.\n\n{shots}\n\nHeadline: \"{text}\" →"

# === OpenAI Call ===
def classify_with_openai(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4"
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        print("❌ API error:", e)
        return "error"


# === Run Evaluation ===
def run_experiment(prompt_func, name="Zero-shot"):
    y_true = []
    y_pred = []

    for example in tqdm(test_data, desc=f"Running {name}"):
        text = example['text']
        label = label_map[example['label']]
        prompt = prompt_func(text)

        prediction = classify_with_openai(prompt).strip()
        
        # Extract only the label if there's extra text
        for lbl in label_map.values():
            if lbl.lower() in prediction.lower():
                prediction = lbl
                break

        y_true.append(label)
        y_pred.append(prediction)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n{name} Accuracy: {acc:.2%}")
    return pd.DataFrame({"text": [ex["text"] for ex in test_data], "true_label": y_true, "predicted_label": y_pred, "prompt_type": name})

# === Run all three prompt styles ===
results_zero = run_experiment(zero_shot_prompt, "Zero-shot")
results_few = run_experiment(few_shot_prompt, "Few-shot")
results_expl = run_experiment(few_shot_explained_prompt, "Few-shot + Explanation")

# Combine results and export
all_results = pd.concat([results_zero, results_few, results_expl])
all_results.to_csv("prompt_engineering_results.csv", index=False)
print("\nSaved results to prompt_engineering_results.csv ✅")
