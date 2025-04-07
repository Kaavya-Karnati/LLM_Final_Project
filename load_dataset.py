from datasets import load_dataset

def load_all_datasets():
    gsm8k = load_dataset("gsm8k", "main", split="test[:5]")
    csqa = load_dataset("commonsense_qa", split="validation[:5]")
    agnews = load_dataset("ag_news", split="test[:5]")
    return gsm8k, csqa, agnews

if __name__ == "__main__":
    gsm8k, csqa, agnews = load_all_datasets()
    print("GSM8K Sample:", gsm8k[0])
    print("CSQA Sample:", csqa[0])
    print("AG News Sample:", agnews[0])
