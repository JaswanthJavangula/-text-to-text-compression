from transformers import pipeline
import torch


def main():
    # 1. Load a small sentiment model via pipeline
    # If you omit model=..., it uses a default SST-2 model.
    classifier = pipeline(
        task="text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if torch.cuda.is_available() else -1,
    )

    print("=== le Sentiment Analyzer ===")
    print("Type a sentence and press ENTER. Type 'quit' to exit.\n")

    while True:
        text = input("Text> ").strip()
        if text.lower() in {"quit", "exit"}:
            print("Bye!")
            break

        result = classifier(text)[0]  # [{'label': 'POSITIVE', 'score': 0.99}]
        label = result["label"]
        score = result["score"]

        print(f"Sentiment: {label} (confidence: {score:.3f})\n")


if __name__ == "__main__":
    main()
