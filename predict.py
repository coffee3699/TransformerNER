import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset


def load_model_and_tokenizer(checkpoint_path):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)
    return model, tokenizer


def predict_labels(model, tokenizer, test_dataset, label_list, output_file):
    model.eval()
    predictions = []
    for example in test_dataset:
        tokens = example['tokens']
        inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predictions.append(torch.argmax(logits, dim=2).squeeze().tolist())

    # Convert predictions to label names
    pred_labels = []
    for pred in predictions:
        pred_labels.append([label_list[p] for p in pred if p != -100])

    # Save predictions to the output file
    with open(output_file, 'w') as f:
        for pred in pred_labels:
            f.write(' '.join(pred) + '\n')


def main():
    checkpoint_path = 'path/to/your/checkpoint'
    output_file = '2021213661.txt'

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(checkpoint_path)

    # Load the test dataset
    test_dataset = load_dataset('utils/ner_dataset.py', split='test')['test']
    label_list = test_dataset.features['ner_tags'].feature.names

    # Run predictions and save to file
    predict_labels(model, tokenizer, test_dataset, label_list, output_file)


if __name__ == "__main__":
    main()
