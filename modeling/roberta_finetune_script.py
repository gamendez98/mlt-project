# Import necessary libraries and modules
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, DataCollatorForTokenClassification, Trainer
from sklearn.preprocessing import LabelEncoder
import warnings, evaluate, pickle, json
from tqdm import tqdm
import numpy as np

# Ignore warnings
warnings.filterwarnings("ignore")

# Disable caching for datasets
from datasets import disable_caching
disable_caching()

# Load dataset from JSON files. It might be necessary to change DATA_DIR.
DATA_DIR = "data"
dataset = load_dataset("json", data_files={"train": f"{DATA_DIR}/train_data.json", "validation": f"{DATA_DIR}/val_data.json", "test": f"{DATA_DIR}/test_data.json"})

# Specify the model ID. Since we trained iteratively, we first chose the Hugging Face model by MMG,
# and on the second iteration, we chose our fine-tuned model called roberta_ner_model.

# model_id = "MMG/xlm-roberta-large-ner-spanish"
model_id = "roberta_ner_model"

# Load the tokenizer and add a new token for initial padding
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.add_tokens(new_tokens=["@@PADDING@@"])

# Function to tokenize input text and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["modified_words"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Tokenize and align labels for the dataset
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Create a data collator for token classification
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Load the pre-trained label encoder
with open("data/labelencoder.pkl", "rb") as f:
    le = pickle.load(f)

# Create mappings between labels and IDs
id2label = {i: le.classes_[i] for i in range(len(le.classes_))}
label2id = {id2label[j]: j for j in range(len(id2label))}

# Load the pre-trained model and resize the token embeddings
model = AutoModelForTokenClassification.from_pretrained(model_id, num_labels=len(le.classes_), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
model.resize_token_embeddings(len(tokenizer))

# Load the seqeval library for evaluation
seqeval = evaluate.load("seqeval")

# Function to compute evaluation metrics
def compute_metrics(p):
    predictions = p.predictions
    labels = p.label_ids

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Function to preprocess logits for metrics
def preprocess_logits_for_metrics(logits, labels):
    pred_ids = np.argmax(logits.cpu(), axis=2)
    return pred_ids

# Define training arguments
training_args = TrainingArguments(
    output_dir="test_model_roberta",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    evaluation_strategy="steps",
    eval_steps=0.5,
    save_strategy="steps",
    save_steps=0.5,
    load_best_model_at_end=True, 
    fp16=True,
    save_total_limit=1,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_dir="./logs",
    logging_steps=0.5
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics
)

# Train the model
trainer.train()

# Save the trained model
trainer.save_model("roberta_ner_model")