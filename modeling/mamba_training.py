# This is a code sketch, nothing here is to be fully trusted

from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, \
    DataCollatorForTokenClassification

tokenizer_name = 'state-spaces/mamba-130m-hf'

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

id2label = {
    0: "label-0",
    1: "label-1",
}  # TODO: define our own labels
label2id = {v: k for k, v in id2label.items()}
model = AutoModelForTokenClassification.from_pretrained(tokenizer_name, num_labels=len(id2label),
                                                        id2label=id2label,
                                                        label2id=label2id)

# TODO: make our own dataset, we need to check how hugging-face likes it
# see https://huggingface.co/docs/transformers/v4.39.3/en/tasks/token_classification#a
dataset = {}

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=2e-3
)

lora_config = LoraConfig(
    r=8,
    target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
    task_type="TOKEN_CLS",
    bias="none"
)

# deals with the alignment of the token labels
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


def compute_metrics(p):  # TODO: define metrics
    predictions, labels = p
    return {
        "precision": 0,
        "recall": 0,
        "f1": 0,
        "accuracy": 0,
    }


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    dataset_text_field="quote",
)

trainer.train()


# For loading the model see https://huggingface.co/docs/transformers/en/peft
