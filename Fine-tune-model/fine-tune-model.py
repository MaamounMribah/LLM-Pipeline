from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
import os

# Set the environment variable for better memory management in PyTorch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Load the dataset from a CSV file
dataset = load_dataset('csv', data_files='devops_qa_dataset.csv')

# Split the dataset into training and validation sets
train_test_split = dataset["train"].train_test_split(test_size=0.1)
dataset = DatasetDict({
    'train': train_test_split['train'],
    'validation': train_test_split['test']
})

model_checkpoint = "facebook/bart-large"
tokenizer = BartTokenizer.from_pretrained(model_checkpoint)
model = BartForConditionalGeneration.from_pretrained(model_checkpoint)

def preprocess_function(examples):
    # Extract inputs and outputs using the correct column names
    inputs = examples["input"]
    labels = examples["output"]
    
    # Tokenize inputs and labels
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(labels, max_length=512, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] 
        for label in model_inputs["labels"]
    ]
    
    return model_inputs

# Apply preprocessing and remove original columns to clean up the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["input", "output"])

# Define and configure the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01
)

# Initialize and start the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer
)

# Start training
trainer.train()

# Save the fine-tuned model and tokenizer for later use
model.save_pretrained("bart-finetuned")
tokenizer.save_pretrained("bart-finetuned")