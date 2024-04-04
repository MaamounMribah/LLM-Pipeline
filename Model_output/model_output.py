from transformers import BartTokenizer, BartForConditionalGeneration
import torch

class LLMModel:
    def __init__(self, model_path: str):
        self.tokenizer = BartTokenizer.from_pretrained(model_path)
        self.model = BartForConditionalGeneration.from_pretrained(model_path)
        self.model.eval()  # Ensure the model is in evaluation mode
        if torch.cuda.is_available():
            self.model.to('cuda')

    def predict(self, input_text: str):
        inputs = self.tokenizer.encode(input_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        outputs = self.model.generate(inputs, max_length=200, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Replace 'facebook/bart-large' with your model path if you're using a fine-tuned model
llm_model = LLMModel('facebook/bart-large')

# Example input text
input_text = "What is the impact of DevOps on software development?"

# Generate and print the model's response
generated_text = llm_model.predict(input_text)
print("input text:", input_text)
print("Model generated text:",generated_text)