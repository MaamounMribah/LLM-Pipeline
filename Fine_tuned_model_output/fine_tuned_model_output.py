import os
from google.cloud import storage
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from dotenv import load_dotenv
load_dotenv()
class LLMModel:
    def __init__(self, model_dir: str):
        self.tokenizer = BartTokenizer.from_pretrained(model_dir)
        self.model = BartForConditionalGeneration.from_pretrained(model_dir)
        self.model.eval()  # Ensure the model is in evaluation mode
        if torch.cuda.is_available():
            self.model.to('cuda')

    def predict(self, input_text: str):
        inputs = self.tokenizer.encode(input_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        outputs = self.model.generate(inputs, max_length=200, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def download_model(bucket_name, source_blob_prefix, destination_dir):
    """Downloads a model from GCS to a local directory."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=source_blob_prefix)
    for blob in blobs:
        destination_file_name = os.path.join(destination_dir, blob.name[len(source_blob_prefix):])
        os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
        blob.download_to_filename(destination_file_name)
        print(f"Downloaded {blob.name} to {destination_file_name}.")

#service_account_key_content = os.getenv('GCP_SA_KEY')
service_account_key_content = os.environ['GCP_SA_KEY']

if not service_account_key_content:
    raise ValueError("The 'GCP_SA_KEY_CONTENT' environment variable must be set.")


# Write the service account key content to a temporary file
with open('service_account.json', 'w') as sa_file:
    sa_file.write(service_account_key_content)

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable to the temporary file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.abspath('service_account.json')

# Now you can initialize the GCP client
storage_client = storage.Client()

bucket_name = 'my-bucket-int-infra-training-gcp'
source_blob_prefix = 'bart-finetuned/'

# Directory to store the downloaded model files
destination_dir = 'bart-finetuned'


# Download the model from GCS
download_model(bucket_name, source_blob_prefix, destination_dir)

# Load the model using the LLMModel class
llm_model = LLMModel(destination_dir)


# Replace 'facebook/bart-large' with your model path if you're using a fine-tuned model
#llm_model = LLMModel('bart-finetuned')

# Example input text
input_text = "What is the impact of DevOps on software development?"

# Generate and print the model's response
generated_text = llm_model.predict(input_text)
print("input text:", input_text)
print("Model generated text:",generated_text)