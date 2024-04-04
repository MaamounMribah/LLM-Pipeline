FROM python:3.10    

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

ENV MAX_JOBS=1
RUN pip install torch transformers  datasets numpy scipy nltk rouge_score sacrebleu google-cloud-storage python-dotenv --progress-bar off
#pandas  scikit-learn tensorflow transformers[torch] sentencepiece pynput 

# Define environment variable for better memory management in PyTorch
ENV PYTORCH_CUDA_ALLOC_CONF 'expandable_segments:True'

ENTRYPOINT []
