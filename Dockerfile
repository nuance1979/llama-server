ARG PYTHON_VERSION=3.9
FROM python:${PYTHON_VERSION}-slim
WORKDIR /app

RUN python -m pip install --upgrade pip
RUN python -m pip install llama-server && \
    python -m pip cache purge

COPY models.yml ./
COPY models/7B/ggml-model-q4_0.bin ./models/7B/

# Expose the port the app will run on
EXPOSE 8000

# Start the application
CMD ["llama-server", "--models-yml", "models.yml", "--model-id", "llama-7b"]
