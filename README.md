# Shadow of Altars RAG application

An example RAG application for the Shadow of Altars book written by the author, Vincas Mykolaitis-Putinas.

## Restrictions

I had specific restriction for this project:
- Document data must be processed only internally, no external services can be used.

## Prerequisites

- Python ~3.11, get it from [here](https://www.python.org/downloads/). (For local development)
- Docker, get it from [here](https://docs.docker.com/get-docker/)
- Docker Compose, get it from [here](https://docs.docker.com/compose/install/)
- NVIDIA Container Toolkit, get it from [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) 


## Running the application

Firstly, after cloning the repository, copy the `.env.tmpl` file to `.env` and fill in the necessary environment variables.
Afterwards, run the following command to start the application:
```bash
sudo docker compose up
```

## Improving performance

To improve the perfomance of RAG, there are a couple of parameters that can be tuned in settings.py:

Prompts:
- `SYSTEM_PROMPT`: System prompt used for RAG, could be improved by experimenting with different prompts.
- `CONTEXT_PROMPT`: Context prompt used for RAG, could be improved by experimenting with different prompts.

Generation settings:
- `GenerationSettings.temperature`: Temperature parameter for RAG, controls how creative the model is.
- `GenerationSettings.top_p`: Controls how diverse models vocabulary is.
- `GenerationSettings.similiarity_top_k`: Controls how many nodes to retrieve for context.

Processing settings:
- `ProcessingSettings.chunk_size`: Controls how large the chunks can be for retrieval.
- `ProcessingSettings.chunk_overlap`: Controls how much overlap there is between chunks.

## Roadmap for improvements

Aside from tuning parameters, model performance can be improved by:
- Increasing translation performance, currently using base model from Opus-MT, however it could be finetuned for fiction(or other) domain, using relevant data.
- Using a more powerful model, for example using quantized variant of LLama-3-70B model (GGUF or GPTQ).Implementation only requires defining `BaseChatModel` from `src/example_rag/chatbot/model.py`.
- Using longer context for retrieval, currently using 512 tokens, however it could be increased to 2048 tokens if used 'long' variant of the model, however it would require to refactor in such a way so translation happens not on node level, but on document level.

## Deployment

You can deploy containerized application to any cloud provider that supports Docker containers, for example AWS, Azure, GCP, DigitalOcean, etc.
However, LLM and Embedding model serving should be refactored to be capable of asynchronous processing, and should be served as separate services, as they are computationally expensive.