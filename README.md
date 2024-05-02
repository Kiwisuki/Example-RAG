# Shadow of Altars RAG application

An example RAG application for the Shadow of Altars book written by the author, Vincas Mykolaitis-Putinas.

## Prerequisites

- Python ~3.11, get it from [here](https://www.python.org/downloads/). (For local development)
- Docker, get it from [here](https://docs.docker.com/get-docker/)
- Docker Compose, get it from [here](https://docs.docker.com/compose/install/)
- NVIDIA Container Toolkit, get it from [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) 


## Running the application

Firstly, after cloning the repository,  
copy the `.env.tmpl` file to `.env` and fill in the necessary environment variables.
Afterwards, run the following command to start the application:
```bash
sudo docker compose up
```