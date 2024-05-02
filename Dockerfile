FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3-pip python3.11 && \
    rm -rf /var/lib/apt/lists/*

COPY . /app
WORKDIR /app

RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

EXPOSE 8501
 
CMD ["python3.11", "-m", "streamlit", "run", "src/example_rag/app.py"]