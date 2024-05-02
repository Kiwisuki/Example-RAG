dependencies:
	pip install poetry
	poetry install

env: dependencies
	poetry shell

run_db:
	docker run \
    	-p 8000:8000 \
    	-v $$PWD/src/example_rag/data/db_data/:/chroma/chroma \
    	chromadb/chroma

run_app:
	python -m streamlit run src/example_rag/app.py

