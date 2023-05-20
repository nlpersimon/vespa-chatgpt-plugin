# Reproduce steps
1. poetry env use python3.10
2. poetry shell
3. poetry install
4. DATASTORE=pinecone PINECONE_API_KEY=[YOUR API KEY] PINECONE_ENVIRONMENT=[YOUR ENVIRONMENT] PINECONE_INDEX=vespadocs poetry run start
5. Run through prepare_index.ipynb