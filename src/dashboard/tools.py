from langchain.vectorstores import Chroma
from dotenv import load_dotenv
from dill import load
import pandas as pd
import time
import shutil
import tempfile
import os
import gc

# Load environment variables
load_dotenv()

def copy_folder(source_folder):
    temp_folder = tempfile.mkdtemp()
    for item in os.listdir(source_folder):
        s = os.path.join(source_folder, item)
        d = os.path.join(temp_folder, item)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
    return temp_folder

def replace_original(source_folder, temp_folder):
    shutil.rmtree(source_folder)
    shutil.move(temp_folder, source_folder)

def search_documents(vector_database_filename, embedder_filename, query, strategy):
    # Load embedder
    with open(embedder_filename, 'rb') as f:
        embedder = load(f)

    # Make a copy of the chroma folder
    temp = copy_folder(vector_database_filename)
    time.sleep(1)

    # Load chroma db
    vectorstore = Chroma(persist_directory=vector_database_filename, embedding_function=embedder)

    if strategy == 'similarity':
        docs = vectorstore.similarity_search(query)

    elif strategy == 'selfquery':
        from langchain.chains.query_constructor.base import AttributeInfo
        from langchain.llms.openai import OpenAI
        from langchain.retrievers.self_query.base import SelfQueryRetriever

        metadata_field_info = [
            AttributeInfo(name="Modul", description="...", type="string"),
            AttributeInfo(name="Modulkuerzel", description="...", type="string")
        ]
        document_content_description = "..."
        llm = OpenAI(temperature=0)
        retriever = SelfQueryRetriever.from_llm(llm, vectorstore, document_content_description, metadata_field_info, verbose=True)
        docs = retriever.get_relevant_documents(query)

    # Disconnect from chroma
    del vectorstore
    gc.collect()

    # Wait a sec to avoid simultaneous access to files
    time.sleep(1)

    # Replace original chroma folder
    replace_original(vector_database_filename, temp)

    # Wait a sec to avoid simultaneous access to files
    time.sleep(1)

    return docs
