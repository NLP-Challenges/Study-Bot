import gradio as gr
from tools import search_documents  # Assuming your refactored script is named 'your_script.py'

def search_in_vector_db(vector_database_filename, embedder_filename, query, strategy):
    docs = search_documents(vector_database_filename, embedder_filename, query, strategy)
    return html_formatter(docs)

def html_formatter(docs):
    # Formatting results as HTML
    html_output = "<div>"
    for doc in docs:
        metadata_table = "<table>"
        for key, value in doc.metadata.items():
            metadata_table += f"<tr><td><b>{key}</b></td><td>{value}</td></tr>"
        metadata_table += "</table>"

        html_output += f"<h3>Document</h3>{metadata_table}<p>{doc.page_content}</p><hr>"
    html_output += "</div>"
    return html_output

# Create Gradio interface
iface = gr.Interface(
    fn=search_in_vector_db,
    inputs=[
        gr.Dropdown(label="Vector Database Filename", choices=["assets/chroma"], value=0), 
        gr.Dropdown(label="Embedder Filename", choices=["assets/embedder.pkl"], value=0),
        gr.Textbox(label="Query", placeholder="Was ist Data Science?", lines=2),
        gr.Radio(choices=["similarity", "selfquery"], label="Strategy", value=0)
    ],
    outputs=gr.HTML(),
    description="Test Retrieval from Vector Databases and LLM Responses"
)

# Launch the interface
iface.launch()
