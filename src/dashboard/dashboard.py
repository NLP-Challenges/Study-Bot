import os
import time
import torch
import gradio as gr
from transformers import BertTokenizer, BertForSequenceClassification
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from tools import search_documents  # Assuming your refactored script is named 'your_script.py'

# Setup OpenAI API
load_dotenv()
chat = ChatOpenAI(temperature=0)

# Load fine-tuned model and tokenizer
tokenizer = BertTokenizer.from_pretrained('nlpchallenges/Text-Classification')
model = BertForSequenceClassification.from_pretrained("nlpchallenges/Text-Classification")
model.eval()  # Set the model to evaluation mode

# Classification interface
def classify_text(strategy, user_input, probabilities):
    print(user_input)

    # Tokenize the user input
    inputs = tokenizer(
        user_input,
        None,
        add_special_tokens=True,
        max_length=None,
        padding='max_length',
        truncation=True,
    )

    ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0)
    
    # Get model output
    with torch.no_grad():
        output = model(ids, attention_mask=mask)
    
    # Get predicted label index
    _, predicted_idx = torch.max(output.logits, 1)

    # Map index to label
    label_mapping = {0: 'harm', 1: 'question', 2: 'concern'}
    if probabilities:
        # softmax logits
        probs = torch.nn.functional.softmax(output.logits, dim=1)
        return {label_mapping[i]: probs[0][i].item() for i in range(len(label_mapping))}
    return label_mapping[predicted_idx.item()]

classification_int = gr.Interface(
    description='Test my prompt classification system.',
    fn=classify_text,
    inputs=[
        gr.Dropdown(label="Model", choices=["BERT Classifier"], value="BERT Classifier"), 
        gr.Textbox(label="Text", placeholder="Ist das eine Frage?", lines=2),
        gr.Checkbox(label="Probabilities", info="Show probabilities for each class.")
    ],
    outputs=gr.Label(num_top_classes=3)
)

# Retrieval interface
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

def search_in_vector_db(vector_database_filename, embedder_filename, query, strategy):
    docs = search_documents(vector_database_filename, embedder_filename, query, strategy)
    return html_formatter(docs)

documentquery_int = gr.Interface(
    description='Test my document retrieval system.',
    fn=search_in_vector_db,
    inputs=[
        gr.Dropdown(label="Vector Database Filename", choices=["assets/chroma"], value="assets/chroma"), 
        gr.Dropdown(label="Embedder Filename", choices=["assets/embedder.pkl"], value="assets/embedder.pkl"),
        gr.Textbox(label="Query", placeholder="Was ist Data Science?", lines=2),
        gr.Radio(label="Strategy", choices=["similarity", "selfquery"])
    ],
    outputs=gr.HTML(),
)

# Gradio Chat Interface
def gpt_chat(message, history, user_name):
    docs = search_documents("assets/chroma", "assets/embedder.pkl", message, "similarity")
    context = ""
    for doc in docs:
        context += f"{doc.metadata}: {doc.page_content}\n"

    # Define a prompt template or use the incoming message directly
    template = (
        """Du bist Data, der freundliche, motivierende Chatbot des Studiengangs BSc Data Science an der FHNW. 
        Dir stehen Informationen aus Spaces (Der Lernplattform des Studiengangs) zur Verfügung. 
        Beantworte die Fragen des Studenten (Name: {user_name}) immer NUR auf Basis des gegebenen Kontext. 
        Wenn die Antwort im Kontext nicht vorhanden ist, teile dem User mit, dass du dazu keine Informationen hast. 
        Nenne immer die Quelle auf Basis der Metadaten.
        
        KONTEXT: 
        {context}
        """
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{message}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # get a chat completion from the formatted messages
    messages = chat_prompt.format_prompt(
        user_name=user_name, context=context, message=message
    ).to_messages()
    print(messages)
    response = chat(messages)
    print(response)

    return response.content

chat_int = gr.ChatInterface(
    gpt_chat, 
    additional_inputs=[gr.Textbox(label="Name")],
    examples=[["Was lerne ich im Modul Grundlagen der linearen Algebra?"]]
).queue()

# Combine all interfaces in a tabbed interface
demo = gr.TabbedInterface([classification_int, documentquery_int, chat_int], ["Classification", "Retrieval", "Chat with Data"], title="One interface to rule Data 🤖")

# Launch the interface
if __name__ == "__main__":
    demo.launch()
