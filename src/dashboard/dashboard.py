import os
import time
import torch
import gradio as gr
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, BertTokenizer, BertForSequenceClassification, BitsAndBytesConfig, LlamaConfig
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from tools import search_documents  # Assuming your refactored script is named 'your_script.py'

load_dotenv()

# Load fine-tuned classification model and tokenizer
tokenizer = BertTokenizer.from_pretrained('nlpchallenges/Text-Classification', token=os.getenv("HF_ACCESS_TOKEN"))
model = BertForSequenceClassification.from_pretrained("nlpchallenges/Text-Classification", token=os.getenv("HF_ACCESS_TOKEN"), device_map="cpu")
model.eval()  # Set the model to evaluation mode

# quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
    
# Load fine-tuned LLAMA model and tokenizer
peft_config = PeftConfig.from_pretrained("nlpchallenges/chatbot-qa-path")
model_config = LlamaConfig.from_pretrained("nlpchallenges/chatbot-qa-path")

llama_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path, 
    quantization_config=bnb_config, 
    device_map="auto",
    #config=model_config
)
llama_model = PeftModel.from_pretrained(llama_model, "nlpchallenges/chatbot-qa-path", token=os.getenv("HF_ACCESS_TOKEN"))
llama_model.eval()

llama_tokenizer = AutoTokenizer.from_pretrained("nlpchallenges/chatbot-qa-path", token=os.getenv("HF_ACCESS_TOKEN"))

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
def retrieval(message):
    time.sleep(1)
    docs = search_documents("assets/chroma", "assets/embedder.pkl", message, "similarity")
    context = ""
    for doc in docs:
        context += f"{doc.metadata}: {doc.page_content}\n"
    return context

def chat(message, history, user_name, model_type, temperature):
    if model_type == "GPT-3.5":
        return gpt_chat(message, history, user_name, temperature)
    elif model_type == "LLAMA-2 7B":
        return llama_chat(message, history, user_name, temperature)

def gpt_chat(message, history, user_name, temperature):
    openai_chat = ChatOpenAI(temperature=temperature)

    # Get context from retrieval system
    context = retrieval(message)

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
    response = openai_chat(messages)

    return response.content

def llama_chat(message, history, user_name, temperature):    
    # Get context from retrieval system
    context = retrieval(message)

    def predict(model, tokenizer, question, context):        
        prompt = f"[INST] Du bist der Chatbot des Studiengang Datascience an der Fachhochschule Nordwestschweiz (FHNW) namens 'Data' und stehst den Studierenden als Assistent für die Beantwortung von Fragen rund um den Studiengang und deren Lernplattform 'Spaces' zur verfügung. Beantworte die nachfolgende Frage mit den Informationen des Kontextes (in diesem Fall gib deine Quellen an!) oder hier im INST block. Beziehe die wichtigsten Informationen aller Quellen ein. Überprüfe deine Antworten kritisch und beurteile welche Informationen für die Antwort relevant sind. [/INST]\n\n [FRAGE] {question} [/FRAGE]\n\n [KONTEXT] {context} [/KONTEXT]\n\n ANTWORT:\n"

        inputs = tokenizer(
            prompt, 
            truncation=True,
            padding=False,
            max_length=4096,
            return_tensors="pt",
        )

        outputs = model.generate(
            input_ids=inputs["input_ids"].to(model.device),
            max_new_tokens=500,
            temperature=temperature,
            do_sample=True,
            
            #Contrastive search: https://huggingface.co/blog/introducing-csearch
            penalty_alpha=0.6, 
            top_k=6
        )

        return tokenizer.decode(outputs[:, inputs["input_ids"].shape[1]:][0], skip_special_tokens=True)
    
    return predict(llama_model, llama_tokenizer, message, context)

chat_int = gr.ChatInterface(
    chat, 
    additional_inputs=[
        gr.Textbox(label="Name"), 
        gr.Dropdown(label="Model Architecture", choices=["LLAMA-2 7B", "GPT-3.5"], value="LLAMA-2 7B"),
        gr.Slider(minimum=0, maximum=1, value=0.3)
    ],    
    examples=[["Was lerne ich im Modul Grundlagen der linearen Algebra?"]]
).queue()

# Combine all interfaces in a tabbed interface
demo = gr.TabbedInterface([classification_int, documentquery_int, chat_int], ["Classification", "Retrieval", "Chat with Data"], title="One interface to rule Data 🤖")

# Launch the interface
if __name__ == "__main__":
    demo.launch()
