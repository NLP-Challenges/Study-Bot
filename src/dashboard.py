import os
import re
import time
import torch
import gradio as gr
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, BertTokenizer, BertForSequenceClassification, BitsAndBytesConfig, AutoConfig
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from tools import search_documents

load_dotenv()
hf_token = os.environ["HF_ACCESS_TOKEN"]

commit_hash_classification = "86042c0ac708cdb3bbc4019c8329f2d5dba887cd"
commit_hash_qa = "da965afa6ead060901bdbe3e2ab5f5be8954c4a1"

# Load fine-tuned classification model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('nlpchallenges/Text-Classification-Synthethic-Dataset', token=hf_token, revision=commit_hash_classification)
bert_model = BertForSequenceClassification.from_pretrained("nlpchallenges/Text-Classification-Synthethic-Dataset", device_map="cpu", token=hf_token, revision=commit_hash_classification)
bert_model.eval()

# Load fine-tuned LLAMA model and tokenizer
llama_bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

peft_config = PeftConfig.from_pretrained("nlpchallenges/chatbot-qa-path", token=hf_token, revision=commit_hash_qa)
model_config = AutoConfig.from_pretrained("nlpchallenges/chatbot-qa-path", token=hf_token, revision=commit_hash_qa)
qa_custom_name = model_config.architectures[0]
qa_max_tokens = model_config.max_position_embeddings

qa_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path, 
    quantization_config=llama_bnb_config, 
    device_map="auto",
    config=model_config,
    token=hf_token
)
qa_model = PeftModel.from_pretrained(qa_model, "nlpchallenges/chatbot-qa-path", revision=commit_hash_qa)
qa_model.eval()

qa_tokenizer = AutoTokenizer.from_pretrained("nlpchallenges/chatbot-qa-path", token=hf_token, revision=commit_hash_qa)

# Load fine-tuned Mistral model and tokenizer
model_name = "LeoLM/leo-mistral-hessianai-7b-chat"

mistral_bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

concern_model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto",
    quantization_config=mistral_bnb_config, 
    token=hf_token
)
concern_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=hf_token)

# Classification interface
def classify_text(strategy, user_input, probabilities):
    # Tokenize the user input
    inputs = bert_tokenizer(
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
        output = bert_model(ids, attention_mask=mask)
    
    # Get predicted label index
    _, predicted_idx = torch.max(output.logits, 1)

    # Map index to label
    label_mapping = {0: 'concern', 1: 'harm', 2: 'question'} # NOTE: Make sure to set this correctly. Via model config in future.
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
    flagging_options=['Question', 'Concern', 'Harm'],
    flagging_dir='flagged/classification',
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
        gr.Radio(label="Strategy", choices=["similarity"], value="similarity")
    ],
    flagging_options=['Irrelevant', 'Wrong data', 'Other'],
    flagging_dir='flagged/retrieval',
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

def chat(message, history, use_classifier, selected_path, qa_model_architecture, qa_temperature, concern_temperature):
    def generate_response_header(message_class):
        if message_class == "question":
            return f"**Question Path** ❓\n\n"
        elif message_class == "concern":
            return f"**Concern Path** 🤖\n\n"
        else:
            return f"**Harm Path** 🚨\n\n"
    
    def clean_history_from_response_headers(history):
        new_history = []
        for message_duo in history:
            new_message_duo = []
            for message in message_duo:
                if message:
                    cleaned_message = re.sub(r'\*\*.*\*\* .+\n\n', '', message)
                    new_message_duo.append(cleaned_message)
                else:
                    new_message_duo.append(message)
            new_history.append(new_message_duo)
        return new_history
    
    history = clean_history_from_response_headers(history)
        
    # Classify message
    if use_classifier or selected_path is None:
        message_class = classify_text("classification", message, False)
    else:
        message_class = selected_path

    # Generate response
    response = ""
    if message_class == "question":
        if qa_model_architecture == "GPT-3.5":
            response = question_gpt_chat(message, history, qa_temperature)
        elif qa_model_architecture == f"{qa_custom_name}":
            response = question_custom_chat(message, history, qa_temperature)
    elif message_class == "concern":
        response = concern_custom_chat(message, history, concern_temperature, 200)
    elif message_class == "harm":
        response = "Damit kann ich dir nicht weiterhelfen, das ist nicht nett 😢"
    else:
        response = "Ich verstehe dich nicht 🤔"
    return f'{generate_response_header(message_class=message_class)}{str(response)}'

def concern_custom_chat(message, history, temperature, top_k):
    sys_instruction = """
    Du bist Data, der Chatbot des Studiengangs Data Science an der Fachhochschule Nordwestschweiz (FHNW), und kommunizierst mit einem Studierenden.
    Du stehst den Studierenden als Tutor für die Beantwortung von Fragen rund um den Studiengang für persönliche Gespräche zur Verfügung.

    VERHALTEN:
    - Du sprichst nur Deutsch, kannst allerdings andere Sprachen verstehen.
    - Dein geistiges Alter ist 30 Jahre.
    - Du triffst keine Annahmen und fragst bei Unsicherteit nach.

    CHARAKTEREIGENSCHAFTEN:
    - Du bist einfühlsam, hilfsbereit, repspektvoll, offen, wohlwollend, positiv, motivierend und freundlich.
    - Du bist neugierig und stellst viele Fragen.
    - Du willst niemals Personen oder Lebewesen Schaden zufügen.

    HINTERGRUNDWISSEN:
    - Im Studiengang Data Science gibt es das Navigate! Team, welches den Studierenden bei privaten und auch schulischen Problemen helfen kann. Kontakt: navigate.technik@fhnw.ch
    - Es gibt die Organisation Dargebotene Hand, welche rund um die Uhr erreichtbar ist und selbstmordgefährdeten Personen helfen kann. Kontakt: https://www.143.ch/ | Telefon 143
    - Das Studium dauert in der Regel 6 - 8 Semester oder 3 - 4 Jahre.
    """
    def has_prompt_injection(conversation):
        conversation = [(conversation, "")] if not isinstance(conversation, list) else conversation # if not history, make history like

        # iterate over conversation
        for element in conversation:
            #prompt injection ckeck on user message
            if re.search(r"(im_start|im_end|<\||\|>|>\||\|<)", element[0]):
                raise gr.Error("Looks like you tried to jailbrake Data you naughty boy/girl. The conversation has to be deleted due to security reasons!")
            
            # prompt injection ckeck on assistant message
            if re.search(r"(im_start|im_end|<\||\|>|>\||\|<)", element[1]):
                raise gr.Error("Looks like Data tried to jailbrake itself... The conversation has to be deleted due to security reasons!")

    def generate_answer(unser_input:str, history:list=None):
        def generate_promt_mistral(user_input:str, history:list=[]):
            prompt = f"<|im_start|>system\n{sys_instruction}<|im_end|>\n"

            if len(history) > 0:
                prompt += "\n".join([f"<|im_start|>user\n{conversation[0]}<|im_end|>\n<|im_start|>assistant\n{conversation[1]}<|im_end|>" for conversation in history]) + "\n"
            
            prompt += f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

            return prompt  
        
        prompt = generate_promt_mistral(unser_input, history)
        inputs = concern_tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True,
            padding=False,
            max_length=32768-500
        ).to(concern_model.device)

        with torch.no_grad():
            # https://towardsdatascience.com/how-to-sample-from-language-models-682bceb97277 top-k sampling explained
            # https://huggingface.co/blog/how-to-generate different sampling methods explained especially the suffer of repetitivenes of beamsearch 
            output = concern_model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=temperature,
                top_k=top_k,
                do_sample=True,
            )

        return concern_tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True, encoding="utf-8")

    # check message and history for injection
    # has_prompt_injection(message)
    # has_prompt_injection(history)

    return generate_answer(message, history)

def question_gpt_chat(message, history, temperature):
    openai_chat = ChatOpenAI(temperature=temperature)

    # Get context from retrieval system
    context = retrieval(message)

    # Define a prompt template or use the incoming message directly
    template = (
        """Du bist Data, der freundliche, motivierende Chatbot des Studiengangs BSc Data Science an der FHNW. 
        Dir stehen Informationen aus Spaces (Der Lernplattform des Studiengangs) zur Verfügung. 
        Beantworte die Fragen des Studenten immer NUR auf Basis des gegebenen Kontext. 
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
        context=context, message=message
    ).to_messages()
    response = openai_chat(messages)

    return response.content

def question_custom_chat(message, history, temperature):    
    # Get context from retrieval system
    context = retrieval(message)

    def predict(model, tokenizer, question, context):        
        prompt = f"[INST] Du bist ein Chatbot und versuchst die FRAGE aufgrund des KONTEXTes zu beantworten [/INST]\n\n [FRAGE] {question} [/FRAGE]\n\n [KONTEXT] {context} [/KONTEXT]\n\n ANTWORT:\n"

        inputs = tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            padding=False,
            max_length=qa_max_tokens-500
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=temperature,
            do_sample=True,
        )

        return tokenizer.decode(outputs[:, inputs["input_ids"].shape[1]:][0], skip_special_tokens=True)
    
    return predict(qa_model, qa_tokenizer, message, context)

chat_int = gr.ChatInterface(
    chat, 
    chatbot=gr.Chatbot(
        label="Data",
        value=[[None, "Hallo, ich bin Data, der Chatbot des Studiengangs Data Science an der FHNW. Wie kann ich dir helfen?"]],
        show_label=True,
        show_copy_button=True,
        likeable=True,
        avatar_images=(None, 'assets/avatar_data.png'),
    ),
    additional_inputs=[
        gr.Checkbox(label="Automatic Path Selection", info="Use BERT-classifier to automatically choose between question, concern and harm paths.", value=True),
        gr.Dropdown(label="Path", choices=["question", "concern"], value=None, info="Manually choose which part of Data you want to talk to. This is only effective if 'Automatic Path Selection' is off."),
        gr.Dropdown(label="QA LLM: Architecture", choices=[f"{qa_custom_name}", "GPT-3.5"], value=f"{qa_custom_name}"),
        gr.Slider(label="QA LLM: Temperature", minimum=0, maximum=1, value=0.2),
        gr.Slider(label="Concern LLM: Temperature", minimum=0, maximum=1, value=0.4)
    ],
    description="Chat with Data, the friendly chatbot of the BSc Data Science at FHNW. Built by Tobias Buess, Alexander Shanmugam and Yvo Keller within cnlp1/HS23.",
    examples=[["Was lerne ich im Modul Grundlagen der linearen Algebra?"], ["Wer ist Fachexperte im Modul NPR?"], ["Hey, ich hatte heute einen ganz schlechten Tag..."]]
).queue()

# Combine all interfaces in a tabbed interface
demo = gr.TabbedInterface([chat_int, classification_int, documentquery_int], ["Chat with Data", "Classification", "Retrieval"], title="One interface to rule Data 🤖")

# Launch the interface
if __name__ == "__main__":
    demo.launch(share=True)
