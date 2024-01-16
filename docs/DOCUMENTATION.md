# Building Data - A Chatbot for the Data Science Study Programme at FHNW

## Introduction

This documentation will go over how we built Data, the Chatbot for the Data Science Study Programme at FHNW. It will cover the following topics:

- [Concept](#concept)
- [Methodology](#methodology)
  - [Step 1: Classification](#step-1-classification)
  - [Step 2: Question Answering](#step-2-question-answering)
  - [Step 3: Concern Path](#step-3-concern-path)
  - [Step 4: Assemblying the Chatbot](#step-4-assemblying-the-chatbot)
- [Future Work](#future-work)
- [Conclusion](#conclusion)
- [References](#references)

We refer to the specific implementations of the individual components for more details, which are liked here. Each repository has its own README, which explains the implementation in detail, and further references relevant notebooks and python scripts if you want to dive deeper into the code.

## Concept

Our project started based on the definition of the Chatbot Concept and the Chatbot Architecture. The concept outlined planned features and the architecture, and is described in this section. It was written before we started work on the project, and thus might be slightly different from the final implementation.

### Goal

The goal of this challenge is to develop a chatbot named Data. It is intended to assist students of the Data Science program by answering their questions about the content of the Spaces related to the modules. The chatbot should have a personality and follow predefined ethical guidelines. Data will be able to access the content of the module Spaces and answer standard queries using a knowledge base.

The bot should also recognize users' problems and respond in a morally appropriate way, such as with encouraging words or by referring them to a contact person. Additionally, it should contribute to motivating the students.

Our focus in the challenge is to build a version of the bot that functions well in German (the language of most content in the knowledge base). While it is not excluded that the bot could also function in English, we will not explicitly focus on this.

### Bot Capabilities

The bot should be able to provide the following information upon request:

- Explain its capabilities
- Details about the structure of the degree program (concept, handbook, curriculum, regulations)
- Details about the module
  - Subject experts
  - Language
  - ECTS
  - Type
  - Level
- Contents of the "Portrait" tab of the modules
- *optional*:
  - Provide knowledge from PDFs in the module learning materials
  - Suggest learning materials
  - Tab "Tasks"

In its behavior, the bot should consider the following personality aspects:

- Use informal language with users (as is customary in the SG Data Science)
- Speak ethically and morally correctly (fair, appreciative)
- Have a motivating, humorous, and empathic personality
- Be able to conduct a dialogue with the user and know about its own history
- Adequately respond to the user's problems when recognized (e.g., stress in studies, dissatisfaction, depressive phases) and provide contact information for points of contact

#### What the Bot Should NOT Do

- Explicitly support multilingualism
- Access the content of learning materials (e.g., PDFs or external links)
- We will not invest direct effort in preventing prompt injection and similar issues within the scope of the challenge

### Knowledge Base

The bot's knowledge base should be based on multiple sources. These include the necessary information for providing its capabilities, from both the Spaces DB dump and the PDFs about the degree program (concept, handbook, curriculum, and regulations).

The bot should prioritize the context provided when answering questions. If a question is asked that the bot cannot answer based on the existing context from the knowledge base, it declares this and either uses the knowledge in the LLM to answer the request (e.g., "What is linear regression?") or declines to answer the request.

The content can be in both German and English, which we will consider in development.

### Design

We will create an avatar for "Data" to be displayed in the chat interface.

- Avatar image
- *optional*: Synthetic voice

### Architecture & Tech Stack

The planned architecture can be seen in the following sketch. The individual components are briefly explained below, with changes to the architecture or the technologies used not being excluded during the challenge.

![Architecture Sketch](../assets/architecture_sketch.jpeg)

#### Chat Interface

The bot is available to users in a simple web chat interface. We will build the logic for processing requests by the bot using Python, possibly using LangChain to communicate with the various LLMs.

Tech Stack:

- Chat interface with Streamlit (alternatively Gradio from HuggungFace)
- *likely*: LangChain (For embedding and communication with LLMs)

#### Prompt Classification (also: npr MC1)

The bot initially distinguishes between 3 types of requests:

- question
- concern
- not a question or concern

Possible approach:

- Evaluate in an experiment how many training data are needed until the model achieves good performance in classification
- Fine-tuning a BERT model for user prompt classification

Tech Stack:

- HuggingFace Transformers library
- LLM: [BERT Base Multilingual](https://huggingface.co/bert-base-multilingual-cased) or similar

#### LLM for "concern" (also: npr MC2 and eim MC)

The bot addresses the user's concerns when recognized. It should respond empathetically and motivatingly, and provide the user with contact information for points of contact if needed.

Possible approach:

- Fine-tuning a LLAMA2 model (or similar) on the ethical guidelines for advising and supporting the user regarding their concerns.

Tech Stack:

- HuggingFace Transformers library
- LLM: [LLAMA2-13B-Chat](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) or similar

#### LLM for "question" (also: npr MC2)

The bot answers the user's question if it understands it and the answer is available in the knowledge base. Otherwise, it declines to answer the question, or informs the user that there is nothing available in the knowledge base, and tries to help further with internal LLM knowledge.

Possible approach:

- Chunking and embedding the context in the knowledge base.
- Fine-tuning/Instruction-tuning of a LLAMA2 model on answering questions from given context.

Tech Stack:

- Embeddings (BERT/Open AI)
- Vector Storage [PGVector](https://github.com/pgvector/pgvector)
- HuggingFace Transformers library
- LLM: [LLAMA2-13B-Chat](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) or similar

### Target Audience

The target audience for the bot are the students of the Data Science program. We have defined two personas to represent the target group.

#### Persona 1: Anna, the Diligent Student

##### Demographic Data

- Age: 23
- Gender: Female
- Professional Background: Apprenticeship as a computer scientist

##### Personality

- Ambitious and focused
- Detail-oriented
- Loves to plan early

##### Needs and Goals

- Wants to have the best overview of her modules
- Always looking for additional resources for better learning success
- Wants to stay informed about changes in the curriculum

##### Usage Scenarios

- Asks the bot about the assessments in specific modules
- Wants to know which subject experts are responsible for a module
- Plans the next semester and asks the bot about modules in the curriculum
- Is praised by the bot for her good work and feels motivated

#### Persona 2: Markus, the Working Student

##### Demographic Data

- Age: 29
- Gender: Male
- Professional Background: Works part-time in accounting

##### Personality

- Pragmatic and goal-oriented
- Values work-life-study balance
- Somewhat prone to stress due to many commitments

##### Needs and Goals

- Looks for an efficient way to consult study information
- Wants to spend as little time as possible searching for basic information
- Seeks a quick way to get his questions answered to focus on his work and studies

##### Usage Scenarios

- Wants to quickly know how many ECTS a module has
- Wants to know what to expect in a specific module
- Uses the bot's motivating and empathetic functions to reduce stress

These personas can serve as a foundation for the development of the chatbot "Data." They represent the needs and goals of the target group and can help to optimally align the functionality and behavior of the bot.

### Privacy

We use our own fine-tuned models, so no sensitive data leaves our system. The chatbot is only accessible to students of the Data Science program, and therefore only information that students already have access to is available.

### Evaluation

A chatbot that gives incorrect answers or does not address the user's questions is not helpful. Therefore, it should be evaluated in various ways.

#### Prompt Classification

Quantitatively:

- F1-Score
- Accuracy

Qualitatively:

- Test individual examples on several models for comparison

#### LLM for "concern"

Quantitatively:

- Not planned

Qualitatively:

- Individual examples to test whether the bot follows the ethical guidelines

#### LLM for "question"

Quantitatively:

- Retrieval (Are the relevant chunks identified)
- BLEU Score (How good is the answer)

Qualitatively:

- Individual examples to test the usefulness of the answers

These evaluation methods will help ensure that the chatbot "Data" is effective, reliable, and adheres to the intended ethical and operational standards, catering effectively to the specific needs and scenarios of the target student audience in the Data Science program.

## Methodology

### Step 1: Classification

Find the implementation and all details on the Classification here: [Text-Classification](https://github.com/NLP-Challenges/Text-Classification)

#### Objective

The primary objective was to enable our chatbot, named Data, to classify user prompts into one of three categories: questions, harm, or concerns. This classification is vital for the chatbot to respond appropriately to user interactions. For details on why this is vital to our architecture, see [Concept](#concept).

#### Data Selection and Preparation

We selected three datasets for creating a comprehensive training dataset for our classifier. These included:

- **[GermanQUAD for Questions](https://huggingface.co/datasets/deepset/germanquad)**: This dataset provided a robust set of question-answer pairs in German, ideal for training our model to recognize user queries.
- **[GermEval for Harm](https://huggingface.co/datasets/philschmid/germeval18)**: This dataset was chosen to help the classifier identify harmful or inappropriate content.
- **[Stress-Annotated Dataset (SAD) for Concerns](https://github.com/PervasiveWellbeingTech/Stress-Annotated-Dataset-SAD)**: To enable the chatbot to recognize and appropriately address user concerns, especially those of a sensitive or personal nature.

A qualitative assessment using a BERT classifier suggested the feasibility of our approach. We then proceeded with a train-test split to prepare the datasets for model training.

#### Model Selection and Training

We chose three model architectures:

1. **LinearSVC with TF-IDF**: Selected as a baseline model to compare against more advanced deep learning approaches. LinearSVC was chosen based on its superior performance among seven tested machine learning classifiers.
2. **LSTM-CNN**: This model was considered due to the LSTM's ability to handle sequential data, potentially learning relationships between word semantics. The CNN was employed for initial feature extraction before feeding data into the LSTM. Pre-trained embeddings (bert-base-german-cased) were utilized, anticipating a reduction in overfitting risk and less need for extensive training data.
3. **BERT Fine-Tuning as Classifier (bert-base-multilingual-cased)**: This model was selected for its support of both English and German sentence structures, coupled with the expectation that its pre-training would require less training data.

#### Evaluation and Improvement

All models performed well on the test dataset, with BERT achieving the highest accuracy (98%). However, when tested on a benchmark dataset comprised of examples created by our team, all models showed a significant drop in performance (Accuracy: SVC 0.60, LSTM-CNN 0.62, BERT 0.67). Based on these results, we decided to continue with BERT as our primary classifier model.

To enhance model performance, we created a synthetic dataset using GPT-4 with tailored prompting. This dataset, comprising queries from all three classification categories, led to a marked improvement in BERT's performance on our benchmark dataset.

We also recognized some misclassifications in basic queries, such as "Hallo was kannst du" being incorrectly categorized as harm. To address this, we extended our synthetic dataset with a "human feedback" component, that can be continually updated based on flagged results from production use, to further refine accuracy.

### Step 2: Question Answering

TODO: add retrieval information somewhere!

Find the implementation and all details on the Question Answering here: [Implementation](https://github.com/NLP-Challenges/llm-qa-path)

#### Objective

The aim was to develop a large language model (LLM) capable of providing context-informed, abstractive answers to user queries.

#### Data Selection and Preparation

We selected the GermanQUAD dataset, which includes questions, contexts, and corresponding answers in German. This dataset was used to train the LLM in extracting answers from the provided context.

#### Initial Approach and Realization

Initially, we aimed for the model to learn answer extraction from context by fine-tuning it with examples from the GermanQUAD dataset. However, we observed that the answers generated were extractive, mirroring the dataset's nature. To achieve our goal of abstractive response generation, we modified our approach by using GPT-3.5 to create abstractive answers from questions, contexts, and the original answers.

#### Context Swapping and Source Referencing

Context swapping was incorporated into our methodology, as we recognized the importance of the model learning to not provide answers when relevant context was unavailable. We also trained the model to refer to sources at the end of its abstractive responses by splitting the available context into multiple blocks and adding synthetically generated sources to these blocks.

#### Model Training and Selection

We selected and trained three different LLMs:

- Baseline: meta-llama/Llama-2-13b-hf
- German-optimized Llama-2-13b: flozi00/Llama-2-13b-german-assistant-v7
- German-optimized Mistral-7b: VAGOsolutions/SauerkrautLM-7b-v1-mistral

We opted for base models over chat variants for the learning experience, despite potentially lower overall performance.

#### DVC Stages

Using a DVC pipeline allowed us to automate the retrieval and fine-tuning stages of our methodology. This pipeline was designed to be run on a GPU-enabled machine.

##### Retrieval Stages

The retrieval stages aimed to create chunks from relevant documents (Data Science study program documents, course materials, etc.) and save them in a ChromaDB vector store. This store was later queried for relevant documents in response to user queries.

##### Fine-tuning Stages

The fine-tuning stages were designed to enhance the language model's question-answering abilities using our modified GermanQUAD dataset. The resulting fine-tuned models can then be used to generate abstractive answers to user queries.

#### Dataset, Models, and Experiments

##### Fine Tuning Dataset

The dataset, derived from GermanQUAD, was modified to include both answerable and unanswerable questions, with abstractive answers generated using GPT-3.5. The dataset was split into training, validation, and test sets.

##### Fine Tuning Models

We conducted experiments with three different language models:

- Baseline: meta-llama/Llama-2-13b-hf
- German-optimized Llama-2-13b: flozi00/Llama-2-13b-german-assistant-v7
- German-optimized Mistral-7b: VAGOsolutions/SauerkrautLM-7b-v1-mistral

#### Evaluation

We performed quantitative and qualitative evaluations to ascertain the best-performing model in terms of generating context-informed, abstractive answers.

### Step 3: Concern Path

For the concern path, we built a prompt upon ethics theory. Find the comprehensive report including evaluation here: [Concern Path](concern_path.pdf)

The functionality is built directly into the Chatbot Dashboard, see [dashboard.py](../src/dashboard.py).

### Step 4: Assemblying the Chatbot

After the individual components were developed, we assembled the chatbot using the Hugging Face Gradio framework. It allowed us to create a simple, user-friendly interface for the chatbot, which can be accessed via web browser. It has a Chat Tab, a Retrieval Tab and a Classification Tab. This way, each component can be tested individually, which is immensely helpful for debugging and improving the chatbot.

The Chat Tab is the main interface for the user. It first uses the classifier to determine the type of the user input. If the input is a question, it is passed to the [Question Answering](#step-2-question-answering) module, which retrieves relevant contex and generates the answer. If the input is a concern, the [Concern Path](#step-3-concern-path) is triggered, which generates an appropriate messasge to continue the conversation. If the input is harm, the chatbot responds with a predefined message that denies the user's request.

The implementation can be found in [dashboard.py](../src/dashboard.py).

For setup instructions, see [README.md](../README.md).

## Future Work

- When we would move towards deploying "Data" in a production environment, our focus would be on continual retraining and improvement of the classifier. This will be achieved by leveraging new human feedback to refine the model's accuracy and adaptability to real-world interactions.
- Our next steps involve continuous improvement of the question-answering module based on real-world interactions and user feedback.
- Another important improvement would be building a history into the Question Answering module, so that the chatbot can remember previous questions and answers and use them to generate more context-informed answers.
- Instead of opting for a solely classifier, we would include the help of an LLM in that task, and instead of having either the Question Answering model or the Concern model responding, we would opt for one model with both functionalities built into it. This could for example be achieved by the combined model using a response generated by a separate model focused on QA, and using that in it's response, or fine-tuning a model to have both capabilities.

## Conclusion

TODO

## References

TODO
