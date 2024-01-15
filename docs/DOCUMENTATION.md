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

Our project started based on the definition of the Chatbot Concept and the Chatbot Architecture. The concept, outlining planned features and the architecture, can be read [here](CONCEPT.md).

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
