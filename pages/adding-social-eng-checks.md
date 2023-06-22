---
title: "Classification using Large Language Models for Social Engineering Detection"
author: "Karpenko Roman"
editor: GPT-4
---

# Classification using Large Language Models for Social Engineering Detection
Author: Karpenko Roman

## Introduction

**Proposed Research Question:** How can we leverage data science techniques to identify and protect against social engineering threats posed by chatbots?

**Rationale:** In the era of digital communication, chatbots have become a pervasive part of online interactions. They have been used for a variety of purposes, ranging from customer service to user engagement. However, alongside their beneficial uses, chatbots also pose a significant risk in terms of social engineering attacks. Social engineering is a method of tricking individuals into disclosing confidential information, and chatbots provide an effective and scalable way for malicious actors to carry out such attacks. Therefore, understanding how to identify and mitigate these threats is crucial for personal and collective digital security.

## Methodology

In classification tasks using large language models like GPT-3, we typically use a Softmax function for the final layer of the model to obtain probabilities for each class. 

Let's assume that the output of the model before the final layer for a given input is represented by a vector `z` with dimensions `(1 x n)`, where `n` is the number of classes. The Softmax function can then be defined as follows:

```
p_i = e^(z_i) / Σ (from j=1 to n) e^(z_j)
```

where `p_i` is the predicted probability for class `i`, `z_i` is the `i`th element of `z`, and `n` is the total number of classes. `e` is the base of the natural logarithm.

The class with the highest predicted probability is then chosen as the model's prediction:

```
y_hat = argmax (over i) p_i
```

where `y_hat` is the predicted class.

It should be noted that while GPT-3 can be fine-tuned for specific tasks, the original training of GPT-3 (or any of the GPT series) does not involve specific task-oriented labels. Rather, it's trained to predict the next token in a sequence of tokens, based on the previous tokens. The tokens can be words, characters, or subwords, depending on the specific implementation. The model learns to generate meaningful and contextually relevant text by being trained on a large corpus of text data and by learning to minimize the difference between its predictions and the actual next tokens in the training data. This is typically achieved by using a Cross-Entropy loss function during training.

## Discussion

**Decisions to be Made:** This research will require a variety of decisions to be made. First, we need to decide on the dataset to be used - this could either be an existing dataset of chatbot interactions or a dataset we collect ourselves. Next, we need to decide on the techniques to be used to analyze the data. This could involve machine learning to identify patterns typical of chatbots, natural language processing to understand the linguistic nuances of chatbot interactions, or a combination of both. We also need to consider ethical implications, such as privacy concerns when collecting and analyzing chat data.

**Stakes:** The stakes for this research are high. As chatbots become more sophisticated and indistinguishable from human interaction, the potential for their misuse in social engineering attacks also rises. By identifying the characteristics of these chatbots, we can help inform individuals and organizations about the risks and provide strategies for protection. Additionally, our findings could be used to improve the design and regulation of chatbots to prevent their misuse. Ultimately, this research could contribute to safer online spaces

## Data Collection and Preparation

When it comes to collecting data for this research, we have two main options:

1. **Use Existing Datasets:** There are already datasets available that consist of chatbot and human dialogues. For example, the Cornell Movie Dialogs Corpus or the Persona-Chat dataset. These datasets can serve as a good starting point, but they might not be perfect for our specific use case. The dialogs in these datasets are not centered around social engineering scenarios and therefore might not be representative of the data we aim to classify.

2. **Create a New Dataset:** This involves interacting with different chatbots and creating new logs. This can be done either manually, which would be very time consuming, or semi-automatically by designing scenarios and scripts to simulate interactions. 

Once we have the data, the next step would be to prepare it for analysis. This involves cleaning the data to remove any irrelevant information, normalizing the data to ensure consistency, and transforming the data into a format that can be used for machine learning. If our data includes labels, we will also need to encode these labels into a format that can be used for supervised learning.

## Modeling and Evaluation

After the data is prepared, we can start building our model. Since we are dealing with text data, we will likely use natural language processing techniques. For instance, we can start with a simple bag-of-words model and gradually increase the complexity by using techniques like TF-IDF, word embeddings, or even transformers.

The model needs to be trained using a portion of our dataset. The training process involves feeding the input data to the model and adjusting the model's parameters to minimize the difference between the model's predictions and the actual labels.

To evaluate the performance of our model, we will split our data into a training set and a test set. The training set will be used to train our model, while the test set will be used to evaluate its performance. We will use measures like precision, recall, and F1-score to quantify the performance of our model.

If the model performs well on the test set, we can conclude that it is likely to perform well on unseen data. If not, we might need to revisit our data, our preprocessing steps, or our model architecture.

## Implementation and Deployment

Once we are satisfied with our model's performance, the final step is to deploy it in a real-world setting. This could involve integrating the model into an existing system or building a new system around it. 

The model will need to be retrained periodically to ensure it stays relevant as the nature of chatbots and social engineering attacks evolves. It will also need to be monitored to ensure it is working as expected and to identify any potential issues.

In conclusion, the application of large language models for the detection of social engineering in chatbots is a promising avenue of research. By effectively identifying malicious chatbots, we can help protect individuals and organizations from social engineering attacks and contribute to the ongoing efforts to ensure safe and secure digital spaces.

## Building application 

User Input -> [ChatGPT (Prompt Generation) -> Malicious/Non-malicious Classifier] -> System Response

User Input: The user interacts with the system by inputting text.

ChatGPT (Prompt Generation): The system (ChatGPT or similar model) generates a prompt based on the user's input.

Malicious/Non-malicious Classifier: This part of the system would use a trained model (GPT, DistilBERT, etc.) to classify the generated prompts as either 'malicious' (social engineering attempt) or 'non-malicious' (safe).

System Response: The system responds to the user based on the classification. If the prompt is classified as non-malicious, the system continues the conversation normally. If the prompt is classified as malicious, the system can either end the conversation or warn the user about potential social engineering.

To train the malicious/non-malicious classifier, you would need a labeled dataset containing both malicious and non-malicious prompts. Here are a few examples:

Malicious Prompts:

"I'm experiencing some issues with my account. Can you provide me with your login information so I can check what's wrong?"

"Your account has been compromised. Please share your password so we can secure it."

"To fix this issue, we need access to your email. Could you share it?"

Non-malicious Prompts:

"Let's try some troubleshooting steps. Have you tried restarting your device?"

"Can you tell me more about the problem you're experiencing?"

"It seems like this issue might require technical support. I suggest contacting our help desk."



## Future Work

While the research and methodology discussed provide an initial framework for detecting social engineering attempts in chatbot interactions, there are several opportunities for future work.

One of the key areas for further exploration is the application of different Natural Language Processing (NLP) classification algorithms to classify malicious activity. The research outlined here utilized large language models like GPT-3, but other models like BERT, DistilBERT, or LSTM could potentially offer different perspectives or improved performance. Comparative studies involving these different models could help to determine which models are most effective for this particular classification task.

Furthermore, another area for future work is the expansion of the dataset used for model training. In this study, we utilized a binary classification ('malicious' vs 'non-malicious') to label our data. However, different types of social engineering attacks (phishing, baiting, pretexting, etc.) could be explored in more granular detail if the dataset includes these specific labels. This could lead to a multi-class classification model that not only identifies a malicious intent but also categorizes the type of social engineering attack being attempted.

Lastly, considering the rapid advancements in AI and NLP, it is imperative to update and retrain the model periodically to ensure that it stays relevant and accurate. Future work could include the development of automated retraining pipelines, anomaly detection systems to identify new types of attacks, and continuous monitoring and evaluation systems to assess the model's performance over time.

### Loading and Testing Classification Models

Here is a simplified code snippet for loading and testing different classification models using the transformers library in Python:

```python
from transformers import pipeline

# Define a function to load a model and tokenizer and create a pipeline
def load_model(model_name):
    return pipeline("text-classification", model=model_name)

# Define a function to test a model on a single example
def test_model(classification_pipeline, example):
    result = classification_pipeline(example)[0]
    return result["label"], result["score"]

# Load models
gpt3_pipeline = load_model("openai/gpt-3")
bert_pipeline = load_model("bert-base-uncased")
distilbert_pipeline = load_model("distilbert-base-uncased")

# Test models on an example
example = "I'm experiencing some issues with my account. Can you provide me with your login information so I can check what's wrong?"
for pipeline in [gpt3_pipeline, bert_pipeline, distilbert_pipeline]:
    label, score = test_model(pipeline, example)
    print(f"Model: {pipeline.model.name_or_path}, Label: {label}, Score: {score}")
```

That's one of the ways of how we can test the models performances and get the best model out there to solve our task. 

## Conclusion

In conclusion, this research provides an initial exploration into using large language models for the detection of social engineering attacks in chatbot interactions. By building a classifier that identifies malicious prompts, we can help protect individuals and organizations from these types of attacks and contribute to safer online communication.

However, as with all machine learning models, there are limitations and areas for future work. The continual evolution of language use, social engineering tactics, and chatbot technology necessitates ongoing research and model updating. Furthermore, expanding the scope of our classifier to not only detect malicious activity but also to categorize the type of attack being attempted could provide more detailed and actionable insights.

The fusion of NLP, AI, and cybersecurity provides an exciting and crucial area for research. As we continue to refine our models and methodologies, we hope to contribute significantly to the detection and prevention of social engineering attacks, fostering a safer digital environment for all users.


## References 

In our ongoing efforts to innovate and disrupt, we frequently draw upon thought-leadership and industry insights. One such resource is an insightful guide on deceptive design types by [Deceptive Design](https://www.deceptive.design/types). This guide gives us a comprehensive look at the various deceptive design types prevalent in the industry today.

According to Deceptive Design:

> "Design can be used to deceive users into doing things they might not want to do, but what benefits the company in question. These designs are often used to trick users into giving away their data, clicking on ads or buying services."

By being aware of such practices, we ensure our design philosophy remains user-centric, transparent, and free from deceptive tactics.

