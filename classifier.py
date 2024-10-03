import re
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_and_tokenize(sentence):
    """
    Function to clean and tokenize a sentence.

    Args:
    - sentence (str): Sentence to be tokenized.

    Returns:
    - tokenized_sentence (list): Tokenized sentence.
    """
    # Remove leading/trailing quotes
    sentence = sentence[0:-1]

    # Convert Text to Lowercase
    sentence = sentence.lower()

    # Replace <e1> and </e1> tags with tokens
    sentence = sentence.replace("<e1>", " E1_START ").replace("</e1>", " E1_END ")
    # Replace <e2> and </e2> tags with tokens
    sentence = sentence.replace("<e2>", " E2_START ").replace("</e2>", " E2_END ")

    # Remove Noise
    sentence = re.sub(r'[^a-zA-Z0-9\s]', '', sentence)

    # Tokenize the sentence
    tokens = word_tokenize(sentence)

    return tokens

def get_mask_entities(x, word_index):
    ''' 1 for entity words, 0 otherwise '''
    ret = np.zeros_like(x)
    e1 = [0, 0]
    e2 = [0, 0]
    for j in range(x.shape[1]):
        if x[0][j] == word_index.get("e1start", 0):
            e1[0] = j
        elif x[0][j] == word_index.get("e1end", 0):
            e1[1] = j
        elif x[0][j] == word_index.get("e2start", 0):
            e2[0] = j
        elif x[0][j] == word_index.get("e2end", 0):
            e2[1] = j
            break
    for j in range(e1[0]+1, e1[1]):
        ret[0][j] = 1
    for j in range(e2[0]+1, e2[1]):
        ret[0][j] = 1

    return ret

def classify_relation(input_sentence, model, tokenizer, max_sent_len, word_index, label_encoder):
    cleaned_tokens = clean_and_tokenize(input_sentence)

    input_tokens_idx = tokenizer.texts_to_sequences([cleaned_tokens])
    input_tokens_idx = pad_sequences(input_tokens_idx, maxlen=max_sent_len, padding="post")

    input_mask_entities = get_mask_entities(input_tokens_idx, word_index)

    predictions = model.predict([input_tokens_idx, input_mask_entities])
    predicted_class = np.argmax(predictions)
    predicted_relation = label_encoder.classes_[predicted_class]

    return predicted_relation


# SVM

def preprocess_sentence(sentence):
    # Extracting entities and their indices
    def extract_entities(sentence):
        e1_match = re.search(r'<e1>(.*?)<\/e1>', sentence)
        e2_match = re.search(r'<e2>(.*?)<\/e2>', sentence)
        
        e1 = e1_match.group(1) if e1_match else ''
        e2 = e2_match.group(1) if e2_match else ''
        
        e1_index = e1_match.start() if e1_match else -1
        e2_index = e2_match.start() if e2_match else -1
        
        return e1, e2, e1_index, e2_index

    e1, e2, e1_index, e2_index = extract_entities(sentence)

    # Cleaning the sentence
    def clean_data(sentence):
        sentence = re.sub(r'<\/?e[12]>', '', sentence)
        sentence = sentence.lower()
        sentence = re.sub(r'[^a-zA-Z0-9\s]', '', sentence)
        return sentence

    cleaned_sentence = clean_data(sentence)

    # Tokenize, POS tag, and lemmatize
    def tokenize_text(sentence):
        return nltk.word_tokenize(sentence)
    
    def pos_tagging(tokens):
        return nltk.pos_tag(tokens)
    
    def lemmatization(tokens):
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]

    tokens = tokenize_text(cleaned_sentence)
    pos_tags = pos_tagging(tokens)
    lemmatized_tokens = lemmatization(tokens)

    return e1, e2, e1_index, e2_index, cleaned_sentence, tokens, pos_tags, lemmatized_tokens


def tfidf_vectorize_sentence(sentence, vectorizer):
    # Clean the input sentence (you can reuse your clean_data function)

    # Transform the cleaned sentence using the provided vectorizer
    vectorized_sentence = vectorizer.transform([sentence])

    return vectorized_sentence


def classify_relation_svm(input_sentence,vectorizer, stacking_model):
    
    e1, e2, e1_index, e2_index, cleaned_sentence, tokens, pos_tags, lemmatized_tokens = preprocess_sentence(input_sentence)
    
    # Clean and vectorize the input sentence
    vectorized_input = tfidf_vectorize_sentence(cleaned_sentence, vectorizer)

    # Use the pre-trained StackingClassifier for prediction
    predicted_class = stacking_model.predict(vectorized_input)[0]

    return predicted_class



