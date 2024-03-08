# Relation Extraction Classifier

This project provides two approaches for relation extraction classification using neural networks and machine learning techniques.

## Usage in Google Colab

1. Open the provided python notebooks in Google Colab.
    - `Final_Neural_Networks_Model.pynb`
    - `Final_StackingClassifier_Model.ipynb`

2. In the Colab notebook, execute the cells one by one to install the required libraries and load the necessary functions.

3. Upload the `glove.6B.100d.txt` file for the neural-based approach. You can upload it directly from your local machine or use a cloud storage link.

4. **Before User Testing:**
    - Upload the `classifier.py` file to the Colab environment. You can upload it directly from your local machine or use a cloud storage link.
 
5. For Approach 1 (Deep Learning Model):

    - Ensure you have GPU acceleration (T4 GPU) enabled for better performance.
    - Execute the cells for the deep learning model training/testing and user testing.

6. For Approach 2 (Support Vector Machine Model):

    - Execute the cells for SVM model training/testing and user testing.

7. For Both Approaches:

    - After training, users can use the provided `classifier.py` module for classifying new sentences.

    ```python
    # For Approach 1
    from classifier import classify_relation
    predicted_relation = classify_relation(user_input, model, tokenizer, max_sent_len, word_index, label_encoder)

    # For Approach 2
    from classifier import classify_relation_svm
    predicted_relation = classify_relation_svm(user_input_sentence, stacking_model, vectorizer)
    ```
