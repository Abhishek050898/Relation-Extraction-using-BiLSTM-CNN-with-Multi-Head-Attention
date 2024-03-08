# Relation Extraction Classifier

This project provides two approaches for relation extraction classification using deep learning and machine learning techniques.

## Usage in Google Colab

1. Open the provided python notebooks in Google Colab.
    - `Final_Neural_Networks_Model.pynb`
    - `Final_StackingClassifier_Model.ipynb`

2. In the Colab notebook, execute the cells one by one to install the required libraries and load the necessary functions.

3. Upload the `glove.6B.100d.txt` file for the neural-based approach and upload it in the colab runtime in the files section. You can download the file from Google Drive link provided below:
    https://drive.google.com/drive/folders/1elk41Qhy-ppaZpbeabfiXhoEMO9BLvg5?usp=sharing 

4. **Before User Testing:**
    - Upload the `classifier.py` file to the Colab environment. You can upload it directly from your local machine or use a cloud storage link. File has already been provided in the project folder.
 
5. For Approach 1 (Deep Learning Model):

    - Ensure you have GPU acceleration (T4 GPU) enabled.
    - Execute the cells for the deep learning model training/testing and user testing.

6. For Approach 2 (Support Vector Machine Model):

    - Execute the cells for SVM model training/testing and user testing.
    - If you want to perform user testing with a faster compilation time, use the `Final_StackingClassifier_UserTesting.ipynb` notebook.

7. For Both Approaches:

    - After training, users can use the provided `classifier.py` module for classifying new sentences and should include the import statements.

    ```python
    # For Approach 1
    from classifier import classify_relation
    # For Approach 2
    from classifier import classify_relation_svm
    ```
