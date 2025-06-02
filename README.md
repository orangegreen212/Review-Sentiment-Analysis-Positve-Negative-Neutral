# Sentiment Analysis of User Comments

This project performs sentiment analysis on user comments to classify them as Positive, Negative, or Neutral. The primary goal is to build and evaluate a machine learning model capable of understanding the sentiment expressed in text data. Dataset was given from Kaggle, also it uploaded in /data folder

## Project Overview

The project follows these main steps:

1.  **Data Loading & Initial Exploration:**
    *   Loads a dataset of user comments (presumably from Kaggle or a similar source).
    *   Initial inspection of data structure and content.
2.  **Text Preprocessing:**
    *   Converts text to lowercase.
    *   Removes punctuation, special characters, and numerical digits.
    *   Tokenizes text into individual words.
    *   Removes common English stop words.
    *   Applies lemmatization to reduce words to their base form.
    *   Handles and removes any NaN values or empty text strings.
3.  **Exploratory Data Analysis (EDA):**
    *   Analyzes class distribution (Positive, Negative, Neutral).
    *   Visualizes data characteristics, potentially including text length, word counts, and word clouds for different sentiment categories.
4.  **Sentiment Scoring (Rule-Based):**
    *   Utilizes VADER (Valence Aware Dictionary and sEntiment Reasoner) to get initial sentiment scores for example texts.
5.  **Machine Learning Model Training & Evaluation:**
    *   **Feature Engineering:** Converts preprocessed text data into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.
    *   **Data Splitting:** Splits the dataset into training and testing sets, using stratification to maintain class proportions.
    *   **Handling Class Imbalance:** Applies RandomUnderSampler to the training data to create a balanced dataset for model training.
    *   **Model Training:** Trains a Multinomial Naive Bayes classifier on the resampled (balanced) training data.
    *   **Prediction:** Makes sentiment predictions on the original (imbalanced) test set.
    *   **Evaluation:**
        *   Generates a Classification Report (precision, recall, F1-score, support).
        *   Displays a Confusion Matrix.
        *   Calculates and plots ROC AUC curves (One-vs-Rest, Micro-average, Macro-average) to evaluate model performance.
        *   Calculates PR AUC (Precision-Recall Area Under Curve) scores for each class (One-vs-Rest).

## Technologies Used

*   Python 3.x
*   Pandas: For data manipulation and analysis.
*   NLTK (Natural Language Toolkit): For text preprocessing tasks (tokenization, stop words, lemmatization) and VADER sentiment analysis.
*   Scikit-learn (sklearn): For machine learning tasks (TF-IDF, train-test split, Naive Bayes model, evaluation metrics, label binarization).
*   Imbalanced-learn (imblearn): For handling imbalanced datasets (RandomUnderSampler).
*   Matplotlib: For plotting graphs (e.g., ROC curves).
*   WordCloud: For generating word cloud visualizations (if included).
*   Collections (Counter): For inspecting class distributions.


## Results

The model's performance is evaluated using metrics such as accuracy, precision, recall, F1-score, and ROC AUC. The classification report and confusion matrix provide detailed insights into its predictive capabilities for each sentiment class. After applying undersampling, the model shows improved recall for minority classes (Negative and Neutral), though with some trade-offs in precision or overall accuracy compared to a model trained on imbalanced data.

*(Optional: Briefly mention key ROC AUC scores or F1-scores if you want to highlight specific results, e.g., "Achieved a macro-averaged ROC AUC of X.XX after undersampling.")*

## Future Work (Optional)

*   Experiment with different resampling techniques (e.g., SMOTE for oversampling).
*   Try more advanced models
*   Explore different feature engineering techniques 
*   Fine-tune hyperparameters for the chosen model.
