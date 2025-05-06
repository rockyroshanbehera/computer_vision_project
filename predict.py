# 📌 Import libraries
import PyPDF2
import pandas as pd
import numpy as np
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import os

# 📌 Download NLTK data
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# 📌 Function to extract text from PDFs
def extract_text_from_pdf(pdf_file):
    with open(pdf_file, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# 📌 Extract text from multiple PDFs and assign labels based on filename
def process_pdfs_in_directory(directory_path):
    all_questions = []
    all_labels = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):  # Process only PDF files
            pdf_path = os.path.join(directory_path, filename)
            text = extract_text_from_pdf(pdf_path)

            # Split into questions (assuming each question ends with '?')
            questions = [q.strip() for q in text.split('?') if len(q.strip()) > 10]

            # 📌 Assign label based on filename — customize this mapping as per your PDF names
            if "Unit1" in filename:
                label = "Unit 1"
            elif "Unit2" in filename:
                label = "Unit 2"
            elif "2022" in filename:
                label = "2022 Paper"
            elif "2023" in filename:
                label = "2023 Paper"
            else:
                label = "Other"  # Use 'Other' instead of 'Unknown'

            # Add questions and labels
            all_questions.extend(questions)
            all_labels.extend([label] * len(questions))

    return all_questions, all_labels

# 📌 Function to preprocess text (without NLTK tokenizer, using regex)
def preprocess_text(text):
    text = text.lower()
    tokens = re.findall(r'\b[a-zA-Z]{2,}\b', text)  # extract words of 2+ letters
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# 📌 Set the directory where your PDFs are located
directory_path = 'path_to_your_pdfs'  # Update with your local path

# 📌 Process the PDFs and get the questions and labels
all_questions, all_labels = process_pdfs_in_directory(directory_path)

# 📌 Convert to DataFrame
data = pd.DataFrame({
    'Question': all_questions,
    'Label': all_labels
})

# 📌 Clean the questions
data['Cleaned_Question'] = data['Question'].apply(preprocess_text)

# 📌 Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    data['Cleaned_Question'], data['Label'], test_size=0.2, random_state=42
)

# 📌 Vectorize text with TF-IDF (convert words into numerical vectors)
vectorizer = TfidfVectorizer(max_features=2000)
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# 📌 Train Logistic Regression Classifier
model = LogisticRegression()
model.fit(X_train_vectors, y_train)

# 📌 Predict and evaluate
y_pred = model.predict(X_test_vectors)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 📌 Predict for new user input (example prediction)
def predict_question_label(question_text):
    cleaned_text = preprocess_text(question_text)
    vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(vector)
    return prediction[0]

# 📌 Example prediction
new_question = "Explain supervised machine learning with example."
predicted_label = predict_question_label(new_question)
print(f"\nPredicted Label for your question: {predicted_label}")
