import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

# Read the data from the JSONL file
data = pd.read_json('Data/preprocessed_health_data2.json', lines=True, encoding='latin-1')

# Convert the data to a DataFrame
df = pd.DataFrame(data)
print('Data initialized')

# Map ratings to classes
rating_mapping = {1: 'negative', 2: 'negative', 3: 'neutral', 4: 'positive', 5: 'positive'}
df['rating_mapped'] = df['rating'].map(rating_mapping)

# Splitting features and target variable
X = df['text_final']
y = df['rating_mapped']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=9)

Encoder = LabelEncoder()
y_train = Encoder.fit_transform(y_train)
y_test = Encoder.fit_transform(y_test)

textVectorizer = joblib.load('vectorizer.pkl')
Train_X_Tfidf = textVectorizer.transform(X_train)
Test_X_Tfidf = textVectorizer.transform(X_test)

# Define and train the SVM model with verbosity
n_estimators = 1
svm_model = OneVsRestClassifier(BaggingClassifier(SVC(C=10, kernel='sigmoid', gamma=0.5, coef0=0.5, verbose=True, cache_size=7500), max_samples=1.0 / n_estimators, n_estimators=n_estimators))
svm_model.fit(Train_X_Tfidf, y_train)

print(svm_model.get_params())
print("Finished training")

# Predicting on the test set
y_predicted = svm_model.predict(Test_X_Tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_predicted)
print(f'Test Accuracy: {accuracy}')

# Save the trained model to a file
joblib.dump(svm_model, 'svm_model_linear_7.pkl')
