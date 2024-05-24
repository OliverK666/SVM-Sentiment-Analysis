import joblib
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

svm_model = joblib.load('Models\svm_model_linear_2.pkl')
textVectorizer = joblib.load('vectorizer.pkl')
print('Data initialized')
print(svm_model.get_params())

# Initialize WordNetLemmatizer and tag_map
word_Lemmatized = WordNetLemmatizer()
tag_map = defaultdict(lambda: 'n')
tag_map['J'] = 'a'
tag_map['V'] = 'v'
tag_map['R'] = 'r'

text = ''
# Get user input for text data
while True:
    text = input('Enter the text (exit to exit): ')

    if text == 'exit':
        break

    # Tokenize the input sentence
    tokens = text.split()

    # Initialize Final_words list to store processed words
    Final_words = []

    # Perform POS tagging and lemmatization
    for word, tag in pos_tag(tokens):
        # Check for stopwords and alphabets
        if word.lower() not in stopwords.words('english') and word.isalpha():
            # Lemmatize the word using the appropriate POS tag
            word_Final = word_Lemmatized.lemmatize(word, tag_map.get(tag[0].upper(), 'n'))
            # Append the lemmatized word to Final_words list
            Final_words.append(str(word_Final))

    # Make predictions on the custom example
    dfTesting = pd.DataFrame()
    dfTesting.loc[0, 'texter'] = str(Final_words)
    vector = textVectorizer.transform(dfTesting['texter'])
    predicted_result = svm_model.predict(vector)
    print(dfTesting['texter'].iloc[0])

    # Interpret the predictions
    if predicted_result == 0:
        print(f'Predicted class for the custom example: Negative')
    elif predicted_result == 1:
        print(f'Predicted class for the custom example: Neutral')
    else:
        print(f'Predicted class for the custom example: Positive')
