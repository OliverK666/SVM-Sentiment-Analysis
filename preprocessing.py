import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from collections import defaultdict

# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
data = pd.read_json('Data/preprocessed_health_data.json', lines=True)

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Step - a : Remove blank rows if any.
df['title'].dropna(inplace=True)
print('================================')

# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
df['title'] = [entry.lower() for entry in df['title']]
print('================================')

# Step - c : Tokenization : In this each entry in the df will be broken into set of words
df['title'] = [word_tokenize(entry) for entry in df['title']]
print('================================')

# Step - d : Remove Stop words, Non-Numeric and perform Word Stemming/Lemmenting.
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc.
# By default, it is set to Noun
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
print('================================')
for index, entry in enumerate(df['title']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e. if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    df.loc[index, 'title_final'] = str(Final_words)

df.to_json('Data/preprocessed_health_data2.json', orient='records', lines=True)
