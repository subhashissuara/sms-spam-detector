# Importing libraries
import pandas as pd
import pickle
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Loading the dataset
df = pd.read_csv("dataset.csv", encoding="latin-1")

# Preprocessing the dataset
# Dropping unnecessary columns 
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# Renaming columns
df.rename(columns = {"v1": "label", "v2": "message"}, inplace=True)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Cleaning the messages
corpus = []
ps = PorterStemmer()

for sms_string in list(df.message):

  # Cleaning special character from the message
  message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sms_string)

  # Converting the entire message into lower case
  message = message.lower()

  # Tokenizing the review by words
  words = message.split()

  # Removing the stop words
  words = [word for word in words if word not in set(stopwords.words('english'))]

  # Stemming the words
  words = [ps.stem(word) for word in words]

  # Joining the stemmed words
  message = ' '.join(words)

  # Building a corpus of messages
  corpus.append(message)
  
# Creating the Bag of Words model
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

# Extracting dependent variable from the dataset
y = pd.get_dummies(df['label'])
y = y.iloc[:, 1].values

# Creating a pickle file for the CountVectorizer
pickle.dump(cv, open('cv.pkl', 'wb'))

# Building Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=96)

# Fitting Naive Bayes to the Training set
classifier = MultinomialNB(alpha=0.3)
classifier.fit(X_train, y_train)

# Creating a pickle file for the Multinomial Naive Bayes model
pickle.dump(classifier, open('model.pkl', 'wb'))