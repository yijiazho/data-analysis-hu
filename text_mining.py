import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.stem.snowball import SnowballStemmer
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

#url = 'https://raw.githubusercontent.com/sinanuozdemir/sfdat22/master/data/yelp.csv'

url = 'yelp.csv'
yelp = pd.read_csv(url, encoding = "ISO-8859-1")
reviews = yelp['text']
vectorizer = CountVectorizer(encoding='utf-8')
review_words = vectorizer.fit_transform(reviews)
review_words[:3]
print(len(vectorizer.vocabulary_))
print(list(vectorizer.vocabulary_.items()))

## Stop Words Removal and Stemming/Lemmatization

vectorizer_stopwords = CountVectorizer(encoding='utf-8', stop_words='english', lowercase=True)
review_words_stopwords = vectorizer_stopwords.fit_transform(reviews)


sno = nltk.stem.SnowballStemmer('english')

def stemming_tokenizer(str_input):
  words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
  words = [sno.stem(word) for word in words]
  return words
  
vectorizer_stem = CountVectorizer(encoding='utf-8', stop_words='english', lowercase=True, tokenizer=stemming_tokenizer)
review_words_stem = vectorizer_stem.fit_transform(reviews)
print(len(vectorizer_stem.vocabulary_))

## Naive Bayes Modeling

X = yelp['text'].values
Y = yelp['stars'].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=8)
X_train_vec_nb = vectorizer_stopwords.fit_transform(X_train)
X_test_vec_nb = vectorizer_stopwords.transform(X_test)

nb_clf = MultinomialNB()
nb_clf.fit(X_train_vec_nb, y_train)

y_pred_mnb = nb_clf.predict(X_test_vec_nb)
confusion_matrix(y_test, y_pred_mnb, labels=[1, 2, 3, 4, 5])
print(classification_report(y_test, y_pred_mnb))