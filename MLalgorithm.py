import pandas as pd
import numpy as np
import string
import re
import nltk
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def csv2dataset(datafile):
    x = []
    y = []
    with open(datafile) as f:
        for line in f:
            toks = line.split(',')
            sentiment = toks[1]
            tweet = " ".join(toks[3:])
            x.append(tweet)
            y.append(sentiment)

    dataset = pd.DataFrame()
    dataset['target'] = y[1:]
    dataset['text'] = x[1:]
    return dataset
#dataset = dataset[:500]
#print(dataset.head())
stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']



#removing stopwords
STOPWORDS = set(stopwordlist)

def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


#print(dataset.head(20))
#exit()
english_punctuations = string.punctuation
punctuations_list = english_punctuations

def cleaning_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)

def cleaning_repeating_char(text):
    return re.sub(r'(.)1+', r'1', text)

def cleaning_URLs(data):
    return re.sub('((www.[^s]+)|(https?://[^s]+))',' ',data)

def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)

tokenizer = RegexpTokenizer('\s+', gaps = True)

st = nltk.PorterStemmer()
def stemming_on_text(data):
    text = [st.stem(word) for word in data]
    return text

lm = nltk.WordNetLemmatizer()
def lemmatizer_on_text(data):
    text = [lm.lemmatize(word) for word in data]
    return text

# dataset['final'] = dataset['text'].apply(lambda list_of_words: " ".join(list_of_words))
# dataset_neg = dataset[dataset['target'] == '0']['final']


def clean(dataset, write_to_file=False, filename='Cleaned.csv'):
    # converting to lowercase
    dataset['text'] = dataset['text'].astype(str).str.lower() #make sure its a string always else arabic characters were failing
    dataset['text'] = dataset['text'].apply(lambda text: cleaning_stopwords(text))
    dataset['text'] = dataset['text'].apply(lambda x: cleaning_punctuations(x))
    dataset['text'] = dataset['text'].apply(lambda x: cleaning_repeating_char(x))
    dataset['text'] = dataset['text'].apply(lambda x: cleaning_URLs(x))
    dataset['text'] = dataset['text'].apply(lambda x: cleaning_numbers(x))
    dataset['text'] = dataset['text'].apply(tokenizer.tokenize)
    dataset['text'] = dataset['text'].apply(lambda x: stemming_on_text(x))
    dataset['text'] = dataset['text'].apply(lambda x: lemmatizer_on_text(x))
    if write_to_file:
        dataset.to_csv(filename, index=False)
    return dataset


#dataset = csv2dataset(datafile = 'Sentiment Analysis Dataset.csv')

#clean(dataset, write_to_file=True, filename='Cleaned_dataset.csv')


