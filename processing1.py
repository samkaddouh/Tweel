import pickle as pkl
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import pandas as pd
#from langdetect import detect_langs
from sentiment import *
from collections import defaultdict


auth = tw.OAuthHandler("Consumer Key", "Consumer Secret")
auth.set_access_token("API Token Key", "API Token Secret")
api = tw.API(auth)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')


tweets = []

for tweetobj in set_of_tweets:
    for tweet in tweetobj[0]:
        tweets.append([tweetobj[1], tweet.user.name , tweet._json['full_text']])

cleaned_tweets = []

for t in tweets:
    s = re.sub(r"@\w*", "", t[2].lower())
    s = re.sub(r"[^a-z0-9 ]","", t[2].lower())
    s = re.sub(r"http\w*","", s)
    cleaned_tweets.append(s)

stop_words = set(stopwords.words('english'))

tweets, cleaned_tweets = clean_tweets(set_of_tweets)
countryTweets = {}
countryTweets['UK'] = []
countryTweets['Australia'] = []
countryTweets['USA'] = []
All_tweets = []

for tweet, main_tweet in zip(cleaned_tweets, tweets):
    countryTweets[main_tweet[0]].append(tweet)
    All_tweets.append(tweet)
#print(word_tokens)
#print(filtered_sentence)
#print(cleaned_tweets)
#print(All_tweets)
#print(str(countryTweets['UK']))
#print(str(countryTweets['Australia']))
#print(str(countryTweets['USA']))
UK_full_sentence = re.sub(r"[^a-z0-9 ]","", (str(countryTweets['UK'])))
Australia_full_sentence = re.sub(r"[^a-z0-9 ]","", (str(countryTweets['Australia'])))
USA_full_sentence = re.sub(r"[^a-z0-9 ]","", (str(countryTweets['USA'])))
All_Country_Tweets = UK_full_sentence + ' ' + Australia_full_sentence + ' ' + USA_full_sentence

print(UK_full_sentence)
print(Australia_full_sentence)
print(USA_full_sentence)
print(All_Country_Tweets)

def save_WordClouds(sentence, country):
    #my_stopwords = set(STOPWORDS)
    cloud = WordCloud(background_color='black', collocations=False).generate(sentence)
    cloud.to_file("Dummy_" + country + ".png")
    return
    plt.imshow(cloud, interpolation='bilinear')
    plt.title(country)
    plt.axis('off')
    #plt.show()
    plt.savefig("WordCloud_" + country + ".png")



save_WordClouds(UK_full_sentence, 'UK')
save_WordClouds(Australia_full_sentence, 'AUS')
save_WordClouds(USA_full_sentence, 'USA')
save_WordClouds(All_Country_Tweets, 'ALL')
exit(1)
#WordCloud
'''
cloud_country_tweets_Aus = WordCloud(background_color='black', collocations=False, stopwords=my_stopwords).generate(Australia_full_sentence)
plt.imshow(cloud_country_tweets_Aus, interpolation='bilinear')
plt.axis('off')
plt.show()

cloud_country_tweets_USA = WordCloud(background_color='black', collocations=False, stopwords=my_stopwords).generate(USA_full_sentence)
plt.imshow(cloud_country_tweets_USA, interpolation='bilinear')
plt.axis('off')
plt.show()

cloud_country_tweets_All = WordCloud(background_color='black', collocations=False, stopwords=my_stopwords).generate(All_Country_Tweets)
plt.imshow(cloud_country_tweets_All, interpolation='bilinear')
plt.axis('off')
plt.show()
'''

#Polarity and Subjectivity Sentiment - TextBlob object and library

def generate_sentiment(sentence):
    blob_tweets = TextBlob(sentence)
    #print(blob_tweets.sentiment)
    Polarity=blob_tweets.sentiment.polarity
    Subjectivity=blob_tweets.sentiment.subjectivity
    Sentiment = [Polarity, Subjectivity]
    return Sentiment

UK_Sentiment = generate_sentiment(UK_full_sentence)
AUS_Sentiment = generate_sentiment(Australia_full_sentence)
USA_Sentiment = generate_sentiment(USA_full_sentence)
All_Sentiment = generate_sentiment(All_Country_Tweets)

'''
blob_Australia_tweets = TextBlob(Australia_full_sentence)
print(blob_Australia_tweets.sentiment)
AUS_Polarity=blob_Australia_tweets.sentiment.polarity
AUS_Subjectivity=blob_Australia_tweets.sentiment.subjectivity
AUS_Sentiment = [AUS_Polarity, AUS_Subjectivity]

blob_USA_tweets = TextBlob(USA_full_sentence)
print(blob_USA_tweets.sentiment)
USA_Polarity=blob_USA_tweets.sentiment.polarity
USA_Subjectivity=blob_USA_tweets.sentiment.subjectivity
USA_Sentiment = [USA_Polarity, USA_Subjectivity]

blob_All_tweets = TextBlob(All_Country_Tweets)
print(blob_All_tweets.sentiment)
All_Polarity=blob_All_tweets.sentiment.polarity
All_Subjectivity=blob_All_tweets.sentiment.subjectivity
All_Sentiment = [All_Polarity, All_Subjectivity]
'''

#Histogram 1D of Polarity
def generate_histogram():
    x = [UK_Sentiment[0], AUS_Sentiment[0], USA_Sentiment[0], All_Sentiment[0]]
    plt.hist(x, bins=20, range=(-1, 1))
    plt.title("Polarity")
    plt.xlabel("Country")
    plt.ylabel("Polarity")
    plt.show()
    #TODO add subjectivity 1D plot

def generate_histogram():
    x = [UK_Sentiment[1], AUS_Sentiment[1], USA_Sentiment[1], All_Sentiment[1]]
    plt.hist(x, bins=20, range=(-1, 1))
    plt.title("Subjectivity")
    plt.xlabel("Country")
    plt.ylabel("Subjectivity")
    plt.show()


'''
Example of a sample grouped bar graph

simple reference: https://www.geeksforgeeks.org/create-a-grouped-bar-plot-in-matplotlib/
'''
def generate_bar_plots(sentiment):
    xs = np.arange(len(sentiment))
    pols = []
    subs = []
    names = []
    for country, score in sentiment.items():
        pol, sub = score
        pols.append(pol)
        subs.append(sub)
        names.append(country)

    # a couple of assert statements to verify the values are in range.
    assert all(-1 <= p <= 1 for p in pols), 'Polarity must be in [-1, 1] range!'
    assert all(0  <= s <= 1 for s in subs), 'Subjectivity must be in [0, 1] range!'

    width = 0.4   # this is the width of the bar. Try changing it!

    # subjectivity values are already in [0, 1] range.
    # polarity values are in [-1, 1]. Lets bring them to [0, 1] range too.
    for i, pol in enumerate(pols):
        scaled_val = (pol + 1) / 2   # think about this calculation :)
        pols[i] = scaled_val


    # x value is where the bar will appear.
    # y value is the y-axis value itself.
    # see 'matplotlib.pyplot.bar()' for more
    plt.bar(xs - width/2, pols, width, color = 'lightblue')
    plt.bar(xs + width/2, subs, width, color = 'darkblue')

    plt.xticks(xs, names)   # this writes country names as x ticks
    plt.xlabel("Country")
    plt.ylabel("sentiment score")
    plt.legend(["Polarity", "Subjectivity"])
    plt.title("Sentiment score values by region")
    plt.show()
    #plt.savefig('sentiment.png')   # uncomment to save to a file instead.


# some dummy sentiment values
# polarity values are in the range [-1, 1] and
# subjectivity ranges from [0, 1]
sentiment_dict = dict()
sentiment_dict['UK']  = [-0.5, 0.77]
sentiment_dict['US']  = [0.7, 0.63]
sentiment_dict['AUS'] = [0.8, 0.1]
sentiment_dict['All'] = [-0.3, 0.5]

# usage of the function:
generate_bar_plots(sentiment_dict)

#for one country: polarity on y-axis and country on x-axis
#Histogram 1D of Sub



#my_stop_words = ENGLISH_STOP_WORDS.union(['018hekate', '22sweet', 'donut', 'donuts'])

def generate_top_frequent_words(tweets):
    vocab = defaultdict(lambda:0)
    for tweet in tweets:
        words = tweet.split()
        for w in words:
            vocab[w] += 1

    word_frequency = [(w,f) for w,f in vocab.items()]
    word_frequency.sort(key=lambda x:x[1], reverse=True)
    return word_frequency[:10]

def plot_top_frequent_words(tweets, country):
    top_words = generate_top_frequent_words(tweets)
    xs = np.arange(len(tweets))
    ys = [tw[1] for tw in top_words]
    words = [tw[0] for tw in top_words]
    plt.scatter(xs, ys, marker='o', color='lightblue')
    plt.xticks(xs, words)
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.title("Top Frequent Words for %s" % country)
    plt.savefig("top_Frequent_words_" + country + ".png")

def create_df_BOW_All(tweets):
    top_words = generate_top_frequent_words(tweets)
    df = []
    words=[tw[0] for tw in top_words]
    for tweet in tweets:
        tweet_vec = []
        tweet_words = tweet.split()
        for w in words:
            tweet_vec.append(tweet_words.count(w))
        df.append(tweet_vec)
    return pd.DataFrame(df, columns=words)


Vect = CountVectorizer(max_features=100, max_df=100)
Vect.fit(countryTweets)
Bow = Vect.transform(countryTweets)
#print(UK_Bow.toarray())
df = pd.DataFrame(Bow.toarray(), columns=Vect.get_feature_names_out())
print(df)



#at what cost if i add bigram and trigrams(computation time or extra space usage)

#BOW AUS
Aus_Vect = CountVectorizer(max_features=10, max_df=100, ngram_range=(1,2)) #ngram_range=(1,2) means unigrams and bigrams
Aus_Vect.fit(countryTweets['Australia'])
Aus_Bow = Aus_Vect.transform(countryTweets['Australia'])
#print(UK_Bow.toarray())
Aus_df = pd.DataFrame(Aus_Bow.toarray(), columns=Aus_Vect.get_feature_names_out())
print(Aus_df)

#BOW USA
USA_Vect = CountVectorizer(max_features=10, max_df=100)
USA_Vect.fit(countryTweets['USA'])
USA_Bow = USA_Vect.transform(countryTweets['USA'])
#print(UK_Bow.toarray())
USA_df = pd.DataFrame(USA_Bow.toarray(), columns=USA_Vect.get_feature_names_out())
print(USA_df)

#UK Word Tokens
UK_tokens = [word_tokenize(UK_full_sentence) for UK_full_sentence in countryTweets['UK']]
print(UK_tokens)
UK_len_tokens=[]
for i in range(len(UK_tokens)):
    UK_len_tokens.append(len(UK_tokens[i]))
countryTweets['UK','n_tokens'] = UK_len_tokens
#length of tokens in each tweet
print(UK_len_tokens)

#AUS Word Tokens
AUS_tokens = [word_tokenize(Australia_full_sentence) for Australia_full_sentence in countryTweets['Australia']]
print(AUS_tokens)
AUS_len_tokens=[]
for i in range(len(AUS_tokens)):
    AUS_len_tokens.append(len(AUS_tokens[i]))
countryTweets['Australia','n_tokens'] = AUS_len_tokens
print(AUS_len_tokens)

#USA Word Tokens
USA_tokens = [word_tokenize(USA_full_sentence) for USA_full_sentence in countryTweets['USA']]
print(USA_tokens)
USA_len_tokens=[]
for i in range(len(USA_tokens)):
    USA_len_tokens.append(len(USA_tokens[i]))
countryTweets['USA','n_tokens'] = USA_len_tokens
#length of tokens in each tweet
print(USA_len_tokens)

#Creating a dataframe for the sentiment of all the tweets from a country
UK_tweets_df = [UK_Sentiment, AUS_Sentiment, USA_Sentiment, All_Sentiment]
df = pd.DataFrame(UK_tweets_df, columns=['Polarity','Subjectivity'])
print(df)
#BOW is a sparse matrix any way to reduce saprsity? what if we get rid of 10% of thr least frequent words
#Creating a dataframe for all tweets
#tweets_df = [All_tweets]
#tweets_df = pd.DataFrame(tweets_df, columns= ['1','2','1','2','1','2'])
#print(tweets_df)
#tweets_df.head()

#Detecting Language of Tweet
#print(detect_langs(UK_full_sentence))
#print(detect_langs(Australia_full_sentence))
#print(detect_langs(USA_full_sentence))

#Find language of the tweets of each country
languagesUK = []
languagesAUS = []
languagesUSA = []
languages = []
for langUK in countryTweets['UK']:
    languagesUK.append(detect_langs(langUK))
print(languagesUK)

for langAUS in countryTweets['Australia']:
    languagesAUS.append(detect_langs(langAUS))
print(languagesAUS)

for langUSA in countryTweets['USA']:
    languagesUSA.append(detect_langs(langUSA))
print(languagesUSA)

#List of all tweets with language of tweet
#for row in range(len(tweets_df)):
#    languages.append(detect_langs(tweets_df.iloc[row,1]))
#languages = [str(lang).split(':')[0][1:]for lang in languages]
#tweets_df['language']= languages
#print(tweets_df.head())







#     #histogram - for different countries study comparison
#     #what relationship can be dedeuced from
#     #3d graph - x =
#     # histogram one dimensional of polarity (each country)
#     # histogram one dimensional of subjectivity (each country)
#     # 2d histogram - x-axis :polarity, y-axis:subjectivity
#
#     #BAG of WORDS
#     nso = new_sentence.split()
#     #print(nso)
#     vect = CountVectorizer(max_features=10, ngram_range=(1,2), max_df=500)  ##how many features?
#     vect.fit(nso)
#     X = vect.transform(nso)
#     X_df = pd.DataFrame(X.toarray(), columns=vect.get_feature_names_out())
#     print(X_df.head())
