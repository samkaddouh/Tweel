import tweepy as tw
import io
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
import numpy as np
from wordcloud import WordCloud
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from textblob import TextBlob
from collections import defaultdict

#Authenticate to Twitter
#auth = tw.OAuthHandler("", "")
#auth.set_access_token("", "")
#api = tw.API(auth)

# try:
#     api.verify_credentials()
#     print("Authentication OK")
# except:
#     print("Error during authentication")

def get_insights(api,search_words, max_tweets = 100):
    search_words = search_words.lower()
    set_of_tweets = []

    country = dict()
    country['UK'] = '54.160394,-4.579683,500km'
    country['Australia'] = '-24.489606,134.127855,1500km'
    country['USA'] = '40.408169,-99.034083,1500km'


#search_words = 'UKRAINE'
    new_search = search_words + " -filter:retweets"
    if search_words == 'metaverse':
        date_since = "2021-10-28"
    else:
        date_since = '2022-20-02'

    for location in country:
        tweets = [status for status in tw.Cursor(api.search_tweets,
                                                 q=new_search,
                                                 geocode=country[location],
                                                 since_id= date_since,
                                                 tweet_mode = 'extended',
                                                 lang="en").items(max_tweets)]
        set_of_tweets.append((tweets, location))
    return set_of_tweets

    '''
    with open("results.pkl", "wb") as a_file:
        pickle.dump(set_of_tweets, a_file)
    '''

def clean_tweets(set_of_tweets):
    tweets = []

    for tweetsobj in set_of_tweets:
        for tweet in tweetsobj[0]:
            tweets.append([tweetsobj[1], tweet.user.name, tweet._json['full_text']])

    cleaned_tweets = []
    stop_words = set(stopwords.words('english'))
    for t in tweets:
        s = re.sub(r"@\w*", "", t[2].lower())
        s = re.sub(r"[^a-z0-9 ]","", t[2].lower())
        s = re.sub(r"http\w*","", s)
        word_tokens = word_tokenize(s)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        s = ' '.join(filtered_sentence)
        cleaned_tweets.append(s)
#TODO refer to online link, can we reduce regex's?
    return tweets, cleaned_tweets
#print('\n-'.join(cleaned_tweets))
#word_tokens2 = word_tokenize(str(cleaned_tweets))
#print('\n'.join(word_tokens2))

def analyze_tweets(tweets, cleaned_tweets):
    analyzer = SentimentIntensityAnalyzer()
    for tweet, main_tweet in zip(cleaned_tweets, tweets):
        #print(word_tokens)
        #print(filtered_sentence)
        #print(new_sentence)
        ss = analyzer.polarity_scores(tweet)
        main_tweet.append(ss['compound'])
        main_tweet.append(ss['neg'])
        main_tweet.append(ss['neu'])
        main_tweet.append(ss['pos'])
    negAverage = {}
    neuAverage = {}
    posAverage = {}
    comAverage = {}
    num = {}
    for t in tweets:
        if t[0] not in negAverage.keys():
            negAverage[t[0]] = t[3]
            neuAverage[t[0]] = t[4]
            posAverage[t[0]] = t[5]
            comAverage[t[0]] = t[6]
            num[t[0]] = 1.0
        else:
            negAverage[t[0]] = negAverage[t[0]] + t[3]
            neuAverage[t[0]] = neuAverage[t[0]] + t[4]
            posAverage[t[0]] = posAverage[t[0]] + t[5]
            comAverage[t[0]] = comAverage[t[0]] + t[6]
            num[t[0]] = num[t[0]] + 1.0
    for i in negAverage.keys():
        negAverage[i] = negAverage[i]/num[i]
        neuAverage[i] = neuAverage[i]/num[i]
        posAverage[i] = posAverage[i]/num[i]
        comAverage[i] = comAverage[i]/num[i]
    total_average = []
    for i in negAverage.keys():
        total_average.append([i, negAverage[i], neuAverage[i], posAverage[i], comAverage[i]])
    return total_average

def save_pies(scores):
    plots=dict()
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Negative', 'Neutral', 'Positive'
    for s in scores:
        sizes = [abs(s[1])] + s[2:4]
        explode = (0.1,0,0)  # only "explode" the 2nd slice (i.e. 'Hogs')
        fig=Figure()
        ax1=fig.add_subplot(1,1,1)

        #fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        save_file='./static/pie_'+s[0]+'.png'
        #plt.savefig(save_file)
        #plots.append(save_file)
        plots[s[0]]=save_file
        output=io.BytesIO()
        FigureCanvas(fig).print_png(output)
        with open(save_file, "wb") as f:
            f.write(output.getvalue())
    return plots

def save_WordClouds(sentence, country):
    cloud = WordCloud(background_color='white', collocations=False).generate(sentence)
    save_file = "WordCloud_" + country + ".png"
    cloud.to_file('./static/'+save_file)
    return save_file

#Histogram 1D of Polarity
def generate_histogram(sentiment, i, name):
    bars=[]
    countries=[]
    width=0.4
    for c,v in sentiment.items():
        bars.append(v[i])
        countries.append(c)
    xs=np.arange(len(bars))
    fig = Figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.bar(xs, bars, width, color='lightblue')
    ax1.set_xticks(xs, countries)  # this writes country names as x ticks
    ax1.title.set_text(name)
    ax1.set_xlabel("Country")
    ax1.set_ylabel(name)
    save_file = './static/sentiment_' + name + '.png'
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    with open(save_file, "wb") as f:
        f.write(output.getvalue())
    return save_file

def generate_sentiment(sentence):
    blob_tweets = TextBlob(sentence)
    #print(blob_tweets.sentiment)
    Polarity=blob_tweets.sentiment.polarity
    Subjectivity=blob_tweets.sentiment.subjectivity
    Sentiment = [Polarity, Subjectivity]
    return Sentiment



#NORMALIZED THE TWO PARAMATERS TO (0:1) INSTEAD OF (-1,1) AND (0,1) FOR POLARITY AND SUBJECTIVITY RESPECTIVELY
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

    width = 0.4   # this is the width of the bar.

    # subjectivity values are already in [0, 1] range.
    # polarity values are in [-1, 1]. Lets bring them to [0, 1] range too.
    for i, pol in enumerate(pols):
        scaled_val = (pol + 1) / 2
        pols[i] = scaled_val

    fig = Figure()
    ax1 = fig.add_subplot(1, 1, 1)


    # x value is where the bar will appear.
    # y value is the y-axis value itself.
    # see 'matplotlib.pyplot.bar()' for more
    ax1.bar(xs - width/2, pols, width, color = 'lightblue')
    ax1.bar(xs + width/2, subs, width, color = 'darkblue')
    ax1.set_xticks(xs, names)   # this writes country names as x ticks
    ax1.set_xlabel("Country")
    ax1.set_ylabel("Sentiment")
    ax1.legend(["Polarity", "Subjectivity"])
    ax1.title.set_text("Sentiment score values by region")

    save_file = './static/sentiment_grouped.png'
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    with open(save_file, "wb") as f:
        f.write(output.getvalue())
    return save_file

def generate_top_frequent_words(tweets,n=10):
    vocab = defaultdict(lambda:0)
    for tweet in tweets:
        words = tweet.split()
        for w in words:
            vocab[w] += 1

    word_frequency = [(w,f) for w,f in vocab.items()]
    word_frequency.sort(key=lambda x:x[1], reverse=True)
    return word_frequency[:n]



def generate_BOW_plots(tweets, country_name):
    topwords= generate_top_frequent_words(tweets, 10)
    xs=np.arange(len(topwords))
    y=[tw[1] for tw in topwords]
    words=[tw[0] for tw in topwords]

    fig = Figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.scatter(xs, y, marker='.')
    ax1.plot(xs,y, '--', color="lightblue")
    ax1.set_xticks(xs, words)  # this writes country names as x ticks
    ax1.title.set_text(country_name)
    ax1.set_xlabel("Words")
    ax1.set_ylabel("Frequency")
    save_file = './static/BOW_' + country_name + '.png'
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    with open(save_file, "wb") as f:
        f.write(output.getvalue())
    return save_file

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
