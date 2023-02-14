import json
import sys
import tweepy as tw
import geocoder
import pandas as pd
import MLalgorithm, modeling
from sklearn.naive_bayes import BernoulliNB
from datetime import date
import time
import pickle


'''
def handle_post_error(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except tw.TweepError as e:
            if e.api_code == 429:
                # Too many requests, wait and retry
                time.sleep(60)
                return func(*args, **kwargs)
            else:
                raise e
    return wrapper
'''

def populate_locs_for_trends(api):

    available_loc = api.available_trends()
    with open("available_locs_for_trend.json", "w") as wp:
        wp.write(json.dumps(available_loc, indent=1))


def get_trends(api, loc):
    g = geocoder.osm(loc)
    closest_loc = api.closest_trends(g.lat, g.lng)
    trends = api.get_place_trends(closest_loc[0]['woeid'])
    #print("i am in get_trends")
    #print(loc)
    #print(g)
    #print(closest_loc)
    #print(trends)
    with open("twitter_{}_trend.json".format(loc), "w") as wp:
        wp.write(json.dumps(trends, indent=1))
    return trends


def extract_hashtags(trends):
    #hashtags = [trend["name"] for trend in trends if "#" in trend["name"]]
    hashtags = []
    nonHash = []
    #print(trends)
    for obj in trends:
        trendArr = obj["trends"]
        for ti in trendArr:
            name = ti['name']
            if '#' in name:
                hashtags.append(name)
            else:
                nonHash.append(name)
    return hashtags, nonHash

#hashtags,nonHash = extract_hashtags(trends)
#print(hashtags)
#print(nonHash)

def extract_tweets(api, list_of_trends, country):
    tweet_dict = dict()
    for topic in list_of_trends:
        tweets = get_tweets(api, topic, country)
        tweet_dict[topic] = tweets
    return tweet_dict

def get_tweets(api, search_words, country ,max_tweets=5):
    search_words = search_words.lower()
    set_of_tweets = dict()
    g = geocoder.osm(country)
    closest_loc = api.closest_trends(g.lat, g.lng)
    new_search = search_words + " -filter:retweets"


    
    tweets = [status for status in tw.Cursor(api.search_tweets,
                                                 q=new_search,
                                                 geocode=",".join([str(g.lat),str(g.lng),"25km"]),
                                                 tweet_mode='extended',
                                                 lang="en").items(max_tweets)]
    set_of_tweets[country] = tweets
    return set_of_tweets


def analyse_topics(api, list_of_topics, modeller, topic_type, country):

    tweet_dict = extract_tweets(api, list_of_topics, country)

#print(tweet_dict)

    flat_dict = {'topic':[], 'location':[], 'text':[], 'tweet':[]}
    for topic in tweet_dict:
        topic_dict=tweet_dict[topic]
        for location in topic_dict:
            status_list = topic_dict[location]
            for status in status_list:
                #print(status.full_text)
                tweet = status.full_text
                flat_dict['topic'].append(topic)
                flat_dict['location'].append(location)
                flat_dict['text'].append(tweet)
                flat_dict['tweet'].append(tweet)
    flat_dataset =pd.DataFrame(flat_dict)


    test_dataset = MLalgorithm.clean(flat_dataset, write_to_file=True, filename='test_tweets_cleaned.csv')


    #loading from file
    test_dataset = pd.read_csv('test_tweets_cleaned.csv')
    #print(test_dataset.head())

    
    
    BNBmodel = BernoulliNB()
    modeller.set_classifier(BNBmodel)
    y_pred1 = modeller.predict(test_dataset.text)


    

    result_df = pd.DataFrame({'pred': y_pred1, 'topic':test_dataset.topic, 'text':flat_dataset.tweet})
    today = date.today()
    result_df.to_csv('result_'+topic_type+str(today)+'.csv', index=False)
    #print('done')
    return result_df


#modeller = modeling.Modeller('Cleaned_dataset.csv')
#analyse_topics(hashtags,modeller, 'hash')
#analyse_topics(nonHash[:10],modeller, 'nonHash')

def handling_sentiment(df):
    #create a new df with three columns : topic, count_1, percenatage_pos=count_1/count for each and every topic
    #df = pd.DataFrame({'pred':y_pred1, 'count':count, 'count_1':count_1})
    df1 = df.drop(columns=['text']).groupby('topic')['pred'].agg(['sum','count'])
    df1.rename(columns = {'sum':'pred', 'count':'tweets'}, inplace = True)
    df1['percenatage_pos'] = 100 * df1['pred'] / df1['tweets']
    df2 = df1.sort_values('percenatage_pos', ascending=False)
    df3 = df2.reset_index()
    df4 = df3[['topic', 'percenatage_pos']]
    return df4

def get_woeid(place):
    '''Get woeid by location'''
    try:
        trends = api.trends_available()
        for val in trends:
            if (val['name'].lower() == place.lower()):
                return(val['woeid']) 
        #print('Location Not Found')
    except Exception as e:
        #print('Exception:',e)
        return(0)


def countries_drop_down(api):
    response = api.available_trends()
    trends1 = {}
       
    for d in response:
        country1 = d['country']
        city = d['name']
        if country1 in trends1:
            if not isinstance(trends1[country1],list):
                trends1[country1] = [trends1[country1]]
            trends1[country1].append(city)
        else:
            trends1[country1] = [city]
        
    return trends1
      


           







'''
with open('tokens.secret') as f:
    tokens= json.load(f)

c_key = tokens["consumer_key"]
c_secret = tokens["consumer_secret"]
acc_tok = tokens["access_token"]
acc_tok_secret = tokens["access_token_secret"]    
auth = tw.OAuthHandler(c_key, c_secret)
auth.set_access_token(acc_tok, acc_tok_secret)
api = tw.API(auth)
loc = "UK"
ok=get_trends(api, loc)

print(ok)
'''


