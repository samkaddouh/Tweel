from flask import Flask, url_for, render_template, request
import tweepy as tw
from sentiment import *
import json
app = Flask(__name__)

#Authenticate to Twitter
with open('tokens.secret') as f:
    tokens= json.load(f)

c_key = tokens["consumer_key"]
c_secret = tokens["consumer_secret"]
acc_tok = tokens["access_token"]
acc_tok_secret = tokens["access_token_secret"]    
auth = tw.OAuthHandler(c_key, c_secret)
auth.set_access_token(acc_tok, acc_tok_secret)
api = tw.API(auth)

try:
    api.verify_credentials()
    print("Authentication OK")
except:
    print("Error during authentication")

# @app.route('/', methods =['GET'])
# def hello_world():  # put application's code here
#     # print("Sam")
#     # return '<h1>About</h1'
#     return render_template('main.html')
@app.route('/', methods =['GET','POST'])
def get_data():
    if request.method == 'GET':
        return render_template('main.html')
    elif request.method == 'POST':
        keyword = request.form['search']
        keyword=keyword.strip().lower()
        if keyword == '':
            return "Please enter a keyword"
        if not keyword.isalnum() or len(keyword)>16:
            return "Please enter only alpha numeric keywords of length <= 16"

        tweets = get_insights(api, keyword)
        tweets, cleaned_tweets = clean_tweets(tweets)
        scores = analyze_tweets(tweets, cleaned_tweets)
        global PIE_FILES
        PIE_FILES=save_pies(scores)

        countryTweets = {} #TODO makes more sense to make all tweets global
        countryTweets['UK'] = []
        countryTweets['Australia'] = []
        countryTweets['USA'] = []
        All_tweets = []

        for tweet, main_tweet in zip(cleaned_tweets, tweets):
            countryTweets[main_tweet[0]].append(tweet)
            All_tweets.append(tweet)

        UK_full_sentence = re.sub(r"[^a-z0-9 ]", "", (str(countryTweets['UK'])))
        Australia_full_sentence = re.sub(r"[^a-z0-9 ]", "", (str(countryTweets['Australia'])))
        USA_full_sentence = re.sub(r"[^a-z0-9 ]", "", (str(countryTweets['USA'])))
        All_Country_sentence = UK_full_sentence + ' ' + Australia_full_sentence + ' ' + USA_full_sentence
        sentiment = dict()
        sentiment['UK']=generate_sentiment(UK_full_sentence)
        sentiment['AUS'] = generate_sentiment(Australia_full_sentence)
        sentiment['USA'] = generate_sentiment(USA_full_sentence)
        sentiment['ALL'] = generate_sentiment(All_Country_sentence)
        BOW_PLOTS['UK']=generate_BOW_plots(countryTweets['UK'], 'UK')
        BOW_PLOTS['Australia'] = generate_BOW_plots(countryTweets['Australia'], 'Australia')
        BOW_PLOTS['USA'] = generate_BOW_plots(countryTweets['USA'], 'USA')
        BOW_PLOTS['ALL'] = create_df_BOW_All(All_tweets)
        HIST_PLOTS['POLARITY']=generate_histogram(sentiment,0,"Polarity")
        HIST_PLOTS['SUBJECTIVITY'] = generate_histogram(sentiment, 1, "Subjectivity")
        HIST_PLOTS['GROUPED']=generate_bar_plots(sentiment)
        WC_UK=save_WordClouds(UK_full_sentence, 'UK')
        WC_AUS=save_WordClouds(Australia_full_sentence, 'AUS')
        WC_USA=save_WordClouds(USA_full_sentence, 'USA')
        WC_ALL=save_WordClouds(All_Country_sentence, 'ALL')
        WC_FILES['UK']='/static/'+WC_UK
        WC_FILES['AUS'] = '/static/'+WC_AUS
        WC_FILES['USA'] = '/static/'+WC_USA
        WC_FILES['ALL'] = '/static/'+WC_ALL
        df_data=[[i for i in range(1,16)],
                    ['UK']*5 +['AUS']*5 +['USA']*5,
                    countryTweets['UK'][:5]+countryTweets['Australia'][:5]+countryTweets['USA'][:5]]
        return render_template("tweets_df.html", num_tweets=len(cleaned_tweets), tweet_data=df_data)


WC_FILES=dict()
PIE_FILES=dict()
HIST_PLOTS=dict()
BOW_PLOTS=dict()

@app.route('/wc', methods =['GET'])
def wc_endpoint():
        if len(WC_FILES)==4:
            return render_template('wc.html', wc_uk=WC_FILES['UK'], wc_aus=WC_FILES['AUS'],
                               wc_usa=WC_FILES['USA'], wc_all=WC_FILES['ALL'])
        else:
            return "WordClouds arent ready yet"
        #TODO we can render an error message that returns us to homepage

@app.route('/pie',methods=['GET'])
def pie_charts():
    if len(PIE_FILES) != 0:
        return render_template('pie.html', pie_uk=PIE_FILES['UK'], pie_aus=PIE_FILES['Australia'],
                               pie_usa=PIE_FILES['USA'])
    else:
        return "Pie Charts arent ready yet"

@app.route('/sentiment',methods=['GET'])
def plots_1D():
    if len(HIST_PLOTS) == 3:
        return render_template('hist.html', hist_polarity=HIST_PLOTS['POLARITY'], hist_subjectivity=HIST_PLOTS['SUBJECTIVITY'],
                               hist_grouped=HIST_PLOTS['GROUPED'])
    else:
        return "Histograms arent ready yet"

@app.route('/bow',methods=['GET'])
def plots_BOW():
    if len(BOW_PLOTS) == 4:
        df=BOW_PLOTS['ALL']
        return render_template('BOW.html', bow_uk=BOW_PLOTS['UK'], bow_aus=BOW_PLOTS['Australia'],
                               bow_usa=BOW_PLOTS['USA'], tables=[df.to_html(classes='data')], titles=df.columns.values)


    else:
        return "Plots arent ready yet"

app.run(host='0.0.0.0', port=5001)



