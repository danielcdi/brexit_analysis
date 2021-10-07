from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import json
import preprocessing
from pandas.io.json import json_normalize
from sklearn.externals import joblib
import csv
import pandas as pd





#consumer key, consumer secret, access token, access secret.
ckey="key"
csecret="secret"
atoken="token"
asecret="secret"

newtfidf = joblib.load('models/tfidf.pkl')
lr = joblib.load('models/lr-clf.pkl')


SEP = ';'
csv = open('OutputStreaming.csv','a')
csv.write('tweet' + SEP + 'tidy' + SEP + 'sentiment\n')

class listener(StreamListener):

    def on_data(self, data):

        all_data = json.loads(data)

        tweet = all_data["text"]

        username = all_data["user"]["screen_name"]

        date =  all_data["created_at"]

        tidy_tweet = preprocessing.clean_tweets(tweet)

        input = [tidy_tweet]

        tfidf_test = newtfidf.transform(input)

        y_pred = lr.predict(tfidf_test)

        if y_pred == 4 :
            x = 'positive'
        else:
            x = 'negative'
        csv.write(tweet + SEP + tidy_tweet + SEP + x + '\n')
        print((username,tweet,date,tidy_tweet,y_pred, x))

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["brexit"])