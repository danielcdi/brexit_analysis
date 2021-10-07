import os
import pandas as pd
import numpy as np
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
import nltk
#import ssl
#
#try:
#    _create_unverified_https_context = ssl._create_unverified_context
#except AttributeError:
#    pass
#else:
#    ssl._create_default_https_context = _create_unverified_https_context

#nltk.download()
from gensim import corpora, models, similarities
import logging
import tempfile
from nltk.corpus import stopwords
from string import punctuation
from collections import OrderedDict
import seaborn as sns
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import preprocessing
import nltk
import pyLDAvis


import warnings
warnings.filterwarnings("ignore")



#twitterscraper brexit -l 1000 -bd 2016-06-16 -ed 2016-06-40 -o referendum.json

#read data from json
tweets = pd.read_json (r'referendum/referendum.json')

#clean df
cols = [0,2,3,4,6,7,11,13,16,17,18,19,20,15]
tweets.drop(tweets.columns[cols],axis=1,inplace=True)


tweets['tidy_tweet'] = np.vectorize(preprocessing.clean_tweets)(tweets['text'])


tweets['tokenized_sents'] = tweets.apply(lambda row: nltk.word_tokenize(row['tidy_tweet']), axis=1)

#word cloud




#tfidf

#tfidf = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# TF-IDF feature matrix
#tfidf_train = tfidf.fit_transform(tweets['tidy_tweet'])


#corpus

dictionary = corpora.Dictionary(tweets['tokenized_sents'])
corpus = [dictionary.doc2bow(text) for text in tweets['tokenized_sents']]

tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model

corpus_tfidf = tfidf[corpus]  # step 2 -- use the model to transform vectors

total_topics = 5

lda = models.LdaModel(corpus, id2word=dictionary, num_topics=total_topics)
corpus_lda = lda[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

print(lda.show_topics(total_topics,5))


data_lda = {i: OrderedDict(lda.show_topic(i,5)) for i in range(total_topics)}
df_lda = pd.DataFrame(data_lda)
df_lda = df_lda.fillna(0).T
print(df_lda.shape)
#df_lda.to_csv('topics_covid.csv')
print('aici')
print(lda.num_topics)

g=sns.clustermap(df_lda.corr(), center=0, standard_scale=1, cmap="RdBu", metric='cosine', linewidths=.75, figsize=(15, 15))
plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.savefig('topics_heatmap_referendum.png')
plt.show()

for idx, topic in lda.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


import pyLDAvis
import pyLDAvis.gensim
visualisation = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
pyLDAvis.save_html(visualisation, 'LDA_Visualization_Referendum.html')

#import matplotlib.pyplot as plt
