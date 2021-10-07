import re
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from sklearn import metrics
from sklearn.metrics import classification_report
import string
from sklearn.svm import LinearSVC
import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from datetime import date
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)


#read csv
tweets = pd.read_csv('tweets16.csv', encoding = "ISO-8859-1")

#remove unncesesary columns and rename them

cols = [1,2,3,4]
tweets.drop(tweets.columns[cols],axis=1,inplace=True)
tweets = tweets.rename(columns={'0': 'sentiment', "@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D": 'tweet'})

#work only with 100k tweets remove later!



#remove patterns

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)

    return input_txt

# remove twitter handles (@user)
tweets['tidy_tweet'] = np.vectorize(remove_pattern)(tweets['tweet'], "@[\w]*")

# remove special characters, numbers, punctuations
tweets['tidy_tweet'] = tweets['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")

# remove short words
tweets['tidy_tweet'] = tweets['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))


#tokenization
tokenized_tweet = tweets['tidy_tweet'].apply(lambda x: x.split())
tokenized_tweet.head()
print(tokenized_tweet.head(20))

#stemming
from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
tokenized_tweet.head()

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

tweets['tidy_tweet'] = tokenized_tweet


# splitting data into training and validation set
xtrain_tfidf, xvalid_tfidf, ytrain, yvalid = train_test_split(tweets['tidy_tweet'], tweets['sentiment'], random_state=42, test_size=0.3)


#tf-idf

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# TF-IDF feature matrix
tfidf_train = tfidf_vectorizer.fit_transform(xtrain_tfidf)
#print(tfidf_vectorizer.get_feature_names())



#native-bayes
t = time()
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(tfidf_train, ytrain)

training_time = time() - t
print("train time: %0.3fs" % training_time)

# predict the new document from the testing dataset
t = time()
tfidf_test = tfidf_vectorizer.transform(xvalid_tfidf)
y_pred = naive_bayes_classifier.predict(tfidf_test)

test_time = time() - t
print("test time:  %0.3fs" % test_time)

# compute the performance measures
score1 = metrics.accuracy_score(yvalid, y_pred)
print("accuracy:   %0.3f" % score1)

print(metrics.classification_report(yvalid, y_pred, target_names=['Positive', 'Negative']))

print("confusion matrix:")

print(metrics.confusion_matrix(yvalid, y_pred))
cf_matrix = metrics.confusion_matrix(yvalid,y_pred)
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in
                cf_matrix.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
plt.savefig('output.png')
plt.show()

# load libraries
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Create feature matrix and target vector
#X, y = make_classification(n_samples=10000, n_features=100, n_classes=2)

# Split into training and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Create classifier
#clf1 = DecisionTreeClassifier(); clf2 = LogisticRegression();

# Train model
#clf1.fit(X_train, y_train); clf2.fit(X_train, y_train);

# Get predicted probabilities
y_score1 = naive_bayes_classifier.predict_proba(tfidf_test)[:,1]


# Plot Receiving Operating Characteristic Curve
# Create true and false positive rates
false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(yvalid, y_score1, pos_label = 4)
print('roc_auc_score for naive bayes: ', roc_auc_score(yvalid, y_score1))

# Plot ROC curves
plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - DecisionTree')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()







print('------------------------------')

#svm linear
#clf = SVC(probability=True, kernel='rbf')
#clf.fit(xtrain_tfidf, ytrain)



t = time()
clf = LinearSVC(random_state=0, tol=1e-5)
clf.fit(tfidf_train, ytrain)

training_time = time() - t
print("train time: %0.3fs" % training_time)

# predict the new document from the testing dataset
t = time()
tfidf_test = tfidf_vectorizer.transform(xvalid_tfidf)
y_pred = clf.predict(tfidf_test)

test_time = time() - t
print("test time:  %0.3fs" % test_time)

# compute the performance measures
score1 = metrics.accuracy_score(yvalid, y_pred)
print("accuracy:   %0.3f" % score1)

print(metrics.classification_report(yvalid, y_pred, target_names=['Positive', 'Negative']))

print("confusion matrix:")
print(metrics.confusion_matrix(yvalid, y_pred))


print('------------------------------')

#knn

t = time()
clf = neighbors.KNeighborsClassifier()
clf.fit(tfidf_train, ytrain)

training_time = time() - t
print("train time: %0.3fs" % training_time)

# predict the new document from the testing dataset
t = time()
tfidf_test = tfidf_vectorizer.transform(xvalid_tfidf)
y_pred = clf.predict(tfidf_test)

test_time = time() - t
print("test time:  %0.3fs" % test_time)

# compute the performance measures
score1 = metrics.accuracy_score(yvalid, y_pred)
print("accuracy:   %0.3f" % score1)

print(metrics.classification_report(yvalid, y_pred, target_names=['Positive', 'Negative']))

print("confusion matrix:")
print(metrics.confusion_matrix(yvalid, y_pred))


print('------------------------------')

