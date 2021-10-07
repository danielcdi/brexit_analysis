import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import joblib
from time import time
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV




#read csv
tweets = pd.read_csv('tweets16.csv', encoding = "ISO-8859-1", header=None)
#print(tweets.head())
#tweets = tweets.head(100)


#remove unncesesary columns and rename them

cols = [1,2,3,4]
tweets.drop(tweets.columns[cols],axis=1,inplace=True)
tweets.columns = ["sentiment", "tweet"]
#tweets = tweets.rename(columns={'0': 'sentiment', "@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D": 'tweet'})

print('a incarcat datele')
#sns.countplot(x= 'sentiment',data = tweets)
#plt.show()


tweets['tidy_tweet'] = np.vectorize(preprocessing.clean_tweets)(tweets['tweet'])


print('a curatat datele')
#PANA AICI E TOTUL BINE

# splitting data into training and validation set
xtrain_tfidf, xvalid_tfidf, ytrain, yvalid = train_test_split(tweets['tidy_tweet'], tweets['sentiment'], random_state=42, test_size=0.3)
print('a impartit datele')
#tf-idf

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# TF-IDF feature matrix
tfidf_train = tfidf_vectorizer.fit_transform(xtrain_tfidf)
#print(tfidf_vectorizer.get_feature_names())

print('a trecut tfidf')
print(tfidf_train)
#save tfidf
with open('tfidf.pkl', 'wb') as fin:
    pickle.dump(tfidf_vectorizer, fin)


#native-bayes
print("Naive Bayes")
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
plt.savefig('naive-bayes-confusion-matrix.png')
plt.show()

# Get predicted probabilities
y_score1 = naive_bayes_classifier.predict_proba(tfidf_test)[:,1]
# Plot Receiving Operating Characteristic Curve
# Create true and false positive rates
false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(yvalid, y_score1, pos_label = 4)
print('roc_auc_score for naive bayes: ', roc_auc_score(yvalid, y_score1))

# Plot ROC curves
plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - Naive Bayes')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('naive-bayes-roc.png')
plt.show()

# save the model to disk
filename = 'nb-clf.pkl'
pickle.dump(naive_bayes_classifier, open(filename, 'wb'))

print('------------------------------')

#svm linear
print("SVM:")
t = time()
svm = LinearSVC(random_state=0, tol=1e-5)
clf = CalibratedClassifierCV(svm)
clf.fit(tfidf_train, ytrain)
y_score1 = clf.predict_proba(tfidf_test)[:,1]

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
plt.savefig('svm-confusion-matrix.png')
plt.show()

false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(yvalid, y_score1, pos_label = 4)
print('roc_auc_score for svm: ', roc_auc_score(yvalid, y_score1))

# Plot ROC curves
plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic -SVM')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('svm-roc.png')
plt.show()


# save the model to disk
filename = 'svm-clf.pkl'
pickle.dump(clf, open(filename, 'wb'))


print('------------------------------')

#logistic regression
print("Logistic Regression:")
t = time()
clf_lr = LogisticRegression(random_state=0).fit(tfidf_train, ytrain)

training_time = time() - t
print("train time: %0.3fs" % training_time)

# predict the new document from the testing dataset
t = time()
tfidf_test = tfidf_vectorizer.transform(xvalid_tfidf)
y_pred = clf_lr.predict(tfidf_test)

test_time = time() - t
print("test time:  %0.3fs" % test_time)

# compute the performance measures
score1 = metrics.accuracy_score(yvalid, y_pred)
print("accuracy:   %0.3f" % score1)

print(metrics.classification_report(yvalid, y_pred, target_names=['Positive', 'Negative']))

print("confusion matrix:")
print(metrics.confusion_matrix(yvalid, y_pred))

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
plt.savefig('lr-confusion-matrix.png')
plt.show()

# Get predicted probabilities
y_score1 = clf_lr.predict_proba(tfidf_test)[:,1]
# Plot Receiving Operating Characteristic Curve
# Create true and false positive rates
false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(yvalid, y_score1, pos_label = 4)
print('roc_auc_score for lr: ', roc_auc_score(yvalid, y_score1))

# Plot ROC curves
plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - Logistic Regression')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('lr-roc.png')
plt.show()


# save the model to disk
filename = 'lr-clf.pkl'
pickle.dump(clf_lr, open(filename, 'wb'))


print('------------------------------')

#knn
print('kNN:')
t = time()
knn = neighbors.KNeighborsClassifier()
knn.fit(tfidf_train, ytrain)

training_time = time() - t
print("train time: %0.3fs" % training_time)

# predict the new document from the testing dataset
t = time()
tfidf_test = tfidf_vectorizer.transform(xvalid_tfidf)
y_pred = knn.predict(tfidf_test)

test_time = time() - t
print("test time:  %0.3fs" % test_time)

# compute the performance measures
score1 = metrics.accuracy_score(yvalid, y_pred)
print("accuracy:   %0.3f" % score1)

print(metrics.classification_report(yvalid, y_pred, target_names=['Positive', 'Negative']))

print("confusion matrix:")
print(metrics.confusion_matrix(yvalid, y_pred))

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
plt.savefig('knn-confusion-matrix.png')
plt.show()

# Get predicted probabilities
y_score1 = knn.predict_proba(tfidf_test)[:,1]
# Create true and false positive rates
false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(yvalid, y_score1, pos_label = 4)
print('roc_auc_score for knn: ', roc_auc_score(yvalid, y_score1))

# Plot ROC curves
plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - KNN')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('knn-roc.png')
plt.show()


# save the model to disk
filename = 'knn-clf.pkl'
pickle.dump(knn, open(filename, 'wb'))


print('------------------------------')

#mlp
print('MLP:')
t = time()
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
mlp.fit(tfidf_train, ytrain)


training_time = time() - t
print("train time: %0.3fs" % training_time)

# predict the new document from the testing dataset
t = time()
tfidf_test = tfidf_vectorizer.transform(xvalid_tfidf)
y_pred = mlp.predict(tfidf_test)

test_time = time() - t
print("test time:  %0.3fs" % test_time)

# compute the performance measures
score1 = metrics.accuracy_score(yvalid, y_pred)
print("accuracy:   %0.3f" % score1)

print(metrics.classification_report(yvalid, y_pred, target_names=['Positive', 'Negative']))

print("confusion matrix:")
print(metrics.confusion_matrix(yvalid, y_pred))

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
plt.savefig('mlp-confusion-matrix.png')
plt.show()

y_score1 = mlp.predict_proba(tfidf_test)[:,1]
false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(yvalid, y_score1, pos_label = 4)
print('roc_auc_score for mlp: ', roc_auc_score(yvalid, y_score1))

# Plot ROC curves
plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - MLP')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('mlp-roc.png')
plt.show()


# save the model to disk
filename = 'mlp-clf.pkl'
pickle.dump(knn, open(filename, 'wb'))


print('------------------------------')