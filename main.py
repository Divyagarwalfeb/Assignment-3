import nltk
import sklearn_crfsuite
import eli5
import numpy as np 
import pandas as pd 
import nltk 
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn_crfsuite import metrics
import scipy.stats
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
from gensim.models import Word2Vec
from collections import Counter

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
    return features

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))
        
def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))
        
def sent2tokens(sent):
    return [token for token, postag, label in sent]

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

train_sent = []
list1 = [] 
list2 = [] 
sent = ""
label = [] 

with open('ner.txt',encoding = 'latin-1') as fp:
	arr = fp.readlines()
    for entry in arr:
        if entry =="\n":
			token_sent = sent.split()
			sent_pos = nltk.pos_tag(token_sent)
            sent_pos = [l + (''.join(label[i]),) for i,l in enumerate(sent_pos)]
			train_sent.append([sent_pos])
			sent = ""
			label = []  
			continue 
        a1, a2 = entry.split()
		sent = sent +" "+a1 
		label.append(a2)
train_sent = [i[0] for i in train_sent]
check_random_state(100)
train_sents, test_sents = train_test_split(train_sent,test_size=0.2)
X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]
X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c2=0.02214521702680784,
    c1=0.21247664779649446,
    max_iterations=70,
    all_possible_transitions=False,
)
 'c2': scipy.stats.expon(scale=0.05),

labels = ['D','O','T']
crf.fit(X_train, y_train);
y_pred = crf.predict(X_test)
y_pred = [i[0] for i in y_pred]
y_test = [i[0] for i in y_test]

print(metrics.flat_classification_report(y_test, y_pred, labels=labels))
print("Top likely transitions:")
print_transitions(Counter(crf.transition_features_).most_common(3))
print("\nTop unlikely transitions:")
print_transitions(Counter(crf.transition_features_).most_common()[-3:])
print("Top positive:")
print_state_features(Counter(crf.state_features_).most_common(30))
print("\nTop negative:")
print_state_features(Counter(crf.state_features_).most_common()[-30:

