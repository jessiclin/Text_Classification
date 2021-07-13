import sys 
import numpy as np
from sklearn import datasets 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from nltk.stem.porter import *
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.feature_extraction import text
from sklearn.feature_selection import RFE

def naive_bayes(train_corpus, Y_train, test_corpus, Y_test): 
    stemmer = PorterStemmer()
    for i in range(len(train_corpus)): 
        train_corpus[i] = stemmer.stem(train_corpus[i]) 
    for i in range(len(test_corpus)): 
        test_corpus[i] = stemmer.stem(test_corpus[i])
    
        
    vectorizer = CountVectorizer(ngram_range=(1, 1), stop_words=text.ENGLISH_STOP_WORDS)
    
    X_train = vectorizer.fit_transform(train_corpus).toarray()
    X_test = vectorizer.transform(test_corpus).toarray()
    
    selector = SelectKBest(chi2, k=X_train.shape[1]//3)
    X_train = selector.fit_transform(X_train, Y_train)
    X_test = selector.transform(X_test)   
    clf = MultinomialNB(alpha=0.1)
    clf.fit(X_train, Y_train) 
    Y_pred = clf.predict(X_test)
    return precision_recall_fscore_support(Y_test, Y_pred, average='macro')

def logistic_regression(train_corpus, Y_train, test_corpus, Y_test):  
    stemmer = PorterStemmer()
    for i in range(len(train_corpus)): 
        train_corpus[i] = stemmer.stem(train_corpus[i]) 
    for i in range(len(test_corpus)): 
        test_corpus[i] = stemmer.stem(test_corpus[i])

    vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words=text.ENGLISH_STOP_WORDS)
    X_train = vectorizer.fit_transform(train_corpus)
    X_test = vectorizer.transform(test_corpus)
    
    clf = LogisticRegression(penalty='none', max_iter=300, fit_intercept=False, solver='newton-cg') 
    clf.fit(X_train, Y_train) 
    Y_pred = clf.predict(X_test)
    return precision_recall_fscore_support(Y_test, Y_pred, average='macro')
    
def svm(train_corpus, Y_train, test_corpus, Y_test):  
    stemmer = PorterStemmer()
    for i in range(len(train_corpus)): 
        train_corpus[i] = stemmer.stem(train_corpus[i]) 
    for i in range(len(test_corpus)): 
        test_corpus[i] = stemmer.stem(test_corpus[i])

    vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    X_train = vectorizer.fit_transform(train_corpus)
    X_test = vectorizer.transform(test_corpus)
    selector = SelectKBest(chi2, k=X_train.shape[1]//2)
    X_train = selector.fit_transform(X_train, Y_train)
    X_test = selector.transform(X_test)
    
    clf = SVC(kernel='linear')
    clf.fit(X_train, Y_train) 
    Y_pred = clf.predict(X_test)
    return precision_recall_fscore_support(Y_test, Y_pred, average='macro')


def random_forest(train_corpus, Y_train, test_corpus, Y_test):  
    stemmer = PorterStemmer()
    for i in range(len(train_corpus)): 
        train_corpus[i] = stemmer.stem(train_corpus[i]) 
    for i in range(len(test_corpus)): 
        test_corpus[i] = stemmer.stem(test_corpus[i])
        
    vectorizer = CountVectorizer(ngram_range=(1, 1), stop_words=text.ENGLISH_STOP_WORDS)

    X_train = vectorizer.fit_transform(train_corpus)
    X_test = vectorizer.transform(test_corpus)
    selector = SelectKBest(chi2, k=X_train.shape[1]//3)
    X_train = selector.fit_transform(X_train, Y_train)
    X_test = selector.transform(X_test)
    clf = RandomForestClassifier(n_estimators=150, max_features='log2')
    clf.fit(X_train, Y_train) 
    Y_pred = clf.predict(X_test)
    return precision_recall_fscore_support(Y_test, Y_pred, average='macro')  
    
def algo(baseline, train_corpus, Y_train, test_corpus, Y_test, learning_algo): 
    
    if learning_algo == 'naive_bayes': 
        return naive_bayes(baseline, train_corpus, Y_train, test_corpus, Y_test)
    elif learning_algo == 'logistic_regression': 
        return logistic_regression(baseline, train_corpus, Y_train, test_corpus, Y_test)
    elif learning_algo == 'svm': 
        return svm(baseline, train_corpus, Y_train, test_corpus, Y_test)
    else: 
        return random_forest(baseline, train_corpus, Y_train, test_corpus, Y_test)

if __name__ == "__main__": 
    args = sys.argv 
    print(args)
    if len(args) != 4: 
        print("Invalid number of args") 
        exit() 
    
    train_folder = args[1] 
    eval_folder = args[2] 
    output = args[3] 
    
    #sys.stdout = open(output, 'w')
    train_data = datasets.load_files(train_folder, encoding='latin1')
    test_data = datasets.load_files(eval_folder, encoding='latin1')
    
    train_corpus = np.array(train_data.data)
    test_corpus = np.array(test_data.data)
    
    Y_train = np.array(train_data.target)
    Y_test = np.array(test_data.target)
    

    with open(output, 'w') as f: 
        nb = naive_bayes(train_corpus, Y_train, test_corpus, Y_test)
        lr = logistic_regression(train_corpus, Y_train, test_corpus, Y_test)
        svc = svm(train_corpus, Y_train, test_corpus, Y_test)
        rf = random_forest(train_corpus, Y_train, test_corpus, Y_test)
        f.write("NB,CV+UB," + str(nb[0]) + ',' + str(nb[1]) + ',' + str(nb[2]) + '\n')
        f.write("LR,TFIDF+UB," + str(lr[0]) + ',' + str(lr[1]) + ',' + str(lr[2]) + '\n')
        f.write("SVM,TFIDF+UB," + str(svc[0]) + ',' + str(svc[1]) + ',' + str(svc[2]) + '\n')
        f.write("RF,CV+UB," + str(rf[0]) + ',' + str(rf[1]) + ',' + str(rf[2]) + '\n')
        
