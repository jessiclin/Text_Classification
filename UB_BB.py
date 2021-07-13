import sys 
import numpy as np
from sklearn import datasets 
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_recall_fscore_support

import matplotlib.pyplot as plt 


def naive_bayes(X_train, Y_train, X_test, Y_test): 
    clf = MultinomialNB()
    clf.fit(X_train, Y_train) 
    Y_pred = clf.predict(X_test)
    return precision_recall_fscore_support(Y_test, Y_pred, average='macro')

def logistic_regression(X_train, Y_train, X_test, Y_test):  
    clf = LogisticRegression(max_iter=300) 
    clf.fit(X_train, Y_train) 
    Y_pred = clf.predict(X_test)
    return precision_recall_fscore_support(Y_test, Y_pred, average='macro')
    
def svm(X_train, Y_train, X_test, Y_test):  
    clf = SVC()
    clf.fit(X_train, Y_train) 
    Y_pred = clf.predict(X_test)
    return precision_recall_fscore_support(Y_test, Y_pred, average='macro')

def random_forest(X_train, Y_train, X_test, Y_test):  
    clf = RandomForestClassifier()
    clf.fit(X_train, Y_train) 
    Y_pred = clf.predict(X_test)
    return precision_recall_fscore_support(Y_test, Y_pred, average='macro')  
    
def algo(baseline, train_corpus, Y_train, test_corpus, Y_test): 
    
    if baseline == 0: 
        vectorizer = CountVectorizer(ngram_range=(1, 1))
    else: 
        vectorizer = CountVectorizer(ngram_range=(2, 2))
        
    X_train = vectorizer.fit_transform(train_corpus)
    X_test = vectorizer.transform(test_corpus)
    
    output = []
    ### Naive Bayes 
    output.append(naive_bayes(X_train.toarray(), Y_train, X_test.toarray(), Y_test))
    
    ### LogisticRegression 
    output.append(logistic_regression(X_train, Y_train, X_test, Y_test))
    
    ### SVM 
    output.append(svm(X_train, Y_train, X_test, Y_test))
    

    ### Random Forest 
    output.append(random_forest(X_train, Y_train, X_test, Y_test))

    return output 
    
if __name__ == "__main__": 
    args = sys.argv 
 
    if len(args) != 5: 
        print("Invalid number of args") 
        exit() 
    
    train_folder = args[1] 
    eval_folder = args[2] 
    output = args[3] 
    display_LC = int(args[4]) 
    
    #sys.stdout = open(output, 'w')
    train_data = datasets.load_files(train_folder, encoding='latin1')
    test_data = datasets.load_files(eval_folder, encoding='latin1')
    
    train_corpus = np.array(train_data.data)
    test_corpus = np.array(test_data.data)
    Y_train = np.array(train_data.target)
    Y_test = np.array(test_data.target)
    
    unigram = algo(0, train_corpus, Y_train, test_corpus, Y_test)
    bigram = algo(1, train_corpus, Y_train, test_corpus, Y_test)
    with open(output, 'w') as f: 
        f.write("NB,UB," + str(unigram[0][0]) + "," + str(unigram[0][1]) + "," + str(unigram[0][2])+'\n')
        f.write("NB,BB," + str(bigram[0][0]) + "," + str(bigram[0][1]) + "," + str(bigram[0][2])+'\n')
        f.write("LR,UB," + str(unigram[1][0]) + "," + str(unigram[1][1]) + "," + str(unigram[1][2])+'\n')
        f.write("LR,BB," + str(bigram[1][0]) + "," + str(bigram[1][1]) + "," + str(bigram[1][2])+'\n')
        f.write("SVM,UB," + str(unigram[2][0]) + "," + str(unigram[2][1]) + "," + str(unigram[2][2])+'\n')
        f.write("SVM,BB," + str(bigram[2][0]) + "," + str(bigram[2][1]) + "," + str(bigram[2][2])+'\n')
        f.write("RF,UB," + str(unigram[3][0]) + "," + str(unigram[3][1]) + "," + str(unigram[3][2])+'\n')
        f.write("RF,BB," + str(bigram[3][0]) + "," + str(bigram[3][1]) + "," + str(bigram[3][2])+'\n')
        
        

    if display_LC == 1: 
        nb = [] 
        lr = [] 
        svc = [] 
        rf = [] 
        x_coord = [] 
        ten = len(train_corpus)//10 
        i = ten 
        # 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%
        while i <= len(train_corpus): 
            idx = np.random.choice(np.arange(len(train_corpus)), i, replace=False)
            x_sample = train_corpus[idx]
            y_sample = Y_train[idx]
            unigram = algo(0, x_sample, y_sample, test_corpus, Y_test)
            nb.append(unigram[0][2])
            lr.append(unigram[1][2])
            svc.append(unigram[2][2])
            rf.append(unigram[3][2])
            x_coord.append(i)
            print(i, unigram[0][2], unigram[1][2], unigram[2][2], unigram[3][2])
            i += ten
        plt.plot(x_coord, nb, '.b-', label="NB")
        plt.plot(x_coord,lr, '.r-', label="LR")
        plt.plot(x_coord,svc, '.g-', label="SVM")
        plt.plot(x_coord,rf,'.c-', label="RF")
        plt.xlabel('Size of training data (%)')
        plt.legend(loc="lower right")
        plt.ylabel('F1-score')
        plt.grid() 
        plt.show()
    
    
