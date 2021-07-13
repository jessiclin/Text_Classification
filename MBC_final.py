import MBC_exploration as MBC 
import sys 
import numpy as np
from sklearn import datasets 
if __name__ == "__main__": 
    args = sys.argv 
    print(args)
    if len(args) != 4: 
        print("Invalid number of args") 
        exit() 
    
    train_folder = args[1] 
    eval_folder = args[2] 
    output = args[3] 
    
    train_data = datasets.load_files(train_folder, encoding='latin1')
    test_data = datasets.load_files(eval_folder, encoding='latin1')
    
    train_corpus = np.array(train_data.data)
    test_corpus = np.array(test_data.data)
    
    Y_train = np.array(train_data.target)
    Y_test = np.array(test_data.target)
    
    best = MBC.naive_bayes(train_corpus, Y_train, test_corpus, Y_test)
    
    with open(output, 'w') as f: 
        f.write("NB,CV+UB," + str(best[0]) + ',' + str(best[1]) + ',' + str(best[2]) + '\n')