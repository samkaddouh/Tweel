from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pickle
from joblib import dump, load
from sklearn.naive_bayes import BernoulliNB
import pickle
import time
from scipy import sparse

class Modeller:
    def __init__(self, cleaned_csv_file, is_train = True, use_pickle=False):
        self.clf = None
        self.X_test = None
        if is_train:
            dataset = pd.read_csv(cleaned_csv_file)
            X = dataset.text
            y = dataset.target

            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=26105111)
            
            #X_train = vectoriser.transform(X_train)
            #X_test = vectoriser.transform(X_test)
            
            self.y_train = y
            if use_pickle:
                self.X_train = sparse.load_npz("_Xtrain.npz")
                with open('_modeller.pickle', 'rb') as handle:
                    self.vectoriser = pickle.load(handle)
                print("loaded pickled vectoriser and X_train")
            else:
                vectoriser = TfidfVectorizer(ngram_range=(1, 2), max_features=500000)
                vectoriser.fit(X)
                self.X_train = vectoriser.transform(X)
                self.vectoriser = vectoriser
                with open('_modeller.pickle', 'wb') as handle:
                    pickle.dump(vectoriser, handle, protocol=pickle.HIGHEST_PROTOCOL)

                sparse.save_npz("_Xtrain.npz", self.X_train)
                print("pickled vectoriser and transformed X_train")

            print('No. of feature_words: ', len(self.vectoriser.get_feature_names()))
    def set_classifier(self, clf):
        self.clf = clf
        start=time.time()
        self.clf.fit(self.X_train, self.y_train)
        end=time.time()
        print("model fit took", end-start, "s")
    def predict(self, test_dataset_text):
        '''expecting this to be a list of cleaned tokens for each test tweet'''
        if self.clf is None:
            print("Classifier is not set yet, Use set_classfier()")
        else:
            X_test = self.vectoriser.transform(test_dataset_text)
            return self.clf.predict(X_test)




def model_Evaluate(model, X_test, y_test):
    # Predict values for Test dataset
    y_pred = model.predict(X_test)
    # Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',xticklabels = categories, yticklabels = categories)
    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
    plt.show()
'''
BNBmodel = BernoulliNB()
BNBmodel.fit(X_train, y_train)
model_Evaluate(BNBmodel)
y_pred1 = BNBmodel.predict(X_test)

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
'''




'''
LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
LRmodel.fit(X_train, y_train)
model_Evaluate(LRmodel)
y_pred3 = LRmodel.predict(X_test)
dump(LRmodel, 'LR.joblib')
'''


