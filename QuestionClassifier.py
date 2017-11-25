import numpy as np
import pandas as pd
import random
import re
from pickle import load
from pickle import dump
import nltk
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from nltk.stem import SnowballStemmer


class QuestionClassify(object):
    def __init__(self,algorithm="LogisticRegression"):

        self.X=None
        self.y=None
        self.cv = None
        self.tf = None

        self.path = './models/'
        all_models = os.listdir(self.path)
        if(algorithm=="LogisticRegression" and 'model.pkl' in all_models):
            clf_file = open(self.path + 'model.pkl', 'rb')
            self.model = load(clf_file)
            clf_file.close()
        else:
            self.model=LogisticRegression(penalty='l2',C=10.0,multi_class="multinomial",solver="newton-cg")

        if('cv.pkl' in all_models):
            cv_file=open(self.path+'cv.pkl','rb')
            self.cv=load(cv_file)
            cv_file.close()
        if('tf.pkl' in all_models):
            tf_file=open(self.path+'tf.pkl','rb')
            self.tf=load(tf_file)
            tf_file.close()

    def data_processing(self):
        text=self.X
        puncts = [',', "'", '"']
        stemmer = SnowballStemmer("english")
        processed_text = []
        for line in text:
            line = line.lower()
            line = line.decode('utf-8', 'ignore')
            line = re.sub(r'[0-9]+', '', line)

            words = nltk.word_tokenize(line)
            words = [w for w in words if w not in puncts]
            words = [stemmer.stem(w) for w in words]
            processed_text.append(" ".join(words))

        self.X=processed_text

    def feature_extraction(self):
        if(self.cv==None and self.tf==None):
            self.cv = CountVectorizer()
            self.tf= TfidfTransformer()

            cv_data = self.cv.fit_transform(self.X)
            tf_data = self.tf.fit_transform(cv_data).toarray()
            tf_df = pd.DataFrame(tf_data, columns=self.cv.get_feature_names())
            cv_file=open(self.path+'cv.pkl','wb')
            dump(self.cv,cv_file,-1)
            cv_file.close()
            tf_file=open(self.path+'tf.pkl','wb')
            dump(self.tf,tf_file,-1)
            tf_file.close()

            cv_data=self.cv.fit_transform(self.X)
            self.X=self.tf.fit_transform(cv_data).toarray()

        else:
            cv_data=self.cv.transform(self.X)
            self.X = self.tf.transform(cv_data).toarray()

    def train(self,X,y,label_dict):
        self.X=X
        self.y=[label_dict[t] for t in y]

        self.data_processing()
        self.feature_extraction()

        self.model.fit(self.X,self.y)
        model_file=open(self.path+'model.pkl','wb')
        dump(self.model,model_file,-1)
        model_file.close()

        return self.model


    def test(self,X,y,label_dict):
        self.X=X
        self.y = [label_dict[t] for t in y]
        self.data_processing()
        self.feature_extraction()
        preds=self.model.predict(self.X)

        return sum(1.0 for p,y in zip(preds,self.y) if p==y)/float(len(self.X))

    def predict(self,X,label_dict):

        if type(X)==list:
            self.X=X
        else:
            self.X=[X]

        self.data_processing()
        self.feature_extraction()

        predictions=self.model.predict(self.X)


        labels=[]
        for pred in predictions:
            for key in label_dict.keys():
                if(label_dict[key]==pred):
                   labels.append(key)


        return labels




















