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
from QuestionClassifier import QuestionClassify
from sklearn.model_selection import train_test_split



'''
The following function used the QuestionClassify class to train on the given dataset. Some initial
preprocessing of the file is done here, a label_dict is given to map the text labels to numerical values.
A MaxEnt BOW classifier has been used. The models are saved during training in the models folder to be reused later,
and to save time by not training repeatedly.
'''
def train(filename,split=0):
    with open(filename,"r") as text:
        lines=text.readlines()
    lines=[line.replace("\n","") for line in lines]
    lines=[line.split(",,,") for line in lines]
    questions=[line[0] for line in lines]
    question_type=[line[1] for line in lines]
    question_type=[q.strip() for q in question_type]
    questions=[q.strip() for q in questions]
    lines=[[q,w] for (q,w) in zip(questions,question_type)]
    data=pd.DataFrame(lines,columns=["Question","Question_Type"])
    label_dict = {'who': 1, "what": 2, "when": 3, "affirmation": 4, "unknown": 5}

    model = QuestionClassify(label_dict)
    if(split==1):
        Xtrain, Xtest, ytrain, ytest = train_test_split(data["Question"].values, data["Question_Type"].values)
        model.train(Xtrain,ytrain,label_dict)
        print "Training Accuracy",model.test(Xtrain,ytrain,label_dict)
        print "Test Accuracy", model.test(Xtest, ytest,label_dict)
    else:
        model.train(data["Question"].values,data["Question_Type"].values,label_dict)
        print "Training Accuracy", model.test(data["Question"].values,data["Question_Type"].values,label_dict)
        print "No Test Partition"



'''
The following function just runs the trained model on 200 questions sampled from the cornell corpus and reports the
result in a csv file
'''
def test_(filename):
    label_dict = {'who': 1, "what": 2, "when": 3, "affirmation": 4, "unknown": 5}
    with open(filename, 'r') as filename:
        lines = filename.readlines()
    test_labels = []
    test_sents = []
    for line in lines:
        line = line.lstrip()
        line = line.rstrip()
        line = line.replace("\n", "")
        words = line.split()

        test_labels.append(words[0])
        test_sents.append(" ".join(words[1:]))

    # Now lets randomly sample 200 sentences from the dataset:
    test_indices = np.random.permutation(len(test_sents))[:200]
    sents=[test_sents[ind] for ind in test_indices]
    labs=[test_labels[ind] for ind in test_indices]
    model=QuestionClassify()
    predictions=model.predict(sents,label_dict)

    sents=np.array([sents]).transpose()

    labs=np.array([labs]).transpose()

    predictions=np.array([predictions]).transpose()

    all_together=np.hstack((sents,labs,predictions))
    df=pd.DataFrame(all_together,columns=["Question","Given Label","Predicted Label"])

    df.to_csv("test_file.csv",index=None)

    print "File saved in the same folder."



if __name__=='__main__':
    '''
    Train the model on the given dataset. 
    Split= 0 : Do not split the data into training and testing. Trains the model on the entire dataset, returns
                training accuracy.
    Split=1: Splits the data by 80/20 ration. Trains on the 80 %, returns the accuracy on the testing part.
    
    Comment out this part after using it once."
    '''

    filename1 = "LabelledData.txt"
    train(filename1,split=1)

    '''
    Test the model on 200 questions sampled from http://cogcomp.cs.illinois.edu/Data/QA/QC/train_1000.label and saves 
    thee predictions in test_file.csv, the labels will not match the ones in the dataset as a adifferent set of labels
    is used in this corpus.
    Comment out this part after using it once.
    '''
    filename2= "train_1000.label.txt"
    test_(filename2)


















