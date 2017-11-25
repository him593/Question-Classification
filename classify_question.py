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
Custom Predict function to 
'''
def predict(question):
    label_dict = {'who': 1, "what": 2, "when": 3, "affirmation": 4, "unknown": 5}
    model=QuestionClassify()


    return model.predict(question,label_dict)



if __name__=='__main__':
    while (1):
        try:
            sentence = raw_input("Enter the question: ")
            print predict(sentence)
        except EOFError:
            print "End of input"
            break

