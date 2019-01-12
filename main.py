import nltk
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk as nlp
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def read_dataset_from():
    """
    read from dataset only the gender and description (remove the Nan lines)
    :return:
    return all data from csv (pandas)
    """
    data = pd.read_csv("input/gender-classifier-DFE-791531.csv", encoding="latin1")
    data = pd.concat([data.gender, data.description], axis=1)  # get from dataset only the gender and tweet description
    # let's drop NaN values
    data.dropna(axis=0, inplace=True)
    return data

def clean_and_modify_the_data(data):
    """

    :param data: pure data from dataset
    :return: description_list after modify and dataset object after filters
    """
    data = data[data.gender.isin(['female', 'male'])]
    data['gender'] = (data.gender == 'female').astype(int)
    description_list = []
    for description in data.description:  # remove from all description stop words and set lowercase
        description = re.sub("[^a-zA-Z]", " ", description)
        description = description.lower()
        description = nltk.word_tokenize(description)
        # description = [ word for word in description if not word in set(stopwords.words("english"))]
        lemma = nlp.WordNetLemmatizer()
        try:
            description = [lemma.lemmatize(word) for word in description]
        except Exception as e:
            if "NLTK Downloader" in e.args[0].split("\n")[3]:
                print("downloading the NLTK package " + e.args[0].split("\n")[6].split("'")[1])
                nltk.download(e.args[0].split("\n")[6].split("'")[1])
            description = [lemma.lemmatize(word) for word in description]  # change the releated words and gremmers
        description = " ".join(description)
        description_list.append(description)
    return data, description_list

def count_vectorizer_from_descritpion_list(description_list, max_features, stopwords):
    if stopwords is True:
        count_vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
    else:
        count_vectorizer = CountVectorizer(max_features=max_features)
    sparse_matrix = count_vectorizer.fit_transform(description_list).toarray()  # x
    print("{} most common words for CountVectorizer: {}".format(max_features, count_vectorizer.get_feature_names()))
    return sparse_matrix


def tfid_vectorizer_from_descritpion_list(description_list, max_features, stopwords):
    if stopwords is True:
        tfid_vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    else:
        tfid_vectorizer = TfidfVectorizer(max_features=max_features)
    sparse_matrix = tfid_vectorizer.fit_transform(description_list).toarray()  # x
    print("{} most common words for TfidfVectorizer: {}".format(max_features, tfid_vectorizer.get_feature_names()))
    return sparse_matrix


def add_feature_len_of_letters_in_tweet(sparse_matrix, description_list):
    letters_count = np.array([[len(desc) for desc in description_list]]).transpose()  # add length of letters in a tweet
    x = sparse_matrix  # full tweet
    x = np.concatenate([x, letters_count], axis=1)
    return x

def train_and_test_split_data(data, sparse_matrix, test_size):
    y = data.iloc[:, 0].values  # male or female classes (gender)
    x = sparse_matrix
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    return x_train, x_test, y_train, y_test


def train_gaussiannb_model(x_train, x_test, y_train, y_test):
    nb = GaussianNB()
    nb.fit(x_train, y_train)  # train to model by full tweet and gender
    print("using GaussianNB model : ")
    print("accuracy for train : ", nb.score(x_train, y_train))
    print("accuracy for test  : ", nb.score(x_test, y_test))
    return nb


def train_randomforestclassifier_model(x_train, x_test, y_train, y_test, number_of_trees):
    rf = RandomForestClassifier(n_estimators=number_of_trees)
    rf.fit(x_train, y_train)
    print("using RandomForestClassifier  : ")
    print("accuracy for train : ", rf.score(x_train, y_train))
    print("accuracy for test  : ", rf.score(x_test, y_test))
    return rf

#RandomForestClassifier model

def show_the_result_with_plt(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_true = y_test
    cm_nb = confusion_matrix(y_true,y_pred)
    sns.heatmap(cm_nb,annot=True,cmap="RdPu",fmt=".0f",cbar=False)
    plt.show()
    #plt.clf()


data = read_dataset_from()
data, description_list = clean_and_modify_the_data(data)
#create all vectors (sparse matrixs)
count_vector_stop_words = count_vectorizer_from_descritpion_list(description_list, 1000, True)
count_vector = count_vectorizer_from_descritpion_list(description_list, 900, False)
ftid_vector_stop_words = tfid_vectorizer_from_descritpion_list(description_list, 700, True)
ftid_vector = tfid_vectorizer_from_descritpion_list(description_list, 900, False)
#add the the same features for all matrixs
count_vector_stop_words_feature_added = add_feature_len_of_letters_in_tweet(count_vector_stop_words, description_list)
count_vector_feature_added = add_feature_len_of_letters_in_tweet(count_vector, description_list)
ftid_vector_stop_words_feature_added = add_feature_len_of_letters_in_tweet(ftid_vector_stop_words, description_list)
ftid_vector_feature_added = add_feature_len_of_letters_in_tweet(ftid_vector , description_list)
#split the data for train
"""
x_train, x_test, y_train, y_test = train_and_test_split_data(data, count_vector_stop_words_feature_added, 0.15)
train the model with the data
model_gaussiannb = train_gaussiannb_model(x_train, x_test, y_train, y_test)
model_randomforestclassifier = train_randomforestclassifier_model(x_train, x_test, y_train, y_test, 100)
show the result in plt
show_the_result_with_plt(model_gaussiannb, x_test, y_test)
pass
"""

#finish
sparse_matrix_list = []
sparse_matrix_description_list = []
sparse_matrix_list.append(count_vector_stop_words_feature_added)
sparse_matrix_description_list.append("Count vector with english stop words : ")
sparse_matrix_list.append(count_vector_feature_added)
sparse_matrix_description_list.append("Count vector without english stop words : ")
sparse_matrix_list.append(ftid_vector_stop_words_feature_added)
sparse_matrix_description_list.append("FT-IDF vector with english stop words : ")
sparse_matrix_list.append(ftid_vector_feature_added)
sparse_matrix_description_list.append("FT-IDF vector without english stop words : ")

print('\n')
print('\n')
print('\n')

index = 0
for matrix in sparse_matrix_list:
    x_train, x_test, y_train, y_test = train_and_test_split_data(data, matrix, 0.15)
    print (sparse_matrix_description_list[index])
    model_gaussiannb = train_gaussiannb_model(x_train, x_test, y_train, y_test)
    model_randomforestclassifier = train_randomforestclassifier_model(x_train, x_test, y_train, y_test, 100)
    print ('\n')
    index = index + 1
    show_the_result_with_plt(model_gaussiannb, x_test, y_test)
    plt.clf()
    show_the_result_with_plt(model_randomforestclassifier, x_test, y_test)
    plt.clf()

pass