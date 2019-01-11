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




# import twitter data
data = pd.read_csv("input/gender-classifier-DFE-791531.csv",encoding="latin1")

data = pd.concat([data.gender,data.description],axis=1)             #get from dataset only the gender and tweet description

#let's drop NaN values
data.dropna(axis=0,inplace=True)

pass
#to do clean the data MF
data = data[data.gender.isin(['female','male'])]
# data.gender = [1 if each == "female" else 0 for each in data.gender] #change female to 1 and male to 0
data['gender'] = (data.gender == 'female').astype(int)

description_list = []
for description in data.description:                                #remove from all description stop words and set lowercase
    description = re.sub("[^a-zA-Z]"," ",description)
    description = description.lower()
    description = nltk.word_tokenize(description)
    #description = [ word for word in description if not word in set(stopwords.words("english"))]
    lemma = nlp.WordNetLemmatizer()
    try:
        description = [lemma.lemmatize(word) for word in description]
    except Exception as e:
        if "NLTK Downloader" in e.args[0].split("\n")[3]:
            print ("downloading the NLTK package " + e.args[0].split("\n")[6].split("'")[1])
            nltk.download(e.args[0].split("\n")[6].split("'")[1])
        description = [lemma.lemmatize(word) for word in description] #change the releated words and gremmers
    description = " ".join(description)
    description_list.append(description)

#we can define max_features
max_features = 1000
#create the instance of vector
count_vectorizer = CountVectorizer(max_features=max_features,stop_words = "english")
tfid_vectorizer = TfidfVectorizer(max_features=max_features)


sparse_matrix = count_vectorizer.fit_transform(description_list).toarray() # x
sparse_matrix_ftid = tfid_vectorizer.fit_transform(description_list).toarray()

print("{} most common words for CountVectorizer: {}".format(max_features,count_vectorizer.get_feature_names()))
print("{} most common words for TfidfVectorizer: {}".format(max_features,tfid_vectorizer.get_feature_names()))

#Train and Test Split
y = data.iloc[:,0].values   # male or female classes (gender)
x = sparse_matrix           # full tweet

x_ftid = sparse_matrix_ftid

# word_count = np.array([[len(desc.split(" ")) for desc in description_list]]).transpose()
word_count = np.array([[len(desc) for desc in description_list]]).transpose() # add count to world of post
x = np.concatenate([x, word_count], axis=1)
x_ftid = np.concatenate([x_ftid, word_count], axis=1)



# train test split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.15, random_state = 42)
x_train_ftid, x_test_ftid, y_train_ftid, y_test_ftid = train_test_split(x_ftid,y, test_size = 0.15, random_state = 42)


#GaussianNB model
nb = GaussianNB()
nb_ftid = GaussianNB()

nb.fit(x_train,y_train)             #train to model by full tweet and gender
nb_ftid.fit(x_train_ftid,y_train_ftid)             #train to model by full tweet and gender

print("using GaussianNB model : ")
print("accuracy with CountVectorizer for train : ",nb.score(x_train,y_train))
print("accuracy with CountVectorizer for test  : ",nb.score(x_test,y_test))

print("accuracy with TfidfVectorizer vector for train : ",nb_ftid.score(x_train_ftid,y_train_ftid))
print("accuracy with TfidfVectorizer vector for test  : ",nb_ftid.score(x_test_ftid,y_test_ftid))


#RandomForestClassifier model
rf = RandomForestClassifier(n_estimators = 100)
rf_tfid = RandomForestClassifier(n_estimators = 100)

rf.fit(x_train,y_train)
rf_tfid.fit(x_train_ftid,y_train_ftid)

print("using RandomForestClassifier  : ")
print("accuracy with CountVectorizer for train : ",rf.score(x_test,y_test))
print("accuracy with CountVectorizer for test  : ",rf.score(x_test,y_test))

print("accuracy with TfidfVectorizer vector for train : ",rf_tfid.score(x_train_ftid,y_train_ftid))
print("accuracy with TfidfVectorizer vector for test  : ",rf_tfid.score(x_test_ftid,y_test_ftid))


y_pred = nb_ftid.predict(x_test)
y_true = y_test

cm_nb = confusion_matrix(y_true,y_pred)

sns.heatmap(cm_nb,annot=True,cmap="RdPu",fmt=".0f",cbar=False)
#plt.show()
#plt.clf()



#                           Second time                                    #
y_pred_2 = rf.predict(x_test)
y_true_2 = y_test

cm_rf_2 = confusion_matrix(y_true_2,y_pred_2)

sns.heatmap(cm_rf_2,annot=True,cmap="RdPu",fmt=".0f",cbar=False)
plt.show()

pass