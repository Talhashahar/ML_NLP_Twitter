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




# import twitter data
data = pd.read_csv("input/gender-classifier-DFE-791531.csv",encoding="latin1")

data = pd.concat([data.gender,data.description],axis=1)             #get from dataset only the gender and tweet description

#let's drop NaN values
data.dropna(axis=0,inplace=True)

data.gender = [1 if each == "female" else 0 for each in data.gender] #change female to 1 and male to 0

nltk.download('punkt')

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
count_vectorizer = CountVectorizer(max_features=max_features,stop_words = "english")
#count_vectorizer = CountVectorizer(stop_words = "english")

sparce_matrix = count_vectorizer.fit_transform(description_list).toarray() # x

print("{} most common words: {}".format(max_features,count_vectorizer.get_feature_names()))

#Train and Test Split
y = data.iloc[:,0].values   # male or female classes (gender)
x = sparce_matrix           # full tweet
# train test split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state = 42)

nb = GaussianNB()

nb.fit(x_train,y_train)             #train to model by full tweet and gender

print("accuracy: ",nb.score(x_test,y_test))


y_pred = nb.predict(x_test)
y_true = y_test

cm_nb = confusion_matrix(y_true,y_pred)

sns.heatmap(cm_nb,annot=True,cmap="RdPu",fmt=".0f",cbar=False)
#plt.show()

rf = RandomForestClassifier(n_estimators = 100)

rf.fit(x_train,y_train)

print("accuracy atfer RandomForestClassifier: ",rf.score(x_test,y_test))

#                           Second time                                    #
y_pred_2 = rf.predict(x_test)
y_true_2 = y_test

cm_rf_2 = confusion_matrix(y_true_2,y_pred_2)

sns.heatmap(cm_rf_2,annot=True,cmap="RdPu",fmt=".0f",cbar=False)
plt.show()

pass