#Natural Language Processing

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset=pd.read_csv('Restaurant_reviews.tsv',delimiter='\t',quoting=3)

#Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,1000):
	review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])		#Removing punctuations and special symbols from text
	review=review.lower()		#Coverting the entire text into lowercase letters
	review=review.split()		#Splitting text into words
	ps=PorterStemmer()
	all_stopwords=stopwords.words('english')
	all_stopwords.remove('not')		#Removing not from the list of stop words as it is essential in this case
	review=[ps.stem(word) for word in review if not word in set(all_stopwords)]		#Stemming : coverting every word into root word
	review=' '.join(review)
	corpus.append(review)
#print(corpus)

#Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,-1].values
#len(x[0])

#Splitting the dataset into the Training and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#Training Random Forest Classifier model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=501,criterion='entropy')
classifier.fit(x_train,y_train)

#Predicting the Test set result
y_pred=classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)

#Prediction on a single review
new_review = 'I love this restaurant'
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier.predict(new_X_test)
print(new_y_pred)
if(new_y_pred==0):
    #print("Negative Review")
    print("Sorry for the inconvenience caused. We will definitely look into it and improve ourselves.")
else:
    #print("Positive Review")
    print("Thank you so much for your kind words. See you soon!")
