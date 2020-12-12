# importing the Dataset

import pandas as pd

messages = pd.read_csv('SMSSpamCollection', sep='\t',names=["label", "message"])

#Data cleaning and preprocessing
import re
#import nltk
#nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet=WordNetLemmatizer()

corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i]) 
    #Remove all other character except a-z or A-Z
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    
    '''review = [wordnet.lemmatize(word) for word in review if not word in stopwords.words('english')]'''
    review = ' '.join(review)
    corpus.append(review)
    
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
#,TfidfVectorizer
cv = CountVectorizer(max_features=200)
X = cv.fit_transform(corpus).toarray()

#cv = TfidfVectorizer(max_features=1500)
#X = cv.fit_transform(corpus).toarray()



#Target variable Label encoding
y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values


# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score

confusion_m=confusion_matrix(y_test,y_pred)
accuracy=accuracy_score(y_test,y_pred) #0.98 with Stemming
recall_score=recall_score(y_test,y_pred)
precision_score=precision_score(y_test,y_pred)
#0.9829 accuracy with lemitization has taken time

#Try TF-IDF with 1500 features accuracy 0.9811
#Try  Stemmer vs Lemitization 


















