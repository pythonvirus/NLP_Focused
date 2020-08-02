#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 09:00:41 2020

@author: Sachin Gupta
"""

import pandas as pd
import codecs
import os
import re, math
from scipy.sparse import hstack
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
import string
from sklearn.feature_extraction.text import TfidfVectorizer #,CountVectorizer
from sklearn.model_selection import train_test_split
from pickle import dump, load
import numpy as np
from collections import Counter
from nltk.corpus import gutenberg
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
import lightgbm as lgbm
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,precision_score,f1_score
from sklearn.metrics.pairwise import cosine_similarity 
from gensim.models import Word2Vec
from  scipy import spatial
from sklearn.model_selection import GridSearchCV


data=pd.read_csv('/home/sachingupta/Documents/Projects/Clause_using_NLP/train_assign.csv',encoding="utf-8")
Ass_data = data[['File Name','Agreement']]
Ass_data.dropna(inplace= True)


WORD = re.compile(r'\w+') #Cleaning the text


####################Defining Functions#########################################

def get_cosine(vec1, vec2):
    
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


###############################################################################
    '''Word2vec'''
###############################################################################

def avg_feature_vector(text,model,num_features, index2word_set):
    #sentence = nltk.sent_tokenize(text)
    #words = nltk.word_tokenize(sentence)
    words = WORD.findall(text)
    words = [word for word in words if not word in stopwords.words()]
    #words = simple_preprocess(words)
    #words = sentence.split()
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec


def pre_process(text):
#text=str(text.decode('utf8'))
    text = ''.join(c for c in text if c not in string.punctuation)
    text = ''.join(i for i in text if not i.isdigit())
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    words = ""
    for i in text:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i)).lower()+" "
    w = ""
    gram=2
    for i in range(len(words) - gram + 1):
            w += ' '.join(words[i:i + gram])
    return words


###############################################################################
'''Training a tokenizer to understand the difference between 
normal sentence and abbreviations.'''  
###############################################################################

'''
text = ""
for file_id in gutenberg.fileids():
    text += gutenberg.raw(file_id)
trainer = PunktTrainer()
trainer.INCLUDE_ALL_COLLOCS = True
trainer.train(text)
tokenizer = PunktSentenceTokenizer(trainer.get_params())
abbv_list=['s','ss','r','rr','art','arts','regn','regns','cl','cls','o.r.','o.r','o.rr','u/s','u/ss','u/r','u/rr','u/art','u/arts','u/regn','u/regns','u/cl','u/cls','under o','under o.r','pw','pws','dw','dws','rs','ao','a.o','vig','id','w.e.f','wef','g.o','go','ld','l.d','no','vs']
for x in abbv_list:
    tokenizer._params.abbrev_types.add(x)
    
import pickle
filename = 'tokenizer3.tok'
pickle.dump(tokenizer, open(filename, 'wb'),protocol=2)'''


##########################Reading the Tokenizer###############################
read_tokenizer=open("/home/sachingupta/Documents/Projects/Clause_using_NLP/tokenizer3.tok","rb")
import pickle
tokenizer= pickle.load(read_tokenizer)


###############################################################################
'''Converting to the Supervise data'''  
###############################################################################

stories = list()
for row in Ass_data.iterrows():
    filename=row[1]['File Name']
    text=row[1]['Agreement']
    preprocessed_text=""
    all_files=os.listdir("/home/sachingupta/Documents/Projects/Clause_using_NLP/All_Text_python_OCR/")
    idx=-1
    for index,name in enumerate(all_files):
        if filename.lower()==name.lower():
            idx=index
            break
    if idx!=-1:
        f=codecs.open("/home/sachingupta/Documents/Projects/Clause_using_NLP/All_Text_python_OCR/"+all_files[idx],encoding='utf-8')
        full_text=f.read()
        full_text=tokenizer.tokenize(full_text)
    
        full_text=[x for x in full_text if '\n\n' not in x and len(x) > 10]
            
        used_indices=[]
        cosine_score=[]
        Ass_list_target=[]
        Ass_list_original=[]
        for t in text:
            cosine_max=0.0
            target=-1
            for loc,d in enumerate(full_text):
                #vector1 = text_to_vector(t)
                #vector2 = text_to_vector(d)
                #cosine = get_cosine(vector1, vector2)
            
                #For word2vec try
                model = Word2Vec(full_text, size = 100,window=5, min_count=1)
                index2word_set = set(model.wv.index2word)
                vector1 =avg_feature_vector(t, model=model, num_features=100, index2word_set=index2word_set)#text_to_vector(t)
                vector2 = avg_feature_vector(d, model=model, num_features=100, index2word_set=index2word_set)
                cosine = 1 - spatial.distance.cosine(vector1, vector2)
                
                if float(cosine) > float(0.70):
                    target=loc
                    stories.append({'text':full_text[target],'location':float(target)/float(len(full_text)),'summary':t,'label':"Ass",'filename':filename})
                    used_indices.append(target)
                    cosine_score.append(cosine)
                    Ass_list_target.append(d)
                    Ass_list_original.append(t)
             
        for i in range(0,len(full_text)):
            if i not in used_indices:
                stories.append({'text':full_text[i],'location':float(i)/float(len(full_text)),'summary':"NA",'label':"NA",'filename':filename})
    
df=pd.DataFrame(stories)

df.groupby(df['label']).count()#Highly Inbalanced Class
'''Ass 5331
   NA 42079'''

svm_df=df.drop(['summary','filename'],axis=1).reset_index(drop=True)

#Target Variable encoding
svm_df['label']=np.where(svm_df['label']=='Ass',1,0)


###############################################################################
'''Adding More Features using TFIDF'''
###############################################################################

textFeatures =svm_df['text'].copy()
textFeatures = textFeatures.apply(pre_process)

vectorizer = TfidfVectorizer("english", ngram_range=(2,2))
features = vectorizer.fit_transform(textFeatures)

#combining features
features=hstack((features,np.array(svm_df['location'])[:,None]))

''' Visulizing the matrix '''

#va = features.tocsr()
#va[4,19]

svm_df['label'].unique()
###############################################################################
'''Model Training'''
###############################################################################

features_train, features_test, labels_train, labels_test = train_test_split(features, svm_df['label'], test_size=0.3, random_state=111)

features_train=features_train.astype('float32')
labels_train=labels_train.astype('float32')
features_test=features_test.astype('float32')
labels_test=labels_test.astype('float32')
d_train = lgbm.Dataset(features_train, labels_train)
d_valid = lgbm.Dataset(features_test,labels_test)

################## Base Model Creation#########################################
'''params = {'boosting_type': 'gbdt',
          'max_depth' : -1,
          'objective': 'binary',
          'nthread': 3, # Updated from nthread
          'num_leaves': 64,
          'learning_rate': 0.05,
          'max_bin': 512,
          'subsample_for_bin': 200,
          'subsample': 1,
          'subsample_freq': 1,
          'colsample_bytree': 0.8,
          'reg_alpha': 5,
          'reg_lambda': 10,
          'min_split_gain': 0.5,
          'min_child_weight': 1,
          'min_child_samples': 5,
          'scale_pos_weight': 1,
          'num_class' : 1,
          'metric' : 'binary_error'}

#Base model creation
mdl = lgbm.LGBMClassifier(boosting_type= 'gbdt',
          objective = 'binary',
          n_jobs = 3, # Updated from 'nthread'
          silent = True,
          max_depth = params['max_depth'],
          max_bin = params['max_bin'],
          subsample_for_bin = params['subsample_for_bin'],
          subsample = params['subsample'],
          subsample_freq = params['subsample_freq'],
          min_split_gain = params['min_split_gain'],
          min_child_weight = params['min_child_weight'],
          min_child_samples = params['min_child_samples'],
          scale_pos_weight = params['scale_pos_weight'])

mdl.get_params().keys()

mdl.fit(features_train, labels_train)


#Testing Base model

dat=mdl.predict(features_test)

features_train.shape
features_test.shape #14223

predict=[1 if x>0.5 else 0 for x in dat]

pd.crosstab(labels_test,predict)
'''
#0.0        12642
#1.0         1581

'''
accuracy_score(labels_test,predict)#0.90
confusion_matrix(labels_test,predict)
'''    #  0         1
    #0  [12597,     45],
    #1  [1353,     228]

'''

recall_score(labels_test,predict)#0.14
precision_score(labels_test,predict)#0.83
f1_score(labels_test,predict)#0.245

############################# Cross validation#################################

from sklearn.model_selection import cross_val_score

accuracy=cross_val_score(estimator=mdl,
                         X=features_train,
                         y=labels_train,
                         cv=10)

accuracy.mean() #0.89
accuracy.std()

accuracy





###################### Tunning the parameter using Grid search#################


gridParams = {
    'learning_rate': [0.005],
    'n_estimators': [40],
    'num_leaves': [6,8,12,16],
    'boosting_type' : ['gbdt'],
    'objective' : ['binary'],
    'random_state' : [501], # Updated from 'seed'
    'colsample_bytree' : [0.65, 0.66],
    'subsample' : [0.7,0.75],
    'reg_alpha' : [1,1.2],
    'reg_lambda' : [1,1.2,1.4],
    }

grid = GridSearchCV(mdl, gridParams,
                    verbose=0,
                    cv=4,
                    n_jobs=-1)
# Run the grid
grid.fit(features_train, labels_train)

# Print the best parameters found
print(grid.best_params_)
'''
#{'boosting_type': 'gbdt', 'colsample_bytree': 0.65, 'learning_rate': 0.005, 
 #'n_estimators': 40, 'num_leaves': 6, 'objective': 'binary', 'random_state': 501,
 #'reg_alpha': 1, 'reg_lambda': 1, 'subsample': 0.7}'''
#print(grid.best_score_) #88.7

'''
params['colsample_bytree'] = grid.best_params_['colsample_bytree']
params['learning_rate'] = grid.best_params_['learning_rate']
# params['max_bin'] = grid.best_params_['max_bin']
params['num_leaves'] = grid.best_params_['num_leaves']
params['reg_alpha'] = grid.best_params_['reg_alpha']
params['reg_lambda'] = grid.best_params_['reg_lambda']
params['subsample'] = grid.best_params_['subsample']

'''
######################Final model creation####################################

params = {
    'objective' :'binary',
    'learning_rate' : 0.05,
    'num_leaves' : 60,
    'feature_fraction': 0.4, 
    'bagging_fraction': 0.4, 
    'bagging_freq':1,
    'boosting_type' : 'gbdt',
    'metric': 'binary_logloss'
}

bst = lgbm.train(params, d_train, 2000, valid_sets=[d_valid], verbose_eval=100, early_stopping_rounds=150)



#Validating the model

dat=bst.predict(features_test)

features_train.shape#33187(3750(1),29437(0))
features_test.shape #14223

predict=[1 if x>0.5 else 0 for x in dat]

#pd.crosstab(labels_test,predict)
'''
0.0        12506
1.0          130

'''
accuracy_score(labels_test,predict)#0.93
confusion_matrix(labels_test,predict)
'''      0         1
    0  [12307,     335],
    1  [601,     980]

'''

recall_score(labels_test,predict)#0.61
precision_score(labels_test,predict)#0.74
f1_score(labels_test,predict)#0.67

###############################################################################
''' Validating the model '''
###############################################################################

Story_Valid=[]
for filename in os.listdir("/home/sachingupta/Documents/Projects/Clause_using_NLP/All_Text_python_OCR/"):
    f=codecs.open('/home/sachingupta/Documents/Projects/Clause_using_NLP/All_Text_python_OCR/'+filename,encoding='utf-8')
    full_text=f.read()
    full_text=tokenizer.tokenize(full_text)
    
    full_text=[x.strip() for x in full_text if len(x.split())>20]

    full_text=[x.strip() for x in full_text if '\n\n' not in x and len(x) > 10]

    for ind,chunk in enumerate(full_text):
        Story_Valid.append({'text':chunk,'location':float(ind)/float(len(full_text)),'filename':filename})
        

svm_test=pd.DataFrame(Story_Valid)#65904
svm_test.shape
svm_test=svm_test[svm_test['location']<0.7] #46425 #Consedering the only 70% of text
svm_test.shape
testFeatures =svm_test['text'].copy()
testFeatures = testFeatures.apply(pre_process)
test_features = vectorizer.transform(testFeatures) 
test_features=hstack((test_features,np.array(svm_test['location'])[:,None]))

gen=bst.predict(test_features)

ls=[x for x in gen if x>0.7]
len(ls)#191
svm_test['label']=['Caluse_Name' if x>0.7 else 'NA' for x in gen]
svm_test=svm_test[svm_test['label']!='NA']
output=svm_test.groupby('filename').agg({'text':lambda x: '****'.join(x)}).reset_index()
output.shape #139

Ass_data.columns=['filename','Clause_Name']
Ass_data.shape#159
output2=pd.merge(Ass_data,output,how='left',on='filename')
output2.shape

output2.to_csv("/home/sachingupta/Documents/Projects/Clause_using_NLP/countvec_output.csv",encoding='utf-8')

#==============================================================================
'''saving the vectorizer and lgbm model'''
#==============================================================================
'''import pickle
filename = 'lgb_Ass_model.sav'
pickle.dump(bst, open(filename, 'wb'))
#storing lgb model

with open('lgb_Ass_vectorizer.pk', 'wb') as fin:
    pickle.dump(vectorizer, fin)'''
