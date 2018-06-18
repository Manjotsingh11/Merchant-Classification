
# coding: utf-8

# In[193]:


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import random as rn


# In[194]:


#read the file
my_data = pd.read_csv('/home/admin/Downloads/query_result_2018-06-13T07_37_35.954Z.csv')


my_data.head()


# In[195]:


my_data= my_data[['billing_amount', 'merchant_name', 'merchant_category']]


# In[196]:


my_data.head()


# In[197]:


my_data.merchant_category.unique()
len(set(my_data.merchant_category))


# In[198]:


print(my_data.groupby('merchant_category')['billing_amount'].nunique())


# In[199]:


my_data = my_data.dropna(how='any',axis=0) 


# In[200]:


my_data.shape


# In[201]:


#lower case
my_data['merchant_name'] = my_data['merchant_name'].apply(lambda x: " ".join(x.lower() for x in x.split()))
my_data['merchant_name'].head()


# In[202]:


#most frequent term.
freq = pd.Series(' '.join(my_data['merchant_name']).split()).value_counts()[:10]
freq


# In[203]:


#most rare terms
freq = pd.Series(' '.join(my_data['merchant_name']).split()).value_counts()[-10:]
freq


# In[204]:


#removing most rare terms
freq = list(freq.index)
my_data['merchant_name'] = my_data['merchant_name'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
my_data['merchant_name'].head()


# In[205]:


#removing punctuation
my_data['merchant_name'] = my_data['merchant_name'].str.replace('[^\w\s]','')
my_data['merchant_name'].head()


# In[206]:


#spelling correction
import textblob
from textblob import TextBlob
my_data['merchant_name'][:5].apply(lambda x: str(TextBlob(x).correct()))


# In[207]:


#tokenization
import nltk
nltk.download('punkt')
TextBlob(my_data['merchant_name'][1]).words


# In[208]:


#lemmitization
from textblob import Word
my_data['merchant_name'] = my_data['merchant_name'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
my_data['merchant_name'].head()


# In[209]:


from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble


# In[210]:


#replacing education, company and gift with Restaurant.
combine = [my_data]
for dataset in combine:
    dataset['merchant_category'] = dataset['merchant_category'].replace(['ENTERTAINMENT', 'INVESTMENTS'], 'FOOD')


# In[211]:


combine = [my_data]

title_mapping = {"FOOD": 1, "FUEL": 2, "GENERAL": 3, "MEDICAL": 4, "MOBILE": 5, "TRAVEL": 6, "UTILITIES": 7}
for dataset in combine:
    dataset['merchant_category'] = dataset['merchant_category'].map(title_mapping)
    dataset['merchant_category'] = dataset['merchant_category'].fillna(0)

my_data.head()


# In[212]:


df = my_data.groupby(['merchant_category']).agg({'billing_amount': 'mean'}).reset_index()

df = df.rename(columns={'billing_amount': 'billing_amt'})
df.head(10)


# In[213]:


#removing the billing variable.
#my_data = my_data.drop(['billing_amount'], axis=1)
my_data=pd.merge(my_data,df[['merchant_category','billing_amt']],on='merchant_category')
my_data.head()


# In[214]:


#removing the billing variable.
my_data = my_data.drop(['billing_amount'], axis=1)
my_data.head()


# In[215]:


# load the pre-trained word-embedding vectors 
embeddings_index = {}
for i, line in enumerate(open('glove.6B.50d.txt')):
    values = line.split()
    embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')


# In[218]:


from keras.preprocessing import text,sequence

# create a tokenizer 
token = text.Tokenizer()
token.fit_on_texts(my_data['merchant_name'])
word_index = token.word_index


# In[219]:


# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(my_data['merchant_name'], my_data['merchant_category'])

# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)


# In[220]:


# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(my_data['merchant_name'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)


# In[221]:


# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(my_data['merchant_name'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)


# In[222]:


# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(my_data['merchant_name'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)


# In[223]:


# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(my_data['merchant_name'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 


# In[224]:


# load the pre-trained word-embedding vectors 
embeddings_index = {}
for i, line in enumerate(open('glove.6B.50d.txt')):
    values = line.split()
    embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')


# In[225]:


import textblob
from keras.preprocessing import text,sequence

# create a tokenizer 
token = text.Tokenizer()
token.fit_on_texts(my_data['merchant_name'])
word_index = token.word_index

# convert text to sequence of tokens and pad them to ensure equal length vectors 
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)

# create token-embedding mapping
embedding_matrix = np.zeros((len(word_index) + 1, 50))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[226]:


def train_model(classifier, feature_vector_train, Category, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, Category)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)


# In[227]:


# RF on Word Level TF IDF Vectors
accuracy = train_model(RandomForestClassifier(n_estimators=100, min_samples_split=2), xtrain_tfidf, train_y, xvalid_tfidf)
print ("RF, WordLevel TF-IDF: ", accuracy)

# RF Classifier on Count Vectors
accuracy = train_model(RandomForestClassifier(n_estimators=100, min_samples_split=2), xtrain_count, train_y, xvalid_count)
print ("RF, Count Vectors: ", accuracy)

# RF on Ngram Level TF IDF Vectors
accuracy = train_model(RandomForestClassifier(n_estimators=100, min_samples_split=2), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print ("RF, N-Gram Vectors: ", accuracy)

# RF on Character Level TF IDF Vectors
accuracy = train_model(RandomForestClassifier(n_estimators=100, min_samples_split=2), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print ("RF, CharLevel Vectors: ", accuracy)


# In[85]:


#Linear Classifier on Count Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
print ("LR, Count Vectors: ", accuracy)

# Linear Classifier on Word Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
print ("LR, WordLevel TF-IDF: ", accuracy)

# Linear Classifier on Ngram Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print ("LR, N-Gram Vectors: ", accuracy)

# Linear Classifier on Character Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print ("LR, CharLevel Vectors: ", accuracy)


# In[86]:


# Naive Bayes on Word Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print ("NB, WordLevel TF-IDF: ", accuracy)

# Naive Bayes on Ngram Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print ("NB, N-Gram Vectors: ", accuracy)

# Naive Bayes on Ngram Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print ("NB, N-Gram Vectors: ", accuracy)

