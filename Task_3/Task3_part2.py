#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement :
# 
# Given is a list of business descriptions for around 24,000 companies (provided in the file company descriptions). There are 2 types of descriptions – long and short. Use your discretion while using either. Perform the following task using the file:  
# Given the same set of the companies with their business descriptions, cluster these companies,
# without using any information from the industry labels. Also, identify ways to calculate the
# accuracy of the clusters generated   
# 
# ## Sequential steps followed during problem solving :
# 1. Exploring the dataset.
# 2. Text Preprocessing ( including lemmatization ).
# 3. Feature Engineering using CountVectorizer.
# 4. Modeling and evaluating quality of cluster.
# 5. Feature Engineering using TfidfVectorizer.
# 6. Modeling and evaluating quality of cluster :
# 7. Conclusion.

# ## 1. Exploring the dataset :

# In[1]:


# importing common used library

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# importing file on google colab

#from google.colab import files
#files.upload()


# In[3]:


# reading the datasets
company = pd.read_excel("company descriptions.xlsx")

# copying the datasets so that after any manupulation we still have original dataset
company_copy=company.copy()

# checking dimension of dataset
company.shape


# In[4]:


# accessing first five rows of dataset to see how it looks like

print("*"*71)
print("first five rows of dataset :-")
print("*"*71)
company.head()


# ## 2. Text Preprocessing :

# In[5]:


import re
import string

# removing number, punctuation etc
# converting into lower case

alphanumeric = lambda x: re.sub(r"""\w*\d\w*""", ' ', x)
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())

company['company_short_description'] = company.company_short_description.map(alphanumeric).map(punc_lower)
company.head()


# In[6]:


# importing library for text preprocessing part

import nltk

# this 2nd line doesn't require in jupyter notebook adter downloading once
# but in google golab it requires again if we end/close this notebook

nltk.download('stopwords')


# In[7]:


# importing stopwords 

from nltk.corpus import stopwords

# remvoing stopwords as it doesn't play any important role

stop = stopwords.words('english')


# In[8]:


company['company_short_description'] = company['company_short_description'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
company.head(3)


# In[9]:


nltk.download('wordnet')


# In[10]:


# bringing derived word to its base or root form

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return  ' '.join(lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text))

company['company_short_description'] = company['company_short_description'].apply(lemmatize_text)
company.head()


# In[11]:


# considering only company_short_description for clustering

comp_short = company['company_short_description']


# ## 3. Feature Engineering using CountVectorizer :

# In[12]:


# vectorizing the sentences

from sklearn.feature_extraction.text import CountVectorizer

# max_features=1000, this parameter I have added after running model once and model is taking long time to fit on large number of columns
# total columns = 24000

vect = CountVectorizer(stop_words='english',max_features=1000)
vect.fit(comp_short)

# printing the vocabulary
vect.vocabulary_


# In[13]:


# checking length 

len(vect.vocabulary_.keys())


# In[14]:


# transforming the features using parameter obtained during fitting

comp_short_transformed = vect.transform(comp_short)
comp_short_transformed


# ## 4. Modeling and evaluating quality of cluster :

# In[15]:


# the reason for selecting KMeans algorithm over Hierarchical clustering because KMeans works well for large dataset

from sklearn.cluster import KMeans

# Silhouette score: Silhouette score tells how far away the datapoints in one cluster are, from the datapoints in another cluster.
# The range of silhouette score is from -1 to 1. Score should be closer to 1 than -1.
from sklearn.metrics import silhouette_score


# In[17]:


# elbow-curve

# Inertia: Intuitively, inertia tells how far away the points within a cluster are.
# Therefore, a small of inertia is aimed for. The range of inertia’s value starts from zero and goes up.

# this list is taken to store intertia value
ssd = []

# the range, I have taken from 2 to 8 (is generally taken in industry and the reason for it is that no industry is going to form 150 clusters
# irrespective of interia value because it is not an easy task to handle such large group of people)

# this code little more time to run

range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50,random_state=99)
    kmeans.fit(comp_short_transformed)
    
    ssd.append(kmeans.inertia_)

    cluster_labels = kmeans.labels_
    # silhouette score
    silhouette_avg = silhouette_score(comp_short_transformed, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))
    
# plotting ssd for each n_clusters

plt.plot(range_n_clusters, ssd, marker='*')
plt.xlabel('number of cluster')
plt.ylabel('Inertia')


# ## Observations :
# 
# 1. Inertia suddenly decreases (compare to all other points) at number of cluster = 4, so best choice for k is 4. (the choice may vary if we change random_state).
# 2. silhouette score is not so good but still it is not bad. Maximum points in a cluster is near to boundary because value of silhouette score is close to zero.

# In[18]:


# now fitting model for k = 4

kmeans = KMeans(n_clusters=4, max_iter=50,random_state=99)
kmeans.fit(comp_short_transformed)


# In[19]:


company_copy['labels'] = kmeans.labels_


# In[20]:


company_copy.head(10)


# In[21]:


company_copy.sort_values(['labels'])


# In[22]:


# checking % of labels value

company_copy['labels'].value_counts()*100/company_copy.shape[0]


# In[23]:


# converting into csv and then downloading into local machine

company_copy.to_csv('labels_cluster_short_description.csv')
#files.download('labels_cluster_short_description.csv')


# ## 5. Feature Engineering using TfidfVectorizer :

# In[24]:


comp_short_t = company['company_short_description']

from sklearn.feature_extraction.text import TfidfVectorizer

vect_t = TfidfVectorizer(max_features=1000)
vect_t.fit(comp_short_t)

# printing the vocabulary
vect_t.vocabulary_


# In[25]:


len(vect_t.vocabulary_.keys())


# In[26]:


comp_short_transformed_t = vect_t.transform(comp_short_t)
comp_short_transformed_t


# ## 6. Modeling and evaluating quality of cluster :

# In[27]:


# elbow-curve

ssd = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50,random_state=99)
    kmeans.fit(comp_short_transformed_t)
    
    ssd.append(kmeans.inertia_)

    cluster_labels = kmeans.labels_
  

    # silhouette score
    silhouette_avg = silhouette_score(comp_short_transformed_t, cluster_labels)
    
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))
    
# plotting ssd for each n_clusters

plt.plot(range_n_clusters, ssd, marker='*')
plt.xlabel('number of cluster')
plt.ylabel('Inertia')


# ## Observations :
# 
# 1. There is no sudden decrease in inertia at any point. So Tfidf is not giving giving good result.
# 2. silhouette score is not so good but still it is not bad. Maximum points in a cluster is near to boundary because value of silhouette score is close to zero.

# ## Conclusion :
# 1. CountVectorizer works well compare to Tfidf.
# 2. Similarly, we can follow all steps to long description column as well.
# 

# References :
# 1. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
# 2. https://towardsdatascience.com/k-means-clustering-from-a-to-z-f6242a314e9a
