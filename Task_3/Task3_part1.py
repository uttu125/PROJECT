#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement :
# 
# Given is a list of business descriptions for around 24,000 companies (provided in the file company descriptions). There are 2 types of descriptions – long and short. Use your discretion while using either. Perform the following task using the file:  
# Classify these companies based on their business descriptions to only one of the industries from
# the industry labels given (provided in the file: Industry Segments – Top 10 Keywords).   
# 
# ## Sequential steps followed during problem solving :
# 1. Exploring the dataset.
# 2. Variables Identification.
# 3. Missing value Treatement.
# 4. Text Preprocessing.
# 5. Applying algorithm (first on short description column).
# 6. Follow step 3, 4, 5 for company description (long) column.
# 7. Finally compare result on basis of both columns.
# 

# ## 1. Exploring the datasets :
# 

# In[1]:


# importing common used library

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# importing datasets on google colab

#from google.colab import files
#files.upload()


# In[3]:


#files.upload()


# In[4]:


# reading the datasets

keyword = pd.read_excel("Industry Segments - Top 10 Keywords.xlsx")
company = pd.read_excel("company descriptions.xlsx")

# copying the datasets so that after any manupulation we still have original dataset

keyword_copy=keyword.copy()
company_copy=company.copy()

# checking dimension of datasets

print("keyword dataset dimension : ",keyword.shape)
print("company dataset dimension : ",company.shape)


# In[5]:


# accessing first five rows of keyword dataset to see how it looks like

print("*"*71)
print("first five rows of keyword data :-")
print("*"*71)
keyword.head()


# In[6]:


# accessing first five rows of company dataset to see how it looks like

print("*"*71)
print("first five rows of company data:-")
print("*"*71)
company.head()


# ## 2. Variables Identification:
# 
# 

# In[7]:


# checking datatype of each columns and any missing values

print(keyword.info())
print("*"*71)
print("*"*71)
print(company.info())


# ## Observations :
# 1. Keyword datset consist of two columns, Idustry segment column has 28 non-missing values/rows out of 30 while Top 10 keywords column has 27 non-missing values/rows out of 30.
# 2. Keyword datset consist of three columns, company_name column has 19965 non-missing values/rows out of 19965 while company_short_description column has 19965 non-missing values/rows out of 19965 and company_description column has 19237 non-missing values/rows out of 19965.
# 3. The datatype of all columns in both datasets are Object.
# 

# ## 3. Missing Value Treatment:

# In[8]:


# Check the percentage of null values per variable

print("% of null values for each column:\n",keyword.isnull().sum()/keyword.shape[0]*100)
print("*"*71)
print("*"*71)
print("% of null values for each column:\n",company.isnull().sum()/company.shape[0]*100)


# ## Observations :
# 
# 1. Dropping all rows containing NAN from keyword dataset because we can't fill directly any value (like mode etc) because keywords is related to that particular company profile.
# 2. Not dropping any values from company dataset because I am considering only two columns (company_name, company_short_description) and there is no missing values for these two columns and further we can extend to third column.
# 

# In[9]:


# dropping missing values from keyword dataset as we can't fill directly value (like mode etc) because keyword is related to that particular company profile

keyword = keyword.dropna()
print("keyword dataset new dimension : ",keyword.shape)

# considering only two columns

company_short = company[['company_name', 'company_short_description']]
print("company_short dataset dimension : ",company_short.shape)


# ## 4. Text Preprocessing : 

# In[10]:


import re
import string

# removing number, punctuation etc
# converting into lower case

alphanumeric = lambda x: re.sub(r"""\w*\d\w*""", ' ', x)
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())

keyword['Top 10 keywords'] = keyword['Top 10 keywords'].map(punc_lower)

company_short['company_short_description'] = company_short.company_short_description.map(alphanumeric).map(punc_lower)
company_short.head(3)


# In[11]:


keyword.head(3)


# In[12]:


# importing library for text preprocessing part

import nltk

# this 2nd line doesn't require in jupyter notebook after downloading once
# but in google golab it requires again if we end/close this notebook

nltk.download('stopwords')


# In[13]:


# importing stopwords 
# the reason for removing stopwords is that if some company description have large number of stopwords then probability will decrease due to this

# even this step can be performed using CountVectorizer (using stop word parameter, I had done in clustering part) 
# and there is no need of CountVectorizer here

from nltk.corpus import stopwords

stop = stopwords.words('english')


# In[14]:


# removing stopwords

company_short['company_short_description'] = company_short['company_short_description'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
company_short.head(3)


# In[15]:


# for tokenization, We have to download punkt

nltk.download('punkt')


# In[16]:


# tokenizing short description of each word

from nltk.tokenize import word_tokenize

company_short['tokenized_word'] = company_short['company_short_description'].apply(word_tokenize)
company_short.head(3)


# In[17]:


keyword['tokenized'] = keyword['Top 10 keywords'].apply(word_tokenize)
keyword.head(3)


# ## 5. Applying Algorithm :
# 
# 1. Taking first row from company dataset.
# 2. Taking first row from keyword dataset.
# 3. Now finding number of matching keyword (tokenized column) with respect to tokenized_word column (in company_short table).
# 4. Dividing number obtained in 5.3 by number of words in that particular row of tokenized_word column (in company_short table).
# 5. Running 5.2, for all rows of kwyword dataset.
# 6. Finding max probability after running 5.5.
# 7. Running 5.1, for all rows of company dataset.
# 
# Note :
# 1. There may be exception as for some iteration, probability value zero and this will happen when there is no matching words.
# 
# Another Algorithm :
# 1. Tf-idf can also be used to find probability etc (This concept, I have used in clustering).

# In[18]:


# actually this step I have added after running just next line of code and next line is giving zero division error
# it means that some(in this dataset there is only one such row whose index is 8671) of the values in company_short_description are empty string ('') 

print(company_short.loc[8670:8672,:])
company_short=company_short[company_short.company_short_description != '']


# In[19]:


# for storing maximum probability value
result = []

# for storing possible Industry segment for each of the company
result1 = []

# for each observation of company dataset, I am going to run all observations of keyword dataset
for j in company_short.loc[:,'tokenized_word']:
  # for storing probability value for each rows of keyword dataset 
  li = []
  # initializing sum equals to zero, it will keep track of counting of matcing words
  sum = 0
  
  for i in keyword.iloc[:,2]:
    # running loop for each keyword of each row of keyword dataset
    for k in i:
      # counting occurence of each particular word in tokenized_word column of company_short dataset
      count = j.count(k)
      sum += count

    # finding probability
    sum = sum/len(j)
    li.append(sum)
    
  #print(li)
  #print(len(li))
  #print(li.index(max(li)))
  #print(max(li))

  # now taking out maximum probability value from li list and appending to result list
  result.append(max(li))
  # now taking out index corresponding to maximum probability value and then Industry segment corresponding to this and appending to result1 list
  result1.append(keyword.loc[li.index(max(li)),'Industry segment'])


# In[20]:


# adding two new columns in company_short dataset

company_short['probab'] = result
company_short['tag/Industry_segment'] = result1


# In[21]:


# finally seeing tag/Industry_segment corresponding to each company
# we can observe 5th row, probab value and it is zero meaning that there is no matching keyword and hence there is no tag
# for such case, I am replacing with 'Others'

company_short.head()


# In[22]:


# finding all such where probab is zero and then assigning Others to tag/Industry_segment column

company_short.loc[company_short['probab'] == 0.000000, 'tag/Industry_segment'] = 'Others'


# In[23]:


# now we can see 5th row, column tag/Industry_segment column

company_short.head()


# ## 6. Applying same techinque to column company_description (long description) :

# In[24]:


# selecting two columns only

# under section 3, we have already observed missing values
# we again can't fill any value (like mode etc) so I have decided to drop such rows

# another solution, we can fill short description value into long description value

company_long = company[['company_name', 'company_description']]
company_long = company_long.dropna()
print("company_long dataset dimension : ",company_long.shape)


# In[25]:


import re
import string

# removing number, punctuation etc
# converting into lower case

alphanumeric = lambda x: re.sub(r"""\w*\d\w*""", ' ', x)
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())

company_long['company_description'] = company_long.company_description.map(alphanumeric).map(punc_lower)
company_long.head(3)


# In[26]:


# removing stopwords

company_long['company_description'] = company_long['company_description'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
company_long.head(3)


# In[27]:


# tokenizing company description of each word

company_long['tokenized_word_1'] = company_long['company_description'].apply(word_tokenize)
company_long.head(3)


# In[28]:


# removing rows if it contains empty string

company_long=company_long[company_long.company_description != '']


# In[29]:


# applying same algorithm 

result = []
result1 = []

for j in company_long.loc[:,'tokenized_word_1']:
  
  li = []
  
  sum = 0
  
  for i in keyword.iloc[:,2]:
    for k in i:
      
      count = j.count(k)
      sum += count

    
    sum = sum/len(j)
    li.append(sum)
    
  
  #print(li)
  #print(len(li))
  #print(li.index(max(li)))
  #print(max(li))

  result.append(max(li))
  result1.append(keyword.loc[li.index(max(li)),'Industry segment'])


# In[30]:


company_long['probab_1'] = result
company_long['tag_1'] = result1


# In[31]:


company_long.loc[company_long['probab_1'] == 0.000000, 'tag_1'] = 'Others'


# In[32]:


company_long.head(5)


# ## 7. Observing and comparing which columns out of two works well:

# In[33]:


company_short.head(20)


# In[34]:


company_long.head(20)


# ## Conclusion after comparision :
# 
# 1. For each matching rows from both dataframe, I have visited the site to cross check Industry_segment and find that company description (long)(company_long) works well compare to short_decription column (company_short).
# 2. Both dataframe have different number of rows.

# In[35]:


# converting df to .csv
# downloading into local machine

company_long.to_csv('long_descrip.csv')
#files.download('long_descrip.csv')


# In[36]:


company_short.to_csv('short_descrip.csv')
#files.download('short_descrip.csv')


# ## References :
# 1. Stackoverflow.
# 2. Geeks for Geeks.
