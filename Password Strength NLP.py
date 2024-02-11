#!/usr/bin/env python
# coding: utf-8

# In[126]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[127]:


import sqlite3


# In[128]:


con = sqlite3.connect(r"D:\Study\Sharayu\REAL WORLD PROJECT  RESOURCES/password_data.sqlite")


# In[129]:


data = pd.read_sql_query("SELECT * from Users",con)


# In[130]:


data.shape


# In[131]:


data.head(4)


# In[ ]:





# # Data Cleaning

# In[132]:


data.columns


# In[133]:


##Remove irrelevant column
data.drop(['index'],axis = 1,inplace = True)


# In[134]:


data.head(4)


# In[ ]:





# In[135]:


##Check duplicate 
data.duplicated().sum()


# In[ ]:





# In[136]:


##Check null values
data.isnull().any()


# In[137]:


data.isnull().any().sum()


# In[ ]:





# In[138]:


##Check whether we have negative strength or not
data['strength']


# In[139]:


data['strength'].unique()


# In[ ]:





# # Data Analysis

# In[140]:


data.columns


# In[141]:


data['password'][0]


# In[142]:


type(data['password'][0])


# In[143]:


data['password'].str.isnumeric()


# In[144]:


##How many password texts have only numeric character
data[data['password'].str.isnumeric()]


# In[145]:


data[data['password'].str.isnumeric()].shape


# In[ ]:





# In[146]:


##How many password have upper case charcter 
data[data['password'].str.isupper()]


# In[147]:


data[data['password'].str.isupper()].shape


# In[ ]:





# In[148]:


##How many password have alphabet only
data[data['password'].str.isalpha()]


# In[149]:


data[data['password'].str.isalpha()].shape


# In[ ]:





# In[150]:


##How many password have alphanumerice character
data[data['password'].str.isalnum()].shape                      


# In[ ]:





# In[151]:


##How many password have first letter in uppercase
data[data['password'].str.istitle()]         


# In[152]:


data[data['password'].str.istitle()].shape         


# In[ ]:





# In[153]:


##Chech how many password have special character
data['password']


# In[154]:


import string


# In[155]:


string.punctuation


# In[156]:


def find_semantics(row):
    for char in row:
        if char in string.punctuation:
            return 1
        else:
            pass
        


# In[157]:


data['password'].apply(find_semantics)


# In[158]:


data[data['password'].apply(find_semantics)==1]


# In[159]:


data[data['password'].apply(find_semantics)==1].shape


# In[ ]:





# # Feature Engineering on password feature

# In[160]:


data['password'].str.len()


# In[161]:


data['length'] = data['password'].str.len()


# In[ ]:





# In[162]:


def freq_lowercase(row):
    return len([char for char in row if char.islower()])/len(row)


# In[163]:


def freq_uppercase(row):
    return len([char for char in row if char.isupper()])/len(row)


# In[164]:


def freq_numerical_case(row):
    return len([char for char in row if char.isdigit()])/len(row)


# In[165]:


data['lowercase_freq'] = np.round(data['password'].apply(freq_lowercase), 3)
data['uppercase_freq']= np.round(data['password'].apply(freq_uppercase), 3)
data['digit_freq'] = np.round(data['password'].apply(freq_numerical_case), 3)


# In[166]:


data.head(4)


# In[ ]:





# In[167]:


def freq_special_case(row):
    special_chars = []
    for char in row:
        if not char.isalpha() and not char.isdigit():
            special_chars.append(char)
    return len(special_chars)              


# In[168]:


data['special_char_freq'] = np.round(data['password'].apply(freq_special_case), 3)


# In[169]:


data.head(4)


# In[170]:


data['special_char_freq'] = data['special_char_freq']/data['length']


# In[171]:


data.head(5)


# In[ ]:





# In[172]:


data.columns


# In[173]:


data[['length','strength']].groupby(['strength']).agg(['min','max','mean','median'])


# In[174]:


cols = ['length', 'lowercase_freq', 'uppercase_freq',
       'digit_freq', 'special_char_freq']

for col in cols:
    print(col)
    print(data[[col,'strength']].groupby(['strength']).agg(['min','max','mean','median']))
    print('\n')


# In[175]:


##Plot a boxplot
data.columns


# In[176]:


fig, ((ax1,ax2) , (ax3,ax4) , (ax5,ax6)) = plt.subplots(3,2,figsize = (15,7))

sns.boxplot(x = 'strength', y = 'length', hue = 'strength', data = data, ax = ax1)
sns.boxplot(x = 'strength', y = 'lowercase_freq', hue = 'strength', data = data, ax = ax2)
sns.boxplot(x = 'strength', y = 'uppercase_freq', hue = 'strength', data = data, ax = ax3)
sns.boxplot(x = 'strength', y = 'digit_freq', hue = 'strength', data = data, ax = ax4)
sns.boxplot(x = 'strength', y = 'special_char_freq', hue = 'strength', data = data, ax = ax5)

plt.subplots_adjust(hspace = 0.6)


# In[ ]:





# # #Feature Engineering
# # Univariate Analysis

# In[177]:


def get_dist(data,feature):
    
    plt.figure(figsize = (10,8))
    plt.subplot(1,2,1)
    sns.violinplot(x= 'strength',y = feature,data = data)
    
    plt.subplot(1,2,2)
    sns.distplot(data[data['strength']==0][feature],color = 'red',label ='0',hist = False )
    sns.distplot(data[data['strength']==0][feature],color = 'blue',label ='1',hist = False)
    sns.distplot(data[data['strength']==0][feature],color = 'orange',label ='2',hist = False ) 
    plt.legend()
    plt.show()


# In[178]:


import warnings
from warnings import filterwarnings
filterwarnings('ignore')


# In[179]:


get_dist(data,'length')


# In[180]:


get_dist(data,'lowercase_freq')


# In[181]:


get_dist(data,'uppercase_freq')


# In[182]:


get_dist(data,'digit_freq')


# In[183]:


get_dist(data,'special_char_freq')


# In[ ]:





# In[184]:


data


# In[185]:


##Shuffling
dataframe = data.sample(frac = 1)


# In[186]:


dataframe


# In[ ]:





# # NLP Algotithm

# In[187]:


x = list(dataframe['password'])


# In[188]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[189]:


vectorizer = TfidfVectorizer(analyzer='char')


# In[190]:


X = vectorizer.fit_transform(x)


# In[191]:


X.shape


# In[192]:


dataframe['password'].shape


# In[193]:


X


# In[194]:


X.toarray()


# In[195]:


X.toarray()[0]


# In[196]:


dataframe['password']


# In[197]:


len(vectorizer.get_feature_names_out())


# In[198]:


vectorizer.get_feature_names_out()


# In[199]:


df2 = pd.DataFrame(X.toarray(), columns = vectorizer.get_feature_names_out())


# In[200]:


df2


# In[ ]:





# In[201]:


df2['length'] = dataframe['length']
df2['lowercase_freq'] = dataframe['lowercase_freq']


# In[202]:


df2


# In[ ]:





# In[ ]:





# # Applying Machine Learning Model

# In[203]:


y = dataframe['strength']


# In[204]:


from sklearn.model_selection import train_test_split


# In[205]:


X_train, X_test, y_train, y_test = train_test_split(df2,y, test_size = 0.20)


# In[206]:


X_train.shape


# In[207]:


y_train.shape


# In[ ]:





# In[208]:


from sklearn.linear_model import LogisticRegression


# In[209]:


clf = LogisticRegression(multi_class = 'multinomial')


# In[210]:


clf.fit(X_train,y_train)


# In[211]:


y_pred = clf.predict(X_test)


# In[212]:


y_pred


# In[ ]:





# In[213]:


from collections import Counter


# In[214]:


Counter(y_pred)


# In[ ]:





# # Doing Predictions

# In[217]:


def predict():
    password = input("Enter a password : ")
    sample_array = np.array([password])
    sample_matrix = vectorizer.transform(sample_array)
    
    length_pass = len(password)
    length_normalised_lowercase = len([char for char in password if char.islower()])/len(password)
    
    new_matrix = np.append(sample_matrix.toarray(),(length_pass, length_normalised_lowercase)).reshape(1,101)
    result = clf.predict(new_matrix)
    
    if result == 0:
        return "Password is Weak"
    elif result == 1:
        return "Password is Normal"
    else: 
        return "Password is Strong"


# In[218]:


predict()


# In[ ]:





# # Checking Accuracy Score

# In[223]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[224]:


accuracy_score(y_test,y_pred)


# In[225]:


confusion_matrix(y_test,y_pred)


# In[226]:


print(classification_report(y_test,y_pred))


# In[ ]:




