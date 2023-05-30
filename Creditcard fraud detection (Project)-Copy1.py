#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


dataset = pd.read_csv("Creditcard.csv")


# In[3]:


dataset.shape


# In[5]:


dataset.isna().sum()


# In[4]:


dataset.head()


# In[5]:


pd.value_counts(dataset['Class'])


# In[6]:


sns.countplot(dataset['Class'])


# In[7]:


corrmat = dataset.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corrmat , vmax=0.5 , square=True)
plt.show()


# In[8]:


len(dataset[dataset['Class']==0])#valid transactions


# In[9]:


len(dataset[dataset['Class']==1])#invalid transctions


# In[10]:


x = dataset.iloc[: , :-1].values
y = dataset.iloc[: , -1].values


# In[12]:


#Converting imbalanced data to balanced data
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)


# In[13]:


x_res , y_res = ros.fit_resample(x,y)


# In[14]:


x.shape


# In[15]:


x_res.shape


# In[16]:


from collections import Counter
print(Counter(y))
print(Counter(y_res))


# In[17]:


from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x_res , y_res , test_size=0.3 , random_state=0)


# In[18]:


x_train.shape


# In[19]:


y_train.shape


# In[20]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 620 , random_state=0)
classifier.fit(x_train , y_train)


# In[24]:


y_pred = classifier.predict(x_test)


# In[25]:


n_errors = (y_pred != y_test).sum()


# In[26]:


n_errors


# In[27]:


y_test.shape


# In[31]:


from sklearn.metrics import confusion_matrix , accuracy_score
cm = confusion_matrix(y_test , y_pred)
sns.heatmap(cm , annot=True)
print(accuracy_score(y_test , y_pred))


# In[32]:


from sklearn.metrics import precision_score
precision_score(y_test , y_pred)


# In[33]:


from sklearn.metrics import recall_score
recall_score(y_test , y_pred)


# In[37]:


from sklearn.metrics import classification_report
classification_report(y_test , y_pred)


# In[ ]:




