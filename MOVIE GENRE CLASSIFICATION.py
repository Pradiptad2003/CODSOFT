#!/usr/bin/env python
# coding: utf-8

# In[81]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as mtp


# In[82]:


# Data collection 
# Training set data 
train_df=pd.read_csv("train_data.txt",sep=':::', names = ['ID', 'TITLE', 'GENRE', 'DESCRIPTION'], engine='python' )
train_df.head()
                      

                      


# In[83]:


# test set data 
test_df=pd.read_csv("test_data.txt",sep=':::', names =['ID', 'TITLE', 'DESCRIPTION' ], engine='python' )
test_df.head()


# In[84]:


# test set solution data 
 
test_sol_df=pd.read_csv("test_data_solution.txt",sep=':::', names = ['ID', 'TITLE', 'GENRE', 'DESCRIPTION'], engine='python' )
test_sol_df.head()
                      

                      


# # Check missing value 

# In[85]:


# Check for missing values in the training data
print(train_df.isnull().sum())

# Check for missing values in the testing data
print(test_df.isnull().sum())

# Check for missing values in the test_solution  data
print(test_sol_df.isnull().sum())


# In[ ]:





# In[ ]:





# In[ ]:





# # # Split the data into independent and dependent variable 

# In[86]:


# Split the training data into independent and dependent variable 

x_train = train_df['DESCRIPTION']
y_train = train_df['GENRE']


# In[87]:


x_train.head()


# In[88]:


y_train.head()


# In[89]:


# Split the test solution data into independent and dependent variable 

x_test = test_df['DESCRIPTION']
y_test = test_sol_df['GENRE']



# In[90]:


x_test.head()


# In[91]:


y_test.head()


# In[ ]:





# In[92]:


#feature extraction using TF-IDF(We will use TF-IDF to convert text data into numerical data suitable for model training)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(stop_words='english',max_features=5000)
x_train_tfidf = tfidf.fit_transform(x_train)
x_test_tfidf = tfidf.transform(x_test)


# # Model Training

# In[93]:


# Use SVM ALGORITHM 
# Initialize the SVM classifier
from sklearn import svm
svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(x_train_tfidf, y_train)


# # Model Prediction and Evaluation

# In[94]:


y_pred=svm_classifier.predict(x_test_tfidf)
y_pred


# In[102]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#Evaluate the model 
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy : ",accuracy*100)

# Classification report
print(classification_report(y_test, y_pred))

#Confusion matrix 
conf_matrix = confusion_matrix(y_test,y_pred)
print(conf_matrix)


# In[122]:


import seaborn as sns

#Plot confusion matrix
mtp.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap='Reds', xticklabels=svm_classifier.classes_, yticklabels = svm_classifier.classes_)
mtp.title('Confusion Matrix')
mtp.xlabel('Predicted')
mtp.ylabel('Actual')
mtp.show()



# # Visualize Training and Test Results

# In[125]:


# Bar plot for accuracy
mtp.figure(figsize=(6, 4))
sns.barplot(x=['Train Accuracy', 'Test Accuracy'], y=[svm_classifier.score(x_train_tfidf, y_train), accuracy])
mtp.ylim(0, 1)
mtp.title('Model Accuracy')
mtp.show()


# In[ ]:




