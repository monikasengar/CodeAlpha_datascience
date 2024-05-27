#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd 
import numpy as np 
from sklearn import tree 
from sklearn import datasets 
from sklearn.datasets import load_iris 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# In[11]:


import pandas as pd
df = pd.read_csv(r'C:\Users\monik\Downloads\codalpha\IRIS.csv')
df


# In[12]:


X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# In[13]:


X


# In[14]:


y


# In[15]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[16]:


# Train a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=123)
clf.fit(X_train, y_train)


# In[17]:


# Make predictions
y_pred = clf.predict(X_test)


# In[18]:


y_pred


# In[19]:


# Plot the Decision Tree
from sklearn.tree import plot_tree
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=df['species'].unique())
plt.title('Decision Tree')
plt.show()


# In[20]:


#Decision Tree Insights
#       ** Iris-setosa is perfectly classified by 'petal length'<=2.45
#       **Iris-versicolor - Most instances are correctly classified, with some futher splits based on 'sepal length' and 'petal width'
#       **Iris-virginica is correctly classified with some splits, but with some futher splits misclassified as Iris-versicolor


# In[21]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# Evaluate the model
confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion_mat)


# In[22]:


#Confusion Matrix
#True Positives(TP): Diagonal elements, correctly classified for each class
#         for class 0:13
#         for class 1:6
#         for class 2:10

#False Positives(FP): The off-diagonal elements in each column, incorrectly predicted as that class
#        for class 0: 0,0
#         for class 1: 0,1
#         for class 2: 0,0
      
#
#

#False Negatives(FN): The off-diagonal elements in each row, that belong to that class but were predicted as some other class
#         for class 0: 0,0
#         for class 1: 0,0
#         for class 2: 0,1
  
#
#


# In[23]:


#Accuracy= total true values/total instances
#Precision= true positives/true positives + false positives
#Recall = true positives/ true positives + false negatives
#F1 Score = 2(Precision*Recall/Precision + Recall)
#
#


# In[24]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[25]:


report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)


# # Checking for a random value

# In[26]:


import pandas as pd
import numpy as np
p = {'sepal_length': ['4.7'], 'sepal_width':['3.2'], 'petal_length':['1.3'], 'petal_width':['0.2']}

dr = pd.DataFrame(p)


# In[27]:


# Make predictions
y_pred = clf.predict(dr)


# In[28]:


y_pred


# In[ ]:




