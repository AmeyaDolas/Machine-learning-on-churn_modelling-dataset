#!/usr/bin/env python
# coding: utf-8

# In[73]:


#step-1 : Importing the dataset on Jupiter Notebook
import pandas as pd
data=pd.read_csv("Desktop/cm.csv")


# In[74]:


data.head()    #To see top 5 entries of dataset,to get idea about the dataset


# In[75]:


#Step-2 : To remove unnecessary columns from the dataset
data.drop(['RowNumber','CustomerId','Surname'],axis=1,inplace=True)


# In[76]:


data.head()


# In[77]:


#step-3 : To convert every categorical features into continuous features(i.e integer datatype).
data['Gender']=[1 if i == 'Female' else 0 for i in data['Gender']]   #here Gender column is converted into integer datatype.


# In[78]:


data.head()


# In[79]:


#Here we convert Geography feature into continuous value using get_dummies method
data=pd.get_dummies(prefix="Geo",data=data,columns=['Geography']) 


# In[80]:


data.head()


# In[81]:


#step-4 : dividing input features and ouput feature into two different variables.
y=data['Exited'].values   #putting target feature(i.e output feature) into variable y.


# In[82]:


x=data.drop(['Exited'],axis=1)   #putting entire dataset except the output feature(i.e Exited) into variable x


# In[83]:


#step-5 : Using describe method to get insights if the dataset
x.describe()


# In[84]:


# step-6 : Performing Normalization to input dataset(x) inorder to reduce loss function and improve accuracy of the model
import numpy as np
x_norm=(x-np.min(x))/(np.max(x)-np.min(x))    #here x_norm contains x in normalized form


# In[85]:


x_norm.describe()


# In[86]:


# step-7: from scikit-Learn liabrary import train_test_split function
from sklearn.model_selection import train_test_split


# In[87]:


#Dividing the input dataset(x_norm into training part(70%) and testing part(30%))
xtrain,xtest,ytrain,ytest=train_test_split(x_norm,y,test_size=0.3,random_state=7)


# In[88]:


# step-8 : Applying ML algorithms
#Decision tree Algorithm
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(xtrain,ytrain)       #fitting xtrain and ytrain to model
y_pred=clf.predict(xtest)          # prediction for xtest and storing it in y_pred
from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_pred)          #Calculating accuracy of the model


# In[89]:


#Random forest Algorithm
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=7,n_estimators=100)
rf = rf.fit(xtrain,ytrain)
y_rpred=rf.predict(xtest)
from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_rpred)


# In[91]:


#Gussian Naive Bayes Algorithm
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(xtrain,ytrain)
y_nbpred=gnb.predict(xtest)
from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_nbpred)


# In[92]:


#Support Vector Machine Algorithm with kernel='rbf'
from sklearn import svm
s = svm.SVC(kernel='rbf')
s = s.fit(xtrain, ytrain)
y_svmpred=s.predict(xtest)
from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_svmpred)


# In[93]:


#Support vector Machine Algorithm with kernel='poly'
from sklearn import svm
s = svm.SVC(kernel='poly')
s = s.fit(xtrain, ytrain)
y_svmpred=s.predict(xtest)
from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_svmpred)


# In[94]:


#Support vector Machine Algorithm with kernel='sigmoid'
from sklearn import svm
s = svm.SVC(kernel='sigmoid')
s = s.fit(xtrain, ytrain)
y_svmpred=s.predict(xtest)
from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_svmpred)


# In[95]:


#K-Nearest Neighbors Algorithm
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(xtrain,ytrain)
y_knnpred=knn.predict(xtest)
from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_knnpred)


# In[96]:


#Logistic Regression Algorithm
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0)
lr.fit(xtrain, ytrain)
y_lrpred=lr.predict(xtest)
from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_lrpred)


# In[97]:


####lets make a new dataset of input columns and try predicting if this model works...


# In[98]:


#following same steps till step-6
import pandas as pd
x2=pd.read_csv("Desktop/cmtopredict.csv")   #Putthing the new dataset into variable x2


# In[99]:


x2.head()


# In[100]:


x2.describe()


# In[101]:


x2.drop(['RowNumber','CustomerId','Surname'],axis=1,inplace=True)


# In[102]:


x2.head()


# In[103]:


x2.Gender=[1 if each == 'Female' else 0 for each in x2.Gender]


# In[104]:


x2.head()


# In[105]:


x2=pd.get_dummies(prefix="Geo",data=x2,columns=['Geography'])
x2.head()


# In[106]:


x2_norm=(x2-np.min(x2))/(np.max(x2)-np.min(x2))  #normalizing x2 and storing in new variable x2_norm


# In[107]:


x2_norm.describe()


# In[108]:


#since I got maximum accuracy for Decision tree and Random Forest algorithm I use them
#To predict new outputs for new input dataset x2_norm
#Decision tree Algorithm
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_norm,y)        #here we fit entire normalized input dataset and entire output dataset
y_pred=clf.predict(x2_norm)    #we do prediction for x2_norm
for i in y_pred:
    print(i)


# In[109]:


#random forest
from sklearn.ensemble import RandomForestClassifier
#rf = RandomForestClassifier(random_state=7,n_estimators=100)
rf = rf.fit(x_norm,y)
y_rpred=rf.predict(x2_norm)
for i in y_rpred:
    print(i)


# In[ ]:




