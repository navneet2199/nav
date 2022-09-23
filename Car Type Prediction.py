#!/usr/bin/env python
# coding: utf-8

# # CAR TYPE PREDICITON

# In[147]:


#Important 
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


# In[148]:


df=pd.read_csv('C:/Users/91816/Downloads/Cars.csv')


# In[149]:


df


# In[150]:


df.info()


# In[151]:


#features_names 
df.keys()


# In[152]:


#we have removed purpose colums which is of no use in data analysis


# In[153]:


df=df.iloc[:,:-1]


# In[154]:


df


# In[155]:


ef=df.values


# In[156]:


ef


# In[157]:


#fill the missing Values
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer=imputer.fit(ef[:,6:])
ef[:,6:]=imputer.transform(ef[:,6:])


# In[158]:


#convert again to Dataframe after filling Missing values
feature=['BRAND', 'TYPE', 'FUEL TYPE?', 'NEW OR OLD','TRANSMISSION TYPE',
       'ENGINE SIZE', 'BUDGET', ' RESALE']
df=pd.DataFrame(ef,columns=feature)


# In[159]:


df


# In[160]:


#emcoding the string values in dataframe
le=LabelEncoder()
BRAND=le.fit_transform(df['BRAND'])
TYPE=le.fit_transform(df['TYPE'])
FUELTYPE=le.fit_transform(df['FUEL TYPE?'])
NEWorOLD=le.fit_transform(df['NEW OR OLD'])
TRANSMISSIONTYPE=le.fit_transform(df['TRANSMISSION TYPE'])
ENGINESIZE=le.fit_transform(df['ENGINE SIZE'])
df['BRAND']=BRAND
df['TYPE']=TYPE
df['FUEL TYPE?']=FUELTYPE
df['NEW OR OLD']=NEWorOLD
df['TRANSMISSION TYPE']=TRANSMISSIONTYPE
df['ENGINE SIZE']=ENGINESIZE
df


# In[161]:


#Split the data into X and Y
Y=df['TRANSMISSION TYPE']
X=df.drop(columns='TRANSMISSION TYPE')


# In[162]:


X


# In[163]:


Y


# In[164]:


#Splitting the Data into training and testing
from sklearn.model_selection import  train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=1)


# # KNN

# In[165]:


#KNN 
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='hamming',p=2)
classifier.fit(x_train,y_train)


# In[166]:


y_pred=classifier.predict(x_test)


# In[167]:


from sklearn import metrics


# In[168]:


print("Accuracy:",metrics.accuracy_score(y_test,y_pred)*100)


# # DECISION TREE 

# In[169]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier


# In[170]:


clf=DecisionTreeClassifier()


# In[171]:


clf.fit(x_train,y_train)


# In[172]:


ypred=clf.predict(x_test)
print('Accuracy ',metrics.accuracy_score(y_test,ypred)*100)


# In[173]:


from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
import pydotplus
dot_data=StringIO()
export_graphviz(clf,out_file=dot_data,filled=True,rounded=True,special_characters=True,feature_names=X.columns,class_names=['0','1'])
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Cars.png')
Image(graph.create_png())


# # NAIVE BAYES

# In[174]:


#NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()


# In[175]:


gnb.fit(x_train,y_train)


# In[176]:


ypre=gnb.predict(x_test)


# In[177]:


print(ypre)


# In[178]:


print("Acccuracy:",metrics.accuracy_score(y_test,ypre)*100)


# In[179]:


acc1=metrics.accuracy_score(y_test,y_pred)*100
acc2=metrics.accuracy_score(y_test,ypred)*100
acc3=metrics.accuracy_score(y_test,ypred)*100


# In[180]:


ed=pd.DataFrame({'Model':['KNN','DECISION TREE','NAIVE BAYES'],'ACCURACY':[acc1,acc2,acc3]})


# In[181]:


ed


# Among this three models we have got high accuracy for Decsion Tree and Naive Bayes.
