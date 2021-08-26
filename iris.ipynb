import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.linear_model import LogisticRegression

df=pd.read_csv("/content/iris.csv")
df.head()

dummies=pd.get_dummies(df.iloc[:,-1])
dummies

merge=pd.concat([df,dummies],axis='columns')
merge

final=merge.drop(['state'],axis='columns')
final.head()

final=final.drop(['Iris-setosa'],axis='columns')
final.head()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
newdf=df
newdf.state=le.fit_transform(newdf.state)
newdf

df.info()

x=df.iloc[:,:4]
y=df.iloc[:,-1]
print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=5)  

mymodel=LogisticRegression(random_state=2)
mymodel.fit(x_train,y_train)

y_pred=mymodel.predict(x_test)#y_test
print(y_pred)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
ac

import seaborn as sns
sns.set_style("whitegrid")
sns.FacetGrid(df, hue ="state",
              height = 6).map(plt.scatter,
                              'sepal length',
                              'petal length').add_legend()

