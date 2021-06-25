import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

df = pd.read_csv("/content/hiring.csv")
df

df.isnull().sum()

df.info()

#fill score with mean value
df['test_score(out of 10)'].fillna(df['test_score(out of 10)'].mean(),inplace=True)
df

df['experience'].fillna(0,inplace=True)
df

def string_to_number(word):
  dict  = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,0:0}
  return dict[word]

df['experience'] = df['experience'].apply(lambda x: string_to_number(x))
df

X = df.iloc[:,:3]
Y = df.iloc[:,-1]

"""#Splitting the data into training and testing"""

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.1,random_state=5)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

import pickle
#saving model to disk
pickle.dump(regressor,open("model.pkl","wb"))

#loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2,9,6]]))

