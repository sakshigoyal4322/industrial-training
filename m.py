import pandas as pd
import math
dataset = pd.read_csv("weight-height.csv")
import warnings
import pickle
warnings.filterwarnings("ignore")
dataset.info()
dataset.describe()
dataset.isnull().sum()
dataset['Gender'].replace('Female',0, inplace=True)
dataset['Gender'].replace('Male',1, inplace=True)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 2].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_pred = lin_reg.predict(X_test)
my_weight_pred = lin_reg.predict([[0,74]])
print(my_weight_pred)
print('R square = ',lin_reg.score(X_train,y_train))
print('Correlation = ',math.sqrt(lin_reg.score(X_train,y_train)))
pickle.dump(lin_reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))