# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Import the standard libraries.
Upload the dataset and check for any null values using .isnull() function.
Import LabelEncoder and encode the dataset.
Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
Predict the values of arrays.
Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
Predict the values of array.
Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Venkatesh A
RegisterNumber: 212225040485 
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = pd.read_csv("Salary_EX7.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()
```

## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)
<img width="720" height="478" alt="image" src="https://github.com/user-attachments/assets/4c0821c8-1305-400d-9725-55551da2fdd8" />
<img width="703" height="457" alt="image" src="https://github.com/user-attachments/assets/8e5d6557-b31e-47a8-b596-0506b62bc034" />
<img width="899" height="529" alt="image" src="https://github.com/user-attachments/assets/bd76500f-519b-4f2b-be8b-c99548e92f5a" />
<img width="621" height="456" alt="image" src="https://github.com/user-attachments/assets/696575f3-a3e6-413e-b131-2768dd088108" />
<img width="1064" height="643" alt="image" src="https://github.com/user-attachments/assets/4f12d110-b6ff-4d27-84a0-beec7715dcdb" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
