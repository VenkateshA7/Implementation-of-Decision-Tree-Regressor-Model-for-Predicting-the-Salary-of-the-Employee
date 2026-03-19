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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


data = {
    'Position': ['Business Analyst', 'Junior Consultant', 'Senior Consultant',
                 'Manager', 'Country Manager', 'Region Manager',
                 'Partner', 'Senior Partner', 'C-level', 'CEO'],
    'Level': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [45000, 50000, 60000, 80000, 110000, 150000, 200000, 300000, 500000, 1000000]
}

df = pd.DataFrame(data)


X = df[['Level']]     # Feature (Level)
y = df['Salary']      # Target (Salary)


regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X, y)


y_pred = regressor.predict(X)
print("Predicted salaries:", y_pred)

# Example: predict salary for a new employee at level 6.5
level = np.array([[6.5]])
predicted_salary = regressor.predict(level)
print(f"Predicted Salary for level {level[0][0]}: {predicted_salary[0]}")
X_grid = np.arange(min(X.values), max(X.values)+0.01, 0.01)  # High-resolution for smoother curve
X_grid = X_grid.reshape(-1, 1)

plt.scatter(X, y, color='red', label='Actual Salary')
plt.plot(X_grid, regressor.predict(X_grid), color='blue', label='Decision Tree Prediction')
plt.title('Decision Tree Regression: Level vs Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.legend()
plt.show()

```

## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)
<img width="900" height="72" alt="image" src="https://github.com/user-attachments/assets/1434c935-9566-4eef-a8e7-73eeae2b07b9" />
<img width="748" height="605" alt="image" src="https://github.com/user-attachments/assets/0e349f72-fbca-4f98-b1e5-8daf1403fe61" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
