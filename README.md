# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the necessary modules.
2. Preprocess data by encoding categorical variables and splitting into features and target.
3. Train logistic regression model on the training data.
4. Evaluate model performance using accuracy, confusion matrix, and classification report on the test data.
5. Optionally, make predictions for new samples using the trained model.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: D Vergin Jenifer 
RegisterNumber: 21223240174
import pandas as pd
data = pd.read_csv("/content/Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = (y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:
![image](https://github.com/VerginJenifer/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/136251012/ceb7facc-3a97-40b9-befe-39cc36751c06)
![image](https://github.com/VerginJenifer/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/136251012/22a8cfef-f149-484e-a153-44c34643cd31)
![image](https://github.com/VerginJenifer/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/136251012/56a893d5-fc3f-4b1a-bf0e-3e997e0153f3)
![image](https://github.com/VerginJenifer/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/136251012/61ab3bc4-42da-435c-9a58-a770390b6fcd)
![image](https://github.com/VerginJenifer/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/136251012/e5e36016-a50f-4276-8848-5f6965015255)
![image](https://github.com/VerginJenifer/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/136251012/9ff54d9f-fb8b-45fd-9665-99ab2bbdfb86)
![image](https://github.com/VerginJenifer/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/136251012/931a27d1-2450-4ceb-a643-8103c28af0e0)
![image](https://github.com/VerginJenifer/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/136251012/90f5b30c-11bd-4390-ba31-14f688976735)
![image](https://github.com/VerginJenifer/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/136251012/c10c2b8b-cf0a-4c30-9f7a-e843ea782d09)
![image](https://github.com/VerginJenifer/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/136251012/432e2dd7-9fd0-4069-8274-ad31b2b5e82e)
![image](https://github.com/VerginJenifer/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/136251012/587c96e6-1eaa-412e-a7cd-296798620dc6)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
