import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("loan_data_set.csv")

le = preprocessing.LabelEncoder()
Gender = le.fit_transform(list(data["Gender"]))
Married = le.fit_transform(list(data["Married"]))
Dependents = le.fit_transform(list(data["Dependents"]))
Education = le.fit_transform(list(data["Education"]))
Self_Employed = le.fit_transform(list(data["Self_Employed"]))
ApplicantIncome = le.fit_transform(list(data["ApplicantIncome"]))
CoapplicantIncome = le.fit_transform(list(data["CoapplicantIncome"]))
Credit_History = le.fit_transform(list(data["Credit_History"]))
Property_Area = le.fit_transform(list(data["Property_Area"]))
Loan_Status = le.fit_transform(list(data["Loan_Status"]))

predict = "Loan_Status"

X = list(zip(Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome,CoapplicantIncome,Credit_History,Property_Area))
y = list(Loan_Status)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

model = KNeighborsClassifier(n_neighbors=7)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)


predicted = model.predict(x_test)
names = ["Y", "N"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 7, True)
    print("N: ", n)
