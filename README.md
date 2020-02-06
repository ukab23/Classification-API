# Classification-API
This code will classify the Titanic passengers on the basis of their Age, Sex, Passenger class and tell if that person would have survived or not.
Used three different classification techniques to compare the accuracy of different training model for this perticular data.

Sample Input for postman:

[
 {"Age":32,"Sex":"female", "Pclass": 1, "Embarked":"Q"}	
]

Sample output:

{
  "LR prediction": "[1]",
  "NB Prediction": "[1]",
  "SVM Prediction": "[1]"
}
