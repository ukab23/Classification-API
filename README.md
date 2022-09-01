# Classification-API
This code will classify the Titanic passengers on the basis of their Age, Sex, Passenger class and tell if that person would have survived or not.
Used three different classification techniques to compare the accuracy of different training model for this perticular data.

Create Docker Image :- 
docker build . -t classification

Run Docker Image :-
docker run -p 8080:8080 "Image ID or Name"

Sample Input for postman :-
POST
URL :- http://0.0.0.0:8080//predict
Data :- 
[
 {"Age":32,"Sex":"female", "Pclass": 1, "Embarked":"Q"}	
]

Sample output :-

{
  "LR prediction": "[1]",
  "NB Prediction": "[1]",
  "SVM Prediction": "[1]"
}
