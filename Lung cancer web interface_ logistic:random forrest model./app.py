from flask import Flask, render_template, redirect, request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# This code is a web application that predicts the chances of an individual having lung cancer. 
# It uses two machine learning models, logistic regression and random forest classifier to make predictions.
# The user inputs their data, which is then passed to the relevant model for prediction. 
# The predictions are then returned as a string and displayed to the user. 

app = Flask(__name__)


@app.route("/")
def index():
  return render_template("index.html")


@app.route("/logistic")
def input_data_1():
  return render_template("input.html", model="Logistical-Regression")
@app.route("/random")
def input_data_2():
  return render_template("input.html", model="Random-forest-classifier")  


@app.route("/predict")
def predict():
  
  args = request.args

  model = args.get("model")
  
  gender = int(args.get("gender"))
  age = int(args.get("age"))
  smoking = int(args.get("smoking"))
  yellow_fingers = int(args.get("yellow_fingers"))
  anxiety = int(args.get("anxiety"))
  peer_pressure = int(args.get("peer_pressure"))
  chronic_disease = int(args.get("chronic_disease"))
  WHEEZING = int(args.get("WHEEZING"))
  alcohol_consuming = int(args.get("alcohol_consuming"))
  coughing = int(args.get("coughing"))
  shortness_of_breath = int(args.get("shortness_of_breath"))
  swallowing_difficulty = int(args.get("swallowing_difficulty"))
  chest_pain = int(args.get("chest_pain"))


  # create dictionary with variable names as keys and values as values
  data_dict = {
    "gender": [gender],
    "age": [age],
    "smoking": [smoking],
    "yellow_fingers": [yellow_fingers],
    "anxiety": [anxiety],
    "peer_pressure": [peer_pressure],
    "chronic_disease": [chronic_disease],
    "WHEEZING": [WHEEZING],
    "alcohol_consuming": [alcohol_consuming],
    "coughing": [coughing],
    "shortness_of_breath": [shortness_of_breath],
    "swallowing_difficulty": [swallowing_difficulty],
    "chest_pain": [chest_pain]
  }

  # create pandas dataframe from dictionary
  df = pd.DataFrame.from_dict(data_dict)

  # convert pandas dataframe to numpy array
  data = df.to_numpy()


  if model == "Logistical-Regression":
    return logistic_model_predict(data)

  elif model == "Random-forest-classifier":
    return random_forest_model_prediction(data)


  

def logistic_model_predict(user_data):
  path = "https://raw.githubusercontent.com/Princhow/CancerPrediction/main/survey%20lung%20cancer.csv"
  data = pd.read_csv(path)
  
  # Cleaning the data
  # Yes - 2, No - 1, M - 0, F - 3
  New_data = data.replace("YES", 2)
  New_data = New_data.replace("NO", 1)
  New_data = New_data.replace("M", 0)
  New_data = New_data.replace("F", 3)
  New_data2 = New_data.drop('FATIGUE ', axis=1)
  New_data3 = New_data2.drop('ALLERGY ', axis=1)


  # spliting data
  X_train, X_test, Y_train, Y_test = train_test_split(New_data3.drop('LUNG_CANCER', axis=1), New_data3['LUNG_CANCER'])

  
  # Training the data
  Logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
  Logreg.fit(X_train, Y_train)


  # Accuracy of the model
  score = Logreg.score(X_test, Y_test)

 
  
  # Predicting
  pred = Logreg.predict(user_data)

  if pred[0] == 2: # YES
    res = " You have Lung cancer"
  elif pred[0] == 1: # NO
    res = " You don't have Lung cancer"

  return res





def random_forest_model_prediction(user_data):
  path = "https://raw.githubusercontent.com/Princhow/CancerPrediction/main/survey%20lung%20cancer.csv"
  data = pd.read_csv(path)
  
  # Cleaning the data
  # Yes - 2, No - 1, M - 0, F - 3
  New_data = data.replace("YES", 2)
  New_data = New_data.replace("NO", 1)
  New_data = New_data.replace("M", 0)
  New_data = New_data.replace("F", 3)
  New_data.head()
  
  rf = RandomForestClassifier(n_estimators=100, min_samples_split=2, random_state=1)
  train = New_data.iloc[:150,:]
  test = New_data.iloc[150:,:]

  Predictors = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE',	'WHEEZING', 'ALCOHOL CONSUMING', 	'COUGHING', 	'SHORTNESS OF BREATH', 	'SWALLOWING DIFFICULTY', 	'CHEST PAIN']
  
  # fitting the model
  rf.fit(train[Predictors], train['LUNG_CANCER'])

  # Making prediction and calcualting accuracy
  pred = rf.predict(test[Predictors])
  accuracy = accuracy_score(test['LUNG_CANCER'], pred)
  percent = np.round(accuracy * 100, 2)


  Mpred = rf.predict(user_data)



  if Mpred[0] == 2: # YES
    res = "There is " + str(percent) + " percent chance that you have Lung cancer"
  elif Mpred[0] == 1: # NO
    res = "There is " + str(percent) + " chance that you don't have Lung cancer"

  return res





if __name__== '__main__':
  app.run(debug=True, port=8000)


