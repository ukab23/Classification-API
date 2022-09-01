from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
import os
from test_ML import CreateModel

# Your API definition
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if lr and svc:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)

            LR_prediction = list(lr.predict(query))
            SVC_prediction = list(svc.predict(query))
            NB_prediction = list(nb.predict(query))
            return jsonify({'LR prediction': str(LR_prediction), 'SVM Prediction': str(SVC_prediction), 'NB Prediction': str(NB_prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 8080 # If you don't provide any port the port will be set to 12345
 
    path ='models' 
    directory= os.listdir(path) 
    if len(directory) == 0: 
        create = CreateModel()
        create.train()
        lr = joblib.load("models/lr_model.pkl") # Load "model.pkl"
        svc = joblib.load("models/svc_model.pkl")
        nb = joblib.load("models/nb_model.pkl")

    else: 
        lr = joblib.load("models/lr_model.pkl") # Load "model.pkl"
        svc = joblib.load("models/svc_model.pkl")
        nb = joblib.load("models/nb_model.pkl")

    print('Model loaded')
    model_columns = joblib.load("models/model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')

    app.run(host="0.0.0.0", port=port, debug=True)
