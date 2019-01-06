from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np

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
        port = 12345 # If you don't provide any port the port will be set to 12345

    lr = joblib.load("lr_model.pkl") # Load "model.pkl"
    svc = joblib.load("svc_model.pkl")
    nb = joblib.load("nb_model.pkl")

    print ('Model loaded')
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')

    app.run(port=port, debug=True)
