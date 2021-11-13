# flask , scikit-learn , pandas , pickle-mixin

from flask import Flask , render_template , request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

data = pd.read_csv("cleaned_data.csv")
pipe = pickle.load(open("RidgeModel.pkl",'rb'))
@app.route('/')
def index():

    locations = sorted(data['location'].unique())
    return render_template('index.html',locations=locations)

@app.route('/predict',methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = float(request.form.get('bath'))
    sqft = float(request.form.get('total_sqft'))

    # print(location,bhk,bath,sqft)

    input = pd.DataFrame([[sqft, bath, location, bhk]], columns=['total_sqft', 'bath', 'location', 'bhk'])
    prediction = pipe.predict(input)[0] * 1e5
    output = str(np.round(prediction, 2))


    # return str(np.round(prediction),2)
    return render_template('index.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True, port=5001)
