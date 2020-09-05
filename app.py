import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ["Married","Dependents","Education","Self_Employed","ApplicantIncome",
                     "CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History","Property_Area"]
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
    
    if output == 1:
        res_val = "** a higher probalility of getting a Loan**"
    else:
        res_val = "very low changes of getting a Loan"
        
    return render_template('form.html', pred='Applicant has {}'.format(res_val))
if __name__ == "__main__":
    app.run()