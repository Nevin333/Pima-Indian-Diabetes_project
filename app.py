#import relevant libraries for flask , html rendering, and loading the ml model
import pickle
from flask import Flask ,request,url_for,redirect,render_template
import pandas as pd
import joblib

app=Flask(__name__) 
#model = pickle.load(open("random.pkl","rb)
#scale = pickle.load(open("scale.pkl","rb"))
model = joblib.load("random.pkl")
scale = joblib.load("scale.pkl")

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])
def predict():

    Pregnancies = request.form['1']
    glucose = request.form['2']
    bloodPressure = request.form['3']
    skinThickness = request.form['4']
    insullin = request.form['5']
    bmi = request.form['6']
    dpf= request.form['7']
    age= request.form['8']

    rowDF=pd.DataFrame([pd.Series([Pregnancies,glucose,bloodPressure,skinThickness,insullin,bmi,dpf,age])])
    rowDF_new=pd.DataFrame(scale.transform(rowDF))
    print(rowDF_new)

    #model prediction
    prediction=model.predict_proba(rowDF_new)
    print(f"the prediction value is :{prediction[0][1]}")

    if prediction[0][1] >= 0.5:
         valpred=round(prediction[0][1],3)
         return render_template('result.html',pred=f'you have a chance of havung diabetics.\n\n probability of you being a diabetics is {valpred*100}%.\n\nAdvice : Excercise Regularly')
    else:
         valpred=round(prediction[0][0],3)
         return render_template('result.html',pred=f'CONGRATULATIONS!!!,YOU ARE IN SAFE ZONE.\n\n probability of you being a non-diabetics is {valpred*100}%.\n\nAdvice : Excercise Regularly and Maintain like this!!!')

if __name__ == "__main__":
    app.run(debug=True)






     






