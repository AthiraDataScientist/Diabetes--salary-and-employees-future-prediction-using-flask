from flask import Flask, request, render_template,  redirect, url_for
import pickle
import numpy as np
from flask import flash

app = Flask(__name__)
app.secret_key = "Sadeed"


model = pickle.load(open('empenv\model.mkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    Education = request.form['Education']
    JoiningYear = request.form['JoiningYear']
    City = request.form['City']
    PaymentTier = request.form['PaymentTier']
    Age = request.form['Age']
    Gender = request.form['Gender']
    EverBenched = request.form['EverBenched']
    ExperienceInCurrentDomain = request.form['ExperienceInCurrentDomain']
    
    if Education == "" or JoiningYear == "" or City == "" or PaymentTier == "" or Age == "" or Gender == "" or EverBenched == "" or ExperienceInCurrentDomain == "":
                flash("ERROR! Please enter all Values!.")
                return redirect(url_for("index"))
        
            
    arr = np.array([Education,JoiningYear,City,PaymentTier,Age,Gender,EverBenched,ExperienceInCurrentDomain])
    arr = arr.astype(np.float64)
    pred = model.predict([arr])
    
    if pred == 1:
        result = "YES"
    else:
        result = "NO"
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)