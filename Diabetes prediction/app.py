import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request

# Load the diabetes dataset
data = pd.read_csv('diabetes.csv')

# Prepare the data
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Create a Flask app instance
app = Flask(__name__)

# Define a route for the home page
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        features = [float(request.form['pregnancies']),
                    float(request.form['glucose']),
                    float(request.form['blood_pressure']),
                    float(request.form['skin_thickness']),
                    float(request.form['insulin']),
                    float(request.form['bmi']),
                    float(request.form['diabetes_pedigree_function']),
                    float(request.form['age'])]

        # Make a prediction
        prediction = model.predict([features])

        # Convert the prediction to a meaningful result
        if prediction == 0:
            result = "Not Diabetic"
        else:
            result = "Diabetic"

        return render_template('result.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
