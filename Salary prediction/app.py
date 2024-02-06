from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.static_folder = 'static'

# Load the CSV data
data = pd.read_csv("Salary_Data.csv")
X = data[["YearsExperience"]]
y = data["Salary"]

# Train the linear regression model
model = LinearRegression()
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    experience = float(request.form['experience'])
    predicted_salary = model.predict([[experience]])
    return f"Predicted Salary: ${predicted_salary[0]:.2f}"  # floating-point number with two decimal places
 
if __name__ == '__main__':
    app.run(debug=True)
