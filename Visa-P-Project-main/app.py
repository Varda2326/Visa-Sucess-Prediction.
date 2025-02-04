
from flask import Flask, render_template, request
import pickle
import joblib
import numpy as np

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('visa_prediction_model_perplexity.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/interview_questions')
def interview_questions():
    return render_template('interview_questions.html')

@app.route('/additional_info')
def additional_info():
    return render_template('additional_info.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from form data
    features = [float(request.form['visaType']),
                int(request.form['purpose']),
                int(request.form['job_offer']),
                int(request.form['salary']),
                int(request.form['passport_validity']),
                int(request.form['education_level']),
                int(request.form['experience']),
                int(request.form['shortage_occupation']),
                int(request.form['family_in_country']),
                int(request.form['international_travel']),
                int(request.form['criminal_record']),
                int(request.form['financial_support']),
                int(request.form['cultural_awareness']),
                int(request.form['career_goals']),
                int(request.form['job_skills'])
                ]
    

    # Reshape features for prediction
    features_array = np.array(features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features_array)
    prediction_prob = model.predict_proba(features_array).max() * 100

    # Convert prediction to human-readable format (0 or 1)
    result = "Success" if prediction[0] == 1 else "Failure"


    return render_template('results.html',result=result,prediction_prob=prediction_prob)


@app.route('/improvements')
def improvements():
    return render_template('improvements.html')

    
    
if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0')

