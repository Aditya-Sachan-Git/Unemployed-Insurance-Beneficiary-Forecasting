from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from prophet import Prophet
import pickle
import plotly.express as px

app = Flask(__name__)

# Load the trained Prophet model
with open('model.pkl', 'rb') as file:
    model_prophet = pickle.load(file)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/inspect', methods=['GET'])
def inspect():
    return render_template('inspect.html')

@app.route('/output', methods=['GET', 'POST'])
def output():
    # Handle prediction requests
    if request.method == 'POST':
        try:
            # Get user input
            input_date = pd.to_datetime(request.form['input_date'])

            # Create a dataframe for the input date
            future_date = pd.DataFrame({'ds': [input_date]})

            # Make prediction
            forecast = model_prophet.predict(future_date)
            predicted_value = forecast['yhat'].values

            # Plot the forecast using Plotly
            fig = px.line(forecast, x='ds', y='yhat', title='Insurance Forecast')
            graph = fig.to_html(full_html=False)

            return render_template('output.html', prediction=np.round(predicted_value, 0).item(), graph=graph)

        except Exception as e:
            return render_template('output.html', error=f"An error occurred: {str(e)}")
    
    # Handle direct GET requests to /output
    return render_template('output.html', prediction=None, graph=None)

if __name__ == '__main__':
    app.run(debug=True)
