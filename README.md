# F1 Race Prediction Dashboard

## Overview

This project is a machine learning-driven web application designed to predict Formula 1 race results based on historical race data and weather conditions. It integrates a Python-based backend with a React frontend to provide dynamic, accurate race predictions.

## Features

- **Race Predictions**: Uses a Gradient Boosting model trained on historical F1 race and weather data.
- **Interactive Dashboard**: React frontend visualizes predictions, qualifying vs. race pace comparisons, and driver standings across multiple Grand Prix events.
- **Real-time API**: Flask backend serves predictions, qualifying data management, and Grand Prix information via RESTful APIs.

## Technologies Used

- **Backend**: Python, Flask, FastF1, scikit-learn, pandas, NumPy
- **Frontend**: React, Recharts, Tailwind CSS

## APIs

- `/api/predict`: Predict race outcomes based on qualifying data and weather.
- `/api/qualifying`: Manage qualifying data.
- `/api/gp-list`: Retrieve available Grand Prix events.

## Setup & Installation

1. **Backend Setup**:
   ```bash
   pip install -r requirements.txt
   python app.py
   ```

2. **Frontend Setup**:
   ```bash
   npm install
   npm start
   ```

## Usage

- Launch the backend server.
- Launch the frontend React application.
- Use the dashboard to add and analyze different Grand Prix predictions interactively.

## Author
Blog : https://medium.com/@science0719/fast-f1-prediction-model-8c19333e424b

Richa Singh © 2025

