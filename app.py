from flask import Flask, request, render_template, jsonify
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = "HUNPCN7PCXOQX6YN"

def fetch_stock_data(ticker):
    try:
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "apikey": ALPHA_VANTAGE_API_KEY,
            "outputsize": "compact"
        }
        response = requests.get(url, params=params)
        data = response.json()

        if "Time Series (Daily)" not in data:
            return None

        # Convert API response to DataFrame
        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
        df = df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume"
        }).astype(float)
        df = df.sort_index()
        return df

    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None

@app.route('/get_stock_data', methods=['GET'])
def get_stock_data():
    ticker = request.args.get('ticker')
    stock_data = fetch_stock_data(ticker)

    if stock_data is None or stock_data.empty:
        return jsonify({"error": "Could not fetch data or invalid ticker"})

    latest_data = stock_data.iloc[-1]

    return jsonify({
        "Open": latest_data['Open'],
        "High": latest_data['High'],
        "Low": latest_data['Low'],
        "Volume": latest_data['Volume']
    })

# Train a simple model based on historical data
def train_model(stock_data):
    stock_data = stock_data.dropna()
    X = stock_data[['Open', 'High', 'Low', 'Volume']]
    y = stock_data['Close']

    model = LinearRegression()
    model.fit(X, y)
    return model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['Ticker']
    open_price = float(request.form['Open'])
    high_price = float(request.form['High'])
    low_price = float(request.form['Low'])
    volume = float(request.form['Volume'])

    stock_data = fetch_stock_data(ticker)
    if stock_data is None:
        return render_template('result.html', prediction="Error: Could not fetch stock data.")

    model = train_model(stock_data)

    input_data = np.array([[open_price, high_price, low_price, volume]])
    prediction = model.predict(input_data)

    formatted_prediction = "{:.2f}".format(prediction[0])

    return render_template('result.html', prediction=formatted_prediction)

if __name__ == '__main__':
    app.run(debug=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)