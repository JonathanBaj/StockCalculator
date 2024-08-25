import tkinter as tk
from tkinter import font, PhotoImage
from datetime import datetime, timedelta
import os

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg




def get_data_and_predict():
    # Gets the data from the text boxes
    stock_symbol = ticker_e.get("1.0", tk.END).strip().upper()
    start_date = date_e1.get("1.0", tk.END).strip()
    end_date = date_e2.get("1.0", tk.END).strip()
    num_days = int(days_e.get("1.0", tk.END).strip())

    # Download the stock data
    data = yf.download(stock_symbol, start=start_date, end=end_date)

    # Only use the close column
    data = data[["Close"]].copy()

    # Create the predictions column
    data["Prediction"] = data["Close"].shift(-1)

    # Delete NaN row in Prediction column
    data = data[:-1]

    # Make x and y labels
    X = np.array(data.drop(columns=["Prediction"]))
    y = np.array(data["Prediction"])

    # Splits the data into training and testing sets (weird)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the model and train it
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Testing the model
    predictions = model.predict(X_test)

    # Function that predicts future stock prices
    def predict_future_prices(model, last_price, n_days):
        future_predictions = []
        current_price = last_price
        for _ in range(n_days):
            current_price = model.predict(np.array([[current_price]]))[0]
            future_predictions.append(current_price)
        return future_predictions

    last_known_price = data["Close"].values[-1]
    future_prices = predict_future_prices(model, last_known_price, num_days)
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date, periods=num_days+1)[1:]

    # Creates a graph diagram and plots data
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data.index, data["Close"], label="Previous Price", color="#4CAF50")
    ax.plot(future_dates, future_prices, label="Predicted Price", color="#ff6c20")
    ax.legend()

    # Puts the graph in the GUI
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().place(x=40, y=320, width=1120, height=540)




root = tk.Tk()
root.title("Stock Predictor")
root.geometry("1200x900")

# Custom image logo
icon_path = r"C:\Users\jonat\Downloads\InfinityWaveLogo.png"
if not os.path.isfile(icon_path):
    print(f"Error: The file '{icon_path}' does not exist.")
else:
    icon = PhotoImage(file=icon_path)
    root.iconphoto(True, icon)

# Background color
root.configure(bg="#e7e7e7")

# Ticker symbol label and text box
ticker_label = tk.Label(root, text="Stock Ticker", font=font.Font(family="Helvetica", size=23), bg="#e7e7e7")
ticker_label.place(x=200, y=45) 

ticker_e = tk.Text(root, font=font.Font(family="Helvetica", size=20), width=6, height=1, bd=2, relief="solid", bg="#e7e7e7")
ticker_e.place(x=240, y=95)

# Date label and text boxes
date_label = tk.Label(root, text="Date", font=font.Font(family="Helvetica", size=23), bg="#e7e7e7")
date_label.place(x=820, y=45) 

date_e1 = tk.Text(root, font=font.Font(family="Helvetica", size=15), width=10, height=1, bd=2, relief="solid", bg="#e7e7e7")
date_e1.insert(tk.END, "2000-01-01")
date_e1.place(x=800, y=95)

date_e2 = tk.Text(root, font=font.Font(family="Helvetica", size=15), width=10, height=1, bd=2, relief="solid", bg="#e7e7e7")
yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
date_e2.insert(tk.END, yesterday)
date_e2.place(x=800, y=130)

# Days counter label and text box
days_label = tk.Label(root, text="Days", font=font.Font(family="Helvetica", size=23), bg="#e7e7e7")
days_label.place(x=1000, y=45) 

days_e = tk.Text(root, font=font.Font(family="Helvetica", size=20), width=6, height=1, bd=2, relief="solid", bg="#e7e7e7")
days_e.place(x=990, y=95)

# Enter button
enter_button = tk.Button(root, text="ENTER", font=font.Font(family="Helvetica", size=18, weight="bold"), width=15, height=2, bg="#4CAF50", relief="solid", bd=2, command=get_data_and_predict)
enter_button.place(x=500, y=195)

# Grey Box  
grey_box = tk.Frame(root, bg="#d2d2d2")
grey_box.place(relx=0.02, rely=0.34, relwidth=.96, relheight=0.63)




root.mainloop()