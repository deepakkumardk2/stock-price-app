!pip install tradingview_ta

!pip install streamlit

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import io
import base64
from PIL import Image
import os
from tradingview_ta import TA_Handler, Interval

# Page configuration
st.set_page_config(
    page_title="SmartStock AI Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better visual appeal
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0277BD;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background-color: #f0f8ff;
        border-radius: 8px;
        padding: 15px;
        margin: 5px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0277BD;
    }
    .metric-label {
        font-size: 1rem;
        color: #616161;
        margin-top: 5px;
    }
    .buy-signal {
        color: #4CAF50;
        font-weight: 700;
    }
    .sell-signal {
        color: #F44336;
        font-weight: 700;
    }
    .hold-signal {
        color: #FFC107;
        font-weight: 700;
    }
    .footer {
        text-align: center;
        color: #9E9E9E;
        margin-top: 40px;
        padding: 10px;
        border-top: 1px solid #E0E0E0;
    }
    .stAlert {
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# App header with logo
st.markdown("<h1 class='main-header'>SmartStock AI Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; margin-bottom: 2rem;'>Advanced stock price analysis and prediction platform powered by LSTM deep learning</p>", unsafe_allow_html=True)

# Function to download data as CSV
def get_csv_download_link(df, filename="stock_data.csv"):
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href

# Function to download figure as image
def get_fig_download_link(fig, filename="chart.png", format="png"):
    buf = io.BytesIO()
    fig.write_image(buf, format=format)
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:image/{format};base64,{b64}" download="{filename}">Download {format.upper()}</a>'
    return href

# Cache for stock data fetching to improve performance
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# Function to calculate technical indicators
def calculate_indicators(data):
    # Copy the dataframe to avoid modifying the original
    df = data.copy()
    
    # Moving Averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA100'] = df['Close'].rolling(window=100).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['20MA'] = df['Close'].rolling(window=20).mean()
    df['20STD'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['20MA'] + (df['20STD'] * 2)
    df['Lower_Band'] = df['20MA'] - (df['20STD'] * 2)
    
    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # ADX (Average Directional Index)
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    # OBV (On-Balance Volume)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['%D'] = df['%K'].rolling(window=3).mean()
    
    # Price Rate of Change (ROC)
    df['ROC'] = df['Close'].pct_change(periods=12) * 100
    
    # Fibonacci levels (for last 6 months)
    recent_data = df.tail(180)
    if not recent_data.empty:
        recent_high = recent_data['High'].max()
        recent_low = recent_data['Low'].min()
        diff = recent_high - recent_low
        df['Fib_0'] = recent_low
        df['Fib_23.6'] = recent_low + 0.236 * diff
        df['Fib_38.2'] = recent_low + 0.382 * diff
        df['Fib_50'] = recent_low + 0.5 * diff
        df['Fib_61.8'] = recent_low + 0.618 * diff
        df['Fib_100'] = recent_high
    
    # Volatility (using standard deviation of returns)
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility_30d'] = df['Daily_Return'].rolling(window=30).std() * np.sqrt(252)  # Annualized
    
    # Average up/down days
    df['Up_Day'] = np.where(df['Close'] > df['Close'].shift(1), 1, 0)
    df['Down_Day'] = np.where(df['Close'] < df['Close'].shift(1), 1, 0)
    df['Up_Day_Ratio_20d'] = df['Up_Day'].rolling(window=20).mean()
    
    # Generate trading signals
    df['Signal'] = 0  # 0: Hold, 1: Buy, -1: Sell
    
    # Simple strategy: Buy when price crosses above 50-day MA, Sell when it crosses below
    df.loc[df['Close'] > df['MA50'], 'Signal'] = 1
    df.loc[df['Close'] < df['MA50'], 'Signal'] = -1
    
    # Add RSI conditions
    df.loc[df['RSI'] < 30, 'Signal'] = 1  # Oversold - Buy signal
    df.loc[df['RSI'] > 70, 'Signal'] = -1  # Overbought - Sell signal
    
    # MACD crossover
    df.loc[(df['MACD'] > df['Signal_Line']) & (df['MACD'].shift(1) <= df['Signal_Line'].shift(1)), 'Signal'] = 1
    df.loc[(df['MACD'] < df['Signal_Line']) & (df['MACD'].shift(1) >= df['Signal_Line'].shift(1)), 'Signal'] = -1
    
    return df

# Create a prediction model
def create_lstm_model(look_back, features=1):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(look_back, features)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Prepare data for LSTM model
def prepare_lstm_data(data, look_back=60, future_days=30, train_split=0.8):
    df = data.copy()
    
    # Select features for model (here we're just using 'Close' price)
    dataset = df['Close'].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    # Create training and test datasets
    train_size = int(len(scaled_data) * train_split)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - look_back:]
    
    # Prepare X and y for training
    X_train, y_train = [], []
    for i in range(look_back, len(train_data)):
        X_train.append(train_data[i-look_back:i, 0])
        y_train.append(train_data[i, 0])
    
    # Prepare X and y for testing
    X_test, y_test = [], []
    for i in range(look_back, len(test_data)):
        X_test.append(test_data[i-look_back:i, 0])
        y_test.append(test_data[i, 0])
    
    # Convert to numpy arrays and reshape for LSTM
    X_train = np.array(X_train).reshape(-1, look_back, 1)
    y_train = np.array(y_train)
    X_test = np.array(X_test).reshape(-1, look_back, 1)
    y_test = np.array(y_test)
    
    # Get the last sequence for future prediction
    last_sequence = scaled_data[-look_back:].reshape(1, look_back, 1)
    
    return X_train, y_train, X_test, y_test, last_sequence, scaler

# Train the LSTM model and make predictions
def train_and_predict(X_train, y_train, X_test, last_sequence, scaler, look_back=60, epochs=50, batch_size=32, future_days=30):
    # Create and train the model
    model = create_lstm_model(look_back)
    
    # Add progress bar for training
    with st.spinner("Training LSTM model... This may take a few minutes."):
        progress_bar = st.progress(0)
        
        class CustomCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
        
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                  verbose=0, callbacks=[CustomCallback()], 
                  validation_split=0.1)
    
    # Make predictions on test data
    test_predictions = model.predict(X_test)
    test_predictions = scaler.inverse_transform(test_predictions)
    
    # Predict future values
    future_predictions = []
    current_sequence = last_sequence
    
    for _ in range(future_days):
        future_pred = model.predict(current_sequence)[0]
        future_predictions.append(future_pred)
        
        # Update sequence for next prediction
        current_sequence = np.append(current_sequence[:, 1:, :], 
                                    [[future_pred]], 
                                    axis=1)
    
    future_predictions = np.array(future_predictions)
    future_predictions = scaler.inverse_transform(future_predictions)
    
    return model, test_predictions, future_predictions

# Calculate performance metrics
def calculate_performance_metrics(y_true, y_pred):
    metrics = {}
    metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    metrics['MAPE'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    metrics['R2'] = r2_score(y_true, y_pred)
    return metrics

# Function to generate buy/sell signals
def generate_trading_signals(data):
    df = data.copy()
    
    # Determine buy/sell signals based on multiple factors
    buy_signals = pd.Series(np.nan, index=df.index)
    sell_signals = pd.Series(np.nan, index=df.index)
    
    # RSI signals
    buy_signals[(df['RSI'] < 30) & (df['RSI'].shift(1) >= 30)] = df['Close']
    sell_signals[(df['RSI'] > 70) & (df['RSI'].shift(1) <= 70)] = df['Close']
    
    # MACD crossover signals
    buy_signals[(df['MACD'] > df['Signal_Line']) & (df['MACD'].shift(1) <= df['Signal_Line'].shift(1))] = df['Close']
    sell_signals[(df['MACD'] < df['Signal_Line']) & (df['MACD'].shift(1) >= df['Signal_Line'].shift(1))] = df['Close']
    
    # Moving average crossover signals
    buy_signals[(df['MA20'] > df['MA50']) & (df['MA20'].shift(1) <= df['MA50'].shift(1))] = df['Close']
    sell_signals[(df['MA20'] < df['MA50']) & (df['MA20'].shift(1) >= df['MA50'].shift(1))] = df['Close']
    
    # Bollinger Bands signals
    buy_signals[df['Close'] < df['Lower_Band']] = df['Close']
    sell_signals[df['Close'] > df['Upper_Band']] = df['Close']
    
    return buy_signals, sell_signals

# Function to simulate backtesting
def run_backtest(data, initial_capital=10000.0):
    df = data.copy()
    
    # Add buy/sell signals
    buy_signals, sell_signals = generate_trading_signals(df)
    
    # Initialize portfolio and positions
    portfolio = pd.DataFrame(index=df.index)
    portfolio['Position'] = 0  # 0: No position, 1: Long
    portfolio['Price'] = df['Close']
    portfolio['Cash'] = initial_capital
    portfolio['Holdings'] = 0.0
    portfolio['Total'] = portfolio['Cash']
    
    # Simulate trading
    position = 0
    for i in range(1, len(portfolio)):
        if not pd.isna(buy_signals[i-1]) and position == 0:
            # Buy with all available cash
            shares_bought = portfolio['Cash'][i-1] / portfolio['Price'][i]
            portfolio.loc[portfolio.index[i], 'Holdings'] = shares_bought * portfolio['Price'][i]
            portfolio.loc[portfolio.index[i], 'Cash'] = portfolio['Cash'][i-1] - shares_bought * portfolio['Price'][i]
            portfolio.loc[portfolio.index[i], 'Position'] = 1
            position = 1
        elif not pd.isna(sell_signals[i-1]) and position == 1:
            # Sell all shares
            portfolio.loc[portfolio.index[i], 'Cash'] = portfolio['Cash'][i-1] + portfolio['Holdings'][i-1]
            portfolio.loc[portfolio.index[i], 'Holdings'] = 0.0
            portfolio.loc[portfolio.index[i], 'Position'] = 0
            position = 0
        else:
            # Hold the position
            portfolio.loc[portfolio.index[i], 'Position'] = position
            portfolio.loc[portfolio.index[i], 'Cash'] = portfolio['Cash'][i-1]
            if position == 1:
                shares = portfolio['Holdings'][i-1] / portfolio['Price'][i-1]
                portfolio.loc[portfolio.index[i], 'Holdings'] = shares * portfolio['Price'][i]
            else:
                portfolio.loc[portfolio.index[i], 'Holdings'] = 0.0
            
    # Calculate total portfolio value
    portfolio['Total'] = portfolio['Cash'] + portfolio['Holdings']
    
    # Calculate daily returns
    portfolio['Returns'] = portfolio['Total'].pct_change() * 100
    
    # Calculate cumulative returns
    portfolio['Cum_Returns'] = (portfolio['Total'] / initial_capital - 1) * 100
    
    # Calculate benchmark returns (buy and hold)
    portfolio['Benchmark'] = initial_capital * (df['Close'] / df['Close'][0])
    portfolio['Benchmark_Returns'] = portfolio['Benchmark'].pct_change() * 100
    portfolio['Benchmark_Cum_Returns'] = (portfolio['Benchmark'] / initial_capital - 1) * 100
    
    return portfolio

# Function to get stock recommendations from TradingView
@st.cache_data(ttl=3600)
def get_tradingview_analysis(ticker, screener="india", exchange="NSE"):
    try:
        # Extract symbol without exchange info
        symbol = ticker.split('.')[0] if '.' in ticker else ticker
        
        handler = TA_Handler(
            symbol=symbol,
            screener=screener,
            exchange=exchange,
            interval=Interval.INTERVAL_1_DAY
        )
        analysis = handler.get_analysis()
        
        recommendations = {
            "RECOMMENDATION": analysis.summary["RECOMMENDATION"],
            "BUY": analysis.summary["BUY"],
            "SELL": analysis.summary["SELL"],
            "NEUTRAL": analysis.summary["NEUTRAL"],
            "OSCILLATORS": analysis.oscillators["RECOMMENDATION"],
            "MOVING_AVERAGES": analysis.moving_averages["RECOMMENDATION"]
        }
        
        return recommendations
    except Exception as e:
        st.warning(f"Could not fetch TradingView analysis: {e}")
        return None

# Main application layout
def main():
    # Sidebar section
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1689/1689198.png", width=100)
    st.sidebar.title("Configuration")
    
    # Stock selection section
    st.sidebar.subheader("Stock Selection")
    default_stocks = {"Reliance": "RELIANCE.NS", "TCS": "TCS.NS", "HDFC Bank": "HDFCBANK.NS", "Infosys": "INFY.NS", "ITC": "ITC.NS"}
    
    # Allow user to select predefined stocks or enter custom stock
    stock_option = st.sidebar.radio("Select Stock Source", ["Popular Indian Stocks", "Enter Custom Symbol"])
    
    if stock_option == "Popular Indian Stocks":
        selected_stock = st.sidebar.selectbox("Choose a stock", list(default_stocks.keys()))
        stock_symbol = default_stocks[selected_stock]
    else:
        stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., RELIANCE.NS for NSE)", "RELIANCE.NS")
    
    # Date range selection
    st.sidebar.subheader("Date Range")
    today = datetime.now().date()
    
    # Allow user to select predefined date ranges or custom dates
    date_option = st.sidebar.radio("Select Date Range", ["1 Month", "3 Months", "6 Months", "1 Year", "5 Years", "Custom"])
    
    if date_option == "1 Month":
        start_date = today - timedelta(days=30)
    elif date_option == "3 Months":
        start_date = today - timedelta(days=90)
    elif date_option == "6 Months":
        start_date = today - timedelta(days=180)
    elif date_option == "1 Year":
        start_date = today - timedelta(days=365)
    elif date_option == "5 Years":
        start_date = today - timedelta(days=365*5)
    else:
        start_date = st.sidebar.date_input("Start Date", today - timedelta(days=365))
    
    end_date = st.sidebar.date_input("End Date", today)
    
    # Model configuration section
    st.sidebar.subheader("Model Configuration")
    with st.sidebar.expander("Advanced Settings"):
        look_back = st.slider("Look Back Window", min_value=10, max_value=120, value=60)
        future_days = st.slider("Future Prediction Days", min_value=1, max_value=90, value=30)
        epochs = st.slider("Training Epochs", min_value=10, max_value=100, value=50)
        batch_size = st.select_slider("Batch Size", options=[8, 16, 32, 64, 128], value=32)
        train_test_split = st.slider("Train-Test Split", min_value=0.5, max_value=0.9, value=0.8, step=0.05)
    
    # Backtesting configuration
    st.sidebar.subheader("Backtesting")
    initial_capital = st.sidebar.number_input("Initial Capital (₹)", min_value=1000, max_value=10000000, value=100000, step=10000)
    
    # Fetch data
    with st.spinner(f"Fetching data for {stock_symbol}..."):
        data = fetch_stock_data(stock_symbol, start_date, end_date)
    
    # Check if data is available
    if data is None or data.empty:
        st.error(f"Could not fetch data for {stock_symbol}. Please check the stock symbol and try again.")
        return
    
    # Calculate indicators
    with st.spinner("Calculating technical indicators..."):
        stock_data = calculate_indicators(data)
    
    # Get TradingView recommendations
    tradingview_data = get_tradingview_analysis(stock_symbol)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📈 Technical Analysis", "🔮 Predictions", "📉 Backtesting", "📝 Reports"])
    
    # Tab 1: Overview
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("<h2 class='sub-header'>Stock Price Chart</h2>", unsafe_allow_html=True)
            
            # Create interactive candlestick chart with Plotly
            fig = go.Figure()
            
            # Add candlestick chart
            fig.add_trace(go.Candlestick(
                x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name="OHLC"
            ))
            
            # Add volume as a bar chart at the bottom
            fig.add_trace(go.Bar(
                x=stock_data.index,
                y=stock_data['Volume'],
                name="Volume",
                marker=dict(color="rgba(0, 0, 255, 0.3)"),
                opacity=0.3,
                yaxis="y2"
            ))
            
            # Customize layout
            fig.update_layout(
                title=f"{stock_symbol} Stock Price",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                height=600,
                xaxis_rangeslider_visible=True,
                yaxis2=dict(
                    title="Volume",
                    overlaying="y",
                    side="right",
                    showgrid=False
                ),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Download option
            st.markdown(get_fig_download_link(fig, f"{stock_symbol}_chart.png"), unsafe_allow_html=True)
        
        with col2:
            # Current stock information card
            with st.container():
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                latest_data = stock_data.iloc[-1]
                
                st.markdown(f"<h3 style='text-align: center;'>{stock_symbol}</h3>", unsafe_allow_html=True)
                
                # Price and change
                current_price = latest_data['Close']
                prev_price = stock_data.iloc[-2]['Close']
                price_change = current_price - prev_price
                price_change_pct = (price_change / prev_price) * 100
                
                change_color = "green" if price_change >= 0 else "red"
                change_symbol = "▲" if price_change >= 0 else "▼"
                
                st.markdown(f"""
                <div style='text-align: center;'>
                    <span style='font-size: 2rem; font-weight: bold;'>₹{current_price:.2f}</span>
                    <span style='font-size: 1.2rem; color: {change_color}; margin-left: 10px;'>
                        {change_symbol} ₹{abs(price_change):.2f} ({abs(price_change_pct):.2f}%)
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                # Trading info
                col1a, col2a = st.columns(2)
                
                with col1a:
                    st.markdown(f"""
                    <div style='text-align: center;'>
                        <p style='margin: 5px;'><b>Open</b>: ₹{latest_data['Open']:.2f}</p>
                        <p style='margin: 5px;'><b>High</b>: ₹{latest_data['High']:.2f}</p>
                        <p style='margin: 5px;'><b>Low</b>: ₹{latest_data['Low']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2a:
                    st.markdown(f"""
                    <div style='text-align: center;'>
                        <p style='margin: 5px;'><b>Volume</b>: {latest_data['Volume']:,.0f}</p>
                        <p style='margin: 5px;'><b>52W High</b>: ₹{stock_data['High'].max():.2f}</p>
                        <p style='margin: 5px;'><b>52W Low</b>: ₹{stock_data['Low'].min():.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # TradingView recommendation
            if tradingview_data:
                with st.container():
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("<h4 style='text-align: center;'>TradingView Analysis</h4>", unsafe_allow_html=True)
                    
                    rec = tradingview_data["RECOMMENDATION"]
                    rec_color = "green" if "BUY" in rec else "red" if "SELL" in rec else "orange"
                    
                    st.markdown(f"""
                    <div style='text-align: center;'>
                        <h2 style='color: {rec_color};'>{rec}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1b, col2b, col3b = st.columns(3)
                    
                    with col1b:
                        st.markdown(f"""
                        <div class='metric-card' style='background-color: rgba(0, 128, 0, 0.1);'>
                            <div class='metric-value'>{tradingview_data['BUY']}</div>
                            <div class='metric-label'>BUY</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2b:
                        st.markdown(f"""
                                               <div class='metric-card' style='background-color: rgba(255, 165, 0, 0.1);'>
                            <div class='metric-value'>{tradingview_data['NEUTRAL']}</div>
                            <div class='metric-label'>NEUTRAL</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3b:
                        st.markdown(f"""
                        <div class='metric-card' style='background-color: rgba(255, 0, 0, 0.1);'>
                            <div class='metric-value'>{tradingview_data['SELL']}</div>
                            <div class='metric-label'>SELL</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style='margin-top: 15px; text-align: center; font-size: 0.9rem;'>
                        <p>Moving Averages: <b>{tradingview_data['MOVING_AVERAGES']}</b></p>
                        <p>Oscillators: <b>{tradingview_data['OSCILLATORS']}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # Stock basic metrics
            with st.container():
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h4 style='text-align: center;'>Key Metrics</h4>", unsafe_allow_html=True)
                
                # Calculate volatility
                volatility = stock_data['Daily_Return'].std() * np.sqrt(252) * 100  # Annualized and as percentage
                
                # Calculate beta (assuming market is NIFTY 50)
                try:
                    market_data = fetch_stock_data("^NSEI", start_date, end_date)
                    if market_data is not None:
                        market_returns = market_data['Close'].pct_change()
                        stock_returns = stock_data['Close'].pct_change()
                        
                        # Remove NaN values
                        clean_data = pd.DataFrame({
                            'stock': stock_returns,
                            'market': market_returns
                        }).dropna()
                        
                        covariance = clean_data['stock'].cov(clean_data['market'])
                        market_variance = clean_data['market'].var()
                        beta = covariance / market_variance
                    else:
                        beta = "N/A"
                except:
                    beta = "N/A"
                
                # Calculate Sharpe ratio (using risk-free rate of 5.5% for India)
                risk_free_rate = 0.055
                excess_return = stock_data['Daily_Return'].mean() * 252 - risk_free_rate
                sharpe_ratio = excess_return / (stock_data['Daily_Return'].std() * np.sqrt(252))
                
                st.markdown(f"""
                <div style='display: flex; justify-content: space-between; flex-wrap: wrap;'>
                    <div class='metric-card' style='flex: 1; min-width: 100px;'>
                        <div class='metric-value'>{volatility:.2f}%</div>
                        <div class='metric-label'>Volatility</div>
                    </div>
                    <div class='metric-card' style='flex: 1; min-width: 100px;'>
                        <div class='metric-value'>{beta if isinstance(beta, str) else f"{beta:.2f}"}</div>
                        <div class='metric-label'>Beta</div>
                    </div>
                    <div class='metric-card' style='flex: 1; min-width: 100px;'>
                        <div class='metric-value'>{sharpe_ratio:.2f}</div>
                        <div class='metric-label'>Sharpe Ratio</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Download data
            with st.container():
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h4 style='text-align: center;'>Download Data</h4>", unsafe_allow_html=True)
                
                st.markdown(get_csv_download_link(stock_data, f"{stock_symbol}_data.csv"), unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Data table with expander
        with st.expander("View Raw Data"):
            st.dataframe(stock_data.tail(100), use_container_width=True)
    
    # Tab 2: Technical Analysis
    with tab2:
        st.markdown("<h2 class='sub-header'>Technical Indicators</h2>", unsafe_allow_html=True)
        
        # Create tabs for different technical indicators
        ta_tab1, ta_tab2, ta_tab3, ta_tab4, ta_tab5 = st.tabs(["Moving Averages", "MACD & RSI", "Bollinger Bands", "Volume Analysis", "Custom Indicators"])
        
        # Tab 1: Moving Averages
        with ta_tab1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            
            # Plot moving averages
            fig = go.Figure()
            
            # Add closing price
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data['Close'],
                name="Close",
                line=dict(color='blue', width=1)
            ))
            
            # Add different moving averages
            ma_periods = [5, 20, 50, 100, 200]
            ma_colors = ['purple', 'orange', 'red', 'green', 'brown']
            
            for period, color in zip(ma_periods, ma_colors):
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data[f'MA{period}'],
                    name=f"{period}-Day MA",
                    line=dict(color=color, width=1)
                ))
            
            # Customize layout
            fig.update_layout(
                title="Moving Averages",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Moving Average Analysis
            st.markdown("<h4>Moving Average Analysis</h4>", unsafe_allow_html=True)
            
            last_row = stock_data.iloc[-1]
            
            # Check if price is above/below MAs
            ma_analysis = []
            for period in ma_periods:
                if last_row['Close'] > last_row[f'MA{period}']:
                    ma_analysis.append(f"Price is <span style='color:green'>ABOVE</span> {period}-Day MA")
                else:
                    ma_analysis.append(f"Price is <span style='color:red'>BELOW</span> {period}-Day MA")
            
            # Check for golden cross / death cross
            if stock_data['MA20'].iloc[-1] > stock_data['MA50'].iloc[-1] and stock_data['MA20'].iloc[-2] <= stock_data['MA50'].iloc[-2]:
                ma_analysis.append("<span style='color:green; font-weight:bold'>GOLDEN CROSS (20MA crossed above 50MA)</span>")
            elif stock_data['MA20'].iloc[-1] < stock_data['MA50'].iloc[-1] and stock_data['MA20'].iloc[-2] >= stock_data['MA50'].iloc[-2]:
                ma_analysis.append("<span style='color:red; font-weight:bold'>DEATH CROSS (20MA crossed below 50MA)</span>")
            
            # Display MA analysis
            for analysis in ma_analysis:
                st.markdown(f"• {analysis}", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Tab 2: MACD & RSI
        with ta_tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h4>MACD Indicator</h4>", unsafe_allow_html=True)
                
                # Create MACD plot
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.05, 
                                    row_heights=[0.7, 0.3])
                
                # Add closing price to top plot
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Close'],
                    name="Close",
                    line=dict(color='blue', width=1)
                ), row=1, col=1)
                
                # Add MACD line to bottom plot
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['MACD'],
                    name="MACD",
                    line=dict(color='blue', width=1)
                ), row=2, col=1)
                
                # Add signal line to bottom plot
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Signal_Line'],
                    name="Signal Line",
                    line=dict(color='red', width=1)
                ), row=2, col=1)
                
                # Add histogram
                fig.add_trace(go.Bar(
                    x=stock_data.index,
                    y=stock_data['MACD_Histogram'],
                    name="Histogram",
                    marker=dict(
                        color=stock_data['MACD_Histogram'].apply(lambda x: 'green' if x > 0 else 'red')
                    )
                ), row=2, col=1)
                
                # Customize layout
                fig.update_layout(
                    title="MACD Analysis",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # MACD Analysis
                last_row = stock_data.iloc[-1]
                
                if last_row['MACD'] > last_row['Signal_Line']:
                    macd_signal = "BULLISH"
                    macd_color = "green"
                else:
                    macd_signal = "BEARISH"
                    macd_color = "red"
                
                st.markdown(f"<p>Current MACD Signal: <span style='color:{macd_color}; font-weight:bold'>{macd_signal}</span></p>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h4>Relative Strength Index (RSI)</h4>", unsafe_allow_html=True)
                
                # Create RSI plot
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.05, 
                                    row_heights=[0.7, 0.3])
                
                # Add closing price to top plot
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Close'],
                    name="Close",
                    line=dict(color='blue', width=1)
                ), row=1, col=1)
                
                # Add RSI to bottom plot
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['RSI'],
                    name="RSI",
                    line=dict(color='purple', width=1)
                ), row=2, col=1)
                
                # Add overbought/oversold lines
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=[70] * len(stock_data),
                    name="Overbought (70)",
                    line=dict(color='red', width=1, dash="dash")
                ), row=2, col=1)
                
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=[30] * len(stock_data),
                    name="Oversold (30)",
                    line=dict(color='green', width=1, dash="dash")
                ), row=2, col=1)
                
                # Customize layout
                fig.update_layout(
                    title="RSI Analysis",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified"
                )
                
                fig.update_yaxes(range=[0, 100], row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # RSI Analysis
                last_row = stock_data.iloc[-1]
                
                if last_row['RSI'] > 70:
                    rsi_signal = "OVERBOUGHT"
                    rsi_color = "red"
                elif last_row['RSI'] < 30:
                    rsi_signal = "OVERSOLD"
                    rsi_color = "green"
                else:
                    rsi_signal = "NEUTRAL"
                    rsi_color = "gray"
                
                st.markdown(f"<p>Current RSI ({last_row['RSI']:.2f}): <span style='color:{rsi_color}; font-weight:bold'>{rsi_signal}</span></p>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Tab 3: Bollinger Bands
        with ta_tab3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h4>Bollinger Bands</h4>", unsafe_allow_html=True)
            
            # Create Bollinger Bands plot
            fig = go.Figure()
            
            # Add closing price
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data['Close'],
                name="Close",
                line=dict(color='blue', width=1)
            ))
            
            # Add middle band (20-day SMA)
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data['20MA'],
                name="20-day MA",
                line=dict(color='orange', width=1)
            ))
            
            # Add upper band
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data['Upper_Band'],
                name="Upper Band",
                line=dict(color='green', width=1, dash="dash")
            ))
            
            # Add lower band
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data['Lower_Band'],
                name="Lower Band",
                line=dict(color='red', width=1, dash="dash")
            ))
            
            # Fill area between upper and lower bands
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data['Upper_Band'],
                name="Band Range",
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data['Lower_Band'],
                fill='tonexty',
                fillcolor='rgba(0, 100, 80, 0.1)',
                line=dict(width=0),
                showlegend=False
            ))
            
            # Customize layout
            fig.update_layout(
                title="Bollinger Bands",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                height=600,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Bollinger Bands Analysis
            last_row = stock_data.iloc[-1]
            
            bb_width = (last_row['Upper_Band'] - last_row['Lower_Band']) / last_row['20MA']
            
            # Check if price is near/outside bands
            if last_row['Close'] > last_row['Upper_Band']:
                bb_signal = "OVERBOUGHT - Price above upper band"
                bb_color = "red"
            elif last_row['Close'] < last_row['Lower_Band']:
                bb_signal = "OVERSOLD - Price below lower band"
                bb_color = "green"
            elif last_row['Close'] > last_row['20MA']:
                bb_signal = "BULLISH - Price above middle band"
                bb_color = "lightgreen"
            else:
                bb_signal = "BEARISH - Price below middle band"
                bb_color = "pink"
            
            st.markdown(f"<p>Current Bollinger Bands Signal: <span style='color:{bb_color}; font-weight:bold'>{bb_signal}</span></p>", unsafe_allow_html=True)
            st.markdown(f"<p>Bollinger Band Width: {bb_width:.4f} (Higher values indicate higher volatility)</p>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Tab 4: Volume Analysis
        with ta_tab4:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h4>Volume Analysis</h4>", unsafe_allow_html=True)
            
            # Create Volume Analysis plot
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, 
                                row_heights=[0.7, 0.3])
            
            # Add closing price to top plot
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data['Close'],
                name="Close",
                line=dict(color='blue', width=1)
            ), row=1, col=1)
            
            # Add volume bars
            colors = ['green' if cl >= op else 'red' for cl, op in zip(stock_data['Close'], stock_data['Open'])]
            
            fig.add_trace(go.Bar(
                x=stock_data.index,
                y=stock_data['Volume'],
                name="Volume",
                marker=dict(color=colors)
            ), row=2, col=1)
            
            # Add OBV
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data['OBV'],
                name="OBV",
                line=dict(color='purple', width=1)
            ), row=2, col=1)
            
            # Customize layout
            fig.update_layout(
                title="Volume Analysis with On-Balance Volume (OBV)",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                height=600,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Volume Analysis
            avg_volume = stock_data['Volume'].mean()
            last_volume = stock_data['Volume'].iloc[-1]
            volume_change = (last_volume / avg_volume - 1) * 100
            
            if volume_change > 50:
                volume_signal = "SIGNIFICANTLY HIGHER THAN AVERAGE"
                volume_color = "green"
            elif volume_change > 20:
                volume_signal = "HIGHER THAN AVERAGE"
                volume_color = "lightgreen"
            elif volume_change < -50:
                volume_signal = "SIGNIFICANTLY LOWER THAN AVERAGE"
                volume_color = "red"
            elif volume_change < -20:
                volume_signal = "LOWER THAN AVERAGE"
                volume_color = "pink"
            else:
                volume_signal = "NORMAL"
                volume_color = "gray"
            
            st.markdown(f"<p>Average Volume: {avg_volume:,.0f}</p>", unsafe_allow_html=True)
            st.markdown(f"<p>Latest Volume: {last_volume:,.0f}</p>", unsafe_allow_html=True)
            st.markdown(f"<p>Volume is <span style='color:{volume_color}; font-weight:bold'>{volume_signal}</span> ({volume_change:.2f}% from average)</p>", unsafe_allow_html=True)
            
            # OBV Analysis
            obv_5d_change = stock_data['OBV'].iloc[-1] - stock_data['OBV'].iloc[-6]
            
            if obv_5d_change > 0 and stock_data['Close'].iloc[-1] > stock_data['Close'].iloc[-6]:
                obv_signal = "BULLISH - OBV and price increasing"
                obv_color = "green"
            elif obv_5d_change < 0 and stock_data['Close'].iloc[-1] < stock_data['Close'].iloc[-6]:
                obv_signal = "BEARISH - OBV and price decreasing"
                obv_color = "red"
            elif obv_5d_change > 0 and stock_data['Close'].iloc[-1] < stock_data['Close'].iloc[-6]:
                obv_signal = "BULLISH DIVERGENCE - OBV increasing while price decreasing"
                obv_color = "green"
            elif obv_5d_change < 0 and stock_data['Close'].iloc[-1] > stock_data['Close'].iloc[-6]:
                obv_signal = "BEARISH DIVERGENCE - OBV decreasing while price increasing"
                obv_color = "red"
            else:
                obv_signal = "NEUTRAL"
                obv_color = "gray"
            
            st.markdown(f"<p>OBV 5-Day Change: {obv_5d_change:,.0f}</p>", unsafe_allow_html=True)
            st.markdown(f"<p>OBV Signal: <span style='color:{obv_color}; font-weight:bold'>{obv_signal}</span></p>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Tab 5: Custom Indicators
        with ta_tab5:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            
            # Allow user to select indicators to display
            indicators = st.multiselect(
                "Select indicators to display:",
                ["Stochastic Oscillator", "ATR (Average True Range)", "ROC (Rate of Change)", "Fibonacci Retracement"],
                default=["Stochastic Oscillator", "ROC (Rate of Change)"]
            )
            
            if "Stochastic Oscillator" in indicators:
                st.markdown("<h4>Stochastic Oscillator</h4>", unsafe_allow_html=True)
                
                # Create Stochastic Oscillator plot
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.05, 
                                    row_heights=[0.7, 0.3])
                
                # Add closing price to top plot
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Close'],
                    name="Close",
                    line=dict(color='blue', width=1)
                ), row=1, col=1)
                
                # Add %K to bottom plot
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['%K'],
                    name="%K",
                    line=dict(color='blue', width=1)
                ), row=2, col=1)
                
                # Add %D to bottom plot
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['%D'],
                    name="%D",
                    line=dict(color='red', width=1)
                ), row=2, col=1)
                
                # Add overbought/oversold lines
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=[80] * len(stock_data),
                    name="Overbought (80)",
                    line=dict(color='red', width=1, dash="dash")
                ), row=2, col=1)
                
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=[20] * len(stock_data),
                    name="Oversold (20)",
                    line=dict(color='green', width=1, dash="dash")
                ), row=2, col=1)
                
                # Customize layout
                fig.update_layout(
                    title="Stochastic Oscillator",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified"
                )
                
                fig.update_yaxes(range=[0, 100], row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Stochastic Analysis
                last_k = stock_data['%K'].iloc[-1]
                last_d = stock_data['%D'].iloc[-1]
                
                if last_k > 80 and last_d > 80:
                    stoch_signal = "OVERBOUGHT"
                    stoch_color = "red"
                elif last_k < 20 and last_d < 20:
                    stoch_signal = "OVERSOLD"
                    stoch_color = "green"
                elif last_k > last_d and last_k.shift(1) <= last_d.shift(1):
                    stoch_signal = "BULLISH CROSSOVER"
                    stoch_color = "green"
                elif last_k < last_d and last_k.shift(1) >= last_d.shift(1):
                    stoch_signal = "BEARISH CROSSOVER"
                    stoch_color = "red"
                else:
                    stoch_signal = "NEUTRAL"
                    stoch_color = "gray"
                
                st.markdown(f"<p>Stochastic Oscillator Signal: <span style='color:{stoch_color}; font-weight:bold'>{stoch_signal}</span></p>", unsafe_allow_html=True)
            
            if "ATR (Average True Range)" in indicators:
                st.markdown("<h4>Average True Range (ATR)</h4>", unsafe_allow_html=True)
                
                # Create ATR plot
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.05, 
                                    row_heights=[0.7, 0.3])
                
                # Add closing price to top plot
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Close'],
                    name="Close",
                    line=dict(color='blue', width=1)
                ), row=1, col=1)
                
                # Add ATR to bottom plot
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['ATR'],
                    name="ATR (14)",
                    line=dict(color='orange', width=1)
                ), row=2, col=1)
                
                # Customize layout
                fig.update_layout(
                    title="Average True Range (ATR)",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # ATR Analysis
                last_price = stock_data['Close'].iloc[-1]
                last_atr = stock_data['ATR'].iloc[-1]
                atr_percent = (last_atr / last_price) * 100
                
                st.markdown(f"<p>Current ATR: ₹{last_atr:.2f} ({atr_percent:.2f}% of price)</p>", unsafe_allow_html=True)
                
                # Volatility interpretation
                if atr_percent > 3:
                    st.markdown("<p>Volatility is <span style='color:red; font-weight:bold'>HIGH</span></p>", unsafe_allow_html=True)
                elif atr_percent > 1.5:
                    st.markdown("<p>Volatility is <span style='color:orange; font-weight:bold'>MODERATE</span></p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p>Volatility is <span style='color:green; font-weight:bold'>LOW</span></p>", unsafe_allow_html=True)
            
            if "ROC (Rate of Change)" in indicators:
                st.markdown("<h4>Rate of Change (ROC)</h4>", unsafe_allow_html=True)
                
                # Create ROC plot
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.05, 
                                    row_heights=[0.7, 0.3])
                
                # Add closing price to top plot
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Close'],
                    name="Close",
                    line=dict(color='blue', width=1)
                ), row=1, col=1)
                
                # Add ROC to bottom plot
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['ROC'],
                    name="ROC (12)",
                    line=dict(color='purple', width=1)
                ), row=2, col=1)
                
                # Add zero line
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=[0] * len(stock_data),
                    name="Zero Line",
                    line=dict(color='black', width=1, dash="dash")
                ), row=2, col=1)
                
                # Customize layout
                fig.update_layout(
                    title="Rate of Change (ROC)",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # ROC Analysis
                last_roc = stock_data['ROC'].iloc[-1]
                
                if last_roc > 10:
                    roc_signal = "STRONG BULLISH MOMENTUM"
                    roc_color = "green"
                elif last_roc > 0:
                    roc_signal = "POSITIVE MOMENTUM"
                    roc_color = "lightgreen"
                elif last_roc < -10:
                    roc_signal = "STRONG BEARISH MOMENTUM"
                    roc_color = "red"
                else:
                    roc_signal = "NEGATIVE MOMENTUM"
                    roc_color = "pink"
                
                st.markdown(f"<p>Current ROC: {last_roc:.2f}%</p>", unsafe_allow_html=True)
                st.markdown(f"<p>Momentum: <span style='color:{roc_color}; font-weight:bold'>{roc_signal}</span></p>", unsafe_allow_html=True)
            
            if "Fibonacci Retracement" in indicators:
                st.markdown("<h4>Fibonacci Retracement</h4>", unsafe_allow_html=True)
                
                # Create Fibonacci Retracement plot
                fig = go.Figure()
                
                # Add closing price
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Close'],
                    name="Close",
                    line=dict(color='blue', width=1)
                ))
                
                # Add Fibonacci levels
                fib_levels = ['Fib_0', 'Fib_23.6', 'Fib_38.2', 'Fib_50', 'Fib_61.8', 'Fib_100']
                fib_colors = ['green', 'purple', 'blue', 'orange', 'red', 'darkred']
                fib_percentages = ['0%', '23.6%', '38.2%', '50%', '61.8%', '100%']
                
                for level, color, percentage in zip(fib_levels, fib_colors, fib_percentages):
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data[level],
                        name=f"Fib {percentage}",
                        line=dict(color=color, width=1, dash="dash")
                    ))
                
                # Customize layout
                fig.update_layout(
                    title="Fibonacci Retracement",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Fibonacci Analysis
                last_price = stock_data['Close'].iloc[-1]
                
                # Find closest Fibonacci level
                fib_values = [stock_data[level].iloc[-1] for level in fib_levels]
                closest_fib_idx = np.argmin(np.abs(np.array(fib_values) - last_price))
                closest_fib = fib_percentages[closest_fib_idx]
                
                st.markdown(f"<p>Current price is closest to <span style='font-weight:bold'>Fibonacci {closest_fib} level</span></p>", unsafe_allow_html=True)
                
                # Show all Fibonacci levels
                st.markdown("<p>Current Fibonacci Levels:</p>", unsafe_allow_html=True)
                for level, percentage in zip(fib_levels, fib_percentages):
                    st.markdown(f"<p>• Fib {percentage}: ₹{stock_data[level].iloc[-1]:.2f}</p>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Tab 3: Predictions
    with tab3:
        st.markdown("<h2 class='sub-header'>LSTM Prediction Model</h2>", unsafe_allow_html=True)
        
        # Allow user to train the model
        if st.button("Train LSTM Model and Generate Predictions"):
            with st.spinner("Preparing data for LSTM model..."):
                # Prepare data for LSTM
                X_train, y_train, X_test, y_test, last_sequence, scaler = prepare_lstm_data(
                    stock_data, look_back=look_back, future_days=future_days, train_split=train_test_split
                )
                
                # Get test dates and actual values for plotting
                test_start = int(len(stock_data) * train_test_split)
                test_dates = stock_data.index[test_start:test_start+len(y_test)]
                actual_test = stock_data['Close'].values[test_start:test_start+len(y_test)]
                
                # Train model and generate predictions
                model, test_predictions, future_predictions = train_and_predict(
                    X_train, y_train, X_test, last_sequence, scaler, 
                    look_back=look_back, epochs=epochs, batch_size=batch_size, future_days=future_days
                )
                
                # Calculate future dates
                last_date = stock_data.index[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
                
                # Calculate performance metrics
                metrics = calculate_performance_metrics(actual_test, test_predictions.flatten())
                
                # Show prediction plot
                fig = go.Figure()
                
                # Add historical data
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Close'],
                    name="Historical",
                    line=dict(color='blue', width=2)
                ))
                
                # Add test predictions
                fig.add_trace(go.Scatter(
                    x=test_dates,
                    y=test_predictions.flatten(),
                    name="Test Predictions",
                    line=dict(color='orange', width=2)
                ))
                
                # Add future predictions
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_predictions.flatten(),
                    name="Future Predictions",
                    line=dict(color='green', width=3)
                ))
                
                # Highlight the prediction range with a vertical line
                fig.add_vline(
                    x=stock_data.index[-1], line=dict(color="red", width=2, dash="dash"),
                    annotation_text="Prediction Start"
                )
                
                # Customize layout
                fig.update_layout(
                    title="LSTM Stock Price Prediction",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    height=600,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Download option
                st.markdown(get_fig_download_link(fig, f"{stock_symbol}_prediction.png"), unsafe_allow_html=True)
                
                # Show performance metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{metrics['RMSE']:.2f}</div>
                        <div class='metric-label'>RMSE</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{metrics['MAE']:.2f}</div>
                        <div class='metric-label'>MAE</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{metrics['MAPE']:.2f}%</div>
                        <div class='metric-label'>MAPE</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{metrics['R2']:.4f}</div>
                        <div class='metric-label'>R² Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Prediction summary
                st.markdown("<h4>Prediction Summary</h4>", unsafe_allow_html=True)
                
                last_actual = stock_data['Close'].iloc[-1]
                pred_prices = future_predictions.flatten()
                last_pred = pred_prices[-1]
                price_change = last_pred - last_actual
                price_change_pct = (price_change / last_actual) * 100
                
                if price_change > 0:
                    pred_trend = "UPWARD"
                    pred_color = "green"
                else:
                    pred_trend = "DOWNWARD"
                    pred_color = "red"
                
                st.markdown(f"""
                <p>The model predicts a <span style='color: {pred_color}; font-weight: bold;'>{pred_trend}</span> trend over the next {future_days} days.</p>
                <p>Current Price: ₹{last_actual:.2f}</p>
                <p>Predicted Price ({future_days} days later): ₹{last_pred:.2f}</p>
                <p>Expected Change: ₹{price_change:.2f} ({price_change_pct:.2f}%)</p>
                """, unsafe_allow_html=True)
                
                # Create table of predicted prices
                pred_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Price': pred_prices,
                })
                
                st.markdown("<h4>Detailed Price Predictions</h4>", unsafe_allow_html=True)
                st.dataframe(pred_df.set_index('Date'), use_container_width=True)
                
                # Add download link for predictions
                st.markdown(get_csv_download_link(pred_df, f"{stock_symbol}_predictions.csv"), unsafe_allow_html=True)
        else:
            st.info("Click the button above to train the LSTM model and generate predictions. This may take a few minutes.")
            
            # Show explanation of the model
            with st.expander("How does the LSTM prediction model work?"):
                st.markdown("""
                **LSTM (Long Short-Term Memory) Neural Network**
                
                The prediction model uses a deep learning architecture called LSTM, which is specially designed for sequence prediction problems like stock prices. Here's how it works:
                
                1. **Data Preparation**: Historical stock prices are split into input sequences and target values.
                2. **Scaling**: All data is scaled between 0 and 1 to improve training stability.
                3. **Model Architecture**: The model uses:
                    - Two LSTM layers with dropout to prevent overfitting
                    - Dense layers for final prediction
                4. **Training**: The model learns patterns from historical price movements.
                5. **Testing**: Model is evaluated on a test set to measure accuracy.
                6. **Future Predictions**: The trained model then predicts future prices based on the most recent data.
                
                **Technical Details:**
                - Look-back window: {look_back} days (how much historical data is used for each prediction)
                - Training epochs: {epochs} (passes through the entire dataset during training)
                - Train-test split: {train_test_split*100}% training, {(1-train_test_split)*100}% testing
                - Future prediction horizon: {future_days} days
                
                **Note**: Stock market predictions are inherently uncertain. This model provides one possible outcome based on historical patterns, not a guaranteed result.
                """)
    
    # Tab 4: Backtesting
    with tab4:
        st.markdown("<h2 class='sub-header'>Strategy Backtesting</h2>", unsafe_allow_html=True)
        
        # Allow user to run backtest
        if st.button("Run Backtest"):
            with st.spinner("Running backtest simulation..."):
                # Run backtest
                backtest_results = run_backtest(stock_data, initial_capital=initial_capital)
                
                # Plot performance comparison
                fig = go.Figure()
                
                # Add strategy performance
                fig.add_trace(go.Scatter(
                    x=backtest_results.index,
                    y=backtest_results['Total'],
                    name="Trading Strategy",
                    line=dict(color='green', width=2)
                ))
                
                # Add buy-and-hold performance
                fig.add_trace(go.Scatter(
                    x=backtest_results.index,
                    y=backtest_results['Benchmark'],
                    name="Buy and Hold",
                    line=dict(color='blue', width=2)
                ))
                
                # Customize layout
                fig.update_layout(
                    title="Trading Strategy vs. Buy and Hold",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value (₹)",
                    height=600,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Plot returns comparison
                fig2 = go.Figure()
                
                # Add strategy cumulative returns
                fig2.add_trace(go.Scatter(
                    x=backtest_results.index,
                    y=backtest_results['Cum_Returns'],
                    name="Trading Strategy",
                    line=dict(color='green', width=2)
                ))
                
                # Add buy-and-hold cumulative returns
                fig2.add_trace(go.Scatter(
                    x=backtest_results.index,
                    y=backtest_results['Benchmark_Cum_Returns'],
                    name="Buy and Hold",
                    line=dict(color='blue', width=2)
                ))
                
                # Customize layout
                fig2.update_layout(
                    title="Cumulative Returns (%)",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return (%)",
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Display backtest metrics
                final_strategy_value = backtest_results['Total'].iloc[-1]
                final_benchmark_value = backtest_results['Benchmark'].iloc[-1]
                strategy_return = (final_strategy_value / initial_capital - 1) * 100
                benchmark_return = (final_benchmark_value / initial_capital - 1) * 100
                outperformance = strategy_return - benchmark_return
                
                # Calculate additional metrics
                strategy_daily_returns = backtest_results['Returns'].dropna()
                benchmark_daily_returns = backtest_results['Benchmark_Returns'].dropna()
                
                strategy_volatility = strategy_daily_returns.std() * np.sqrt(252) * 100  # Annualized
                benchmark_volatility = benchmark_daily_returns.std() * np.sqrt(252) * 100  # Annualized
                
                # Sharpe ratio (assuming risk-free rate of 5.5%)
                risk_free_rate = 0.055
                strategy_sharpe = (strategy_daily_returns.mean() * 252 - risk_free_rate) / (strategy_daily_returns.std() * np.sqrt(252))
                benchmark_sharpe = (benchmark_daily_returns.mean() * 252 - risk_free_rate) / (benchmark_daily_returns.std() * np.sqrt(252))
                
                # Maximum drawdown
                strategy_cum_returns = (1 + strategy_daily_returns).cumprod()
                benchmark_cum_returns = (1 + benchmark_daily_returns).cumprod()
                
                strategy_running_max = strategy_cum_returns.cummax()
                benchmark_running_max = benchmark_cum_returns.cummax()
                
                strategy_drawdown = (strategy_cum_returns / strategy_running_max - 1) * 100
                benchmark_drawdown = (benchmark_cum_returns / benchmark_running_max - 1) * 100
                
                strategy_max_drawdown = strategy_drawdown.min()
                benchmark_max_drawdown = benchmark_drawdown.min()
                
                # Display results in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<h4>Trading Strategy</h4>", unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>₹{final_strategy_value:,.2f}</div>
                        <div class='metric-label'>Final Value</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{strategy_return:.2f}%</div>
                        <div class='metric-label'>Total Return</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{strategy_volatility:.2f}%</div>
                        <div class='metric-label'>Annualized Volatility</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{strategy_sharpe:.2f}</div>
                        <div class='metric-label'>Sharpe Ratio</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{strategy_max_drawdown:.2f}%</div>
                        <div class='metric-label'>Maximum Drawdown</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<h4>Buy and Hold</h4>", unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>₹{final_benchmark_value:,.2f}</div>
                        <div class='metric-label'>Final Value</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{benchmark_return:.2f}%</div>
                        <div class='metric-label'>Total Return</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{benchmark_volatility:.2f}%</div>
                        <div class='metric-label'>Annualized Volatility</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{benchmark_sharpe:.2f}</div>
                        <div class='metric-label'>Sharpe Ratio</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{benchmark_max_drawdown:.2f}%</div>
                        <div class='metric-label'>Maximum Drawdown</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Performance summary
                st.markdown("<h4>Performance Summary</h4>", unsafe_allow_html=True)
                
                if strategy_return > benchmark_return:
                    outperform_color = "green"
                    outperform_text = "OUTPERFORMED"
                else:
                    outperform_color = "red"
                    outperform_text = "UNDERPERFORMED"
                
                st.markdown(f"""
                <p>The trading strategy <span style='color: {outperform_color}; font-weight: bold;'>{outperform_text}</span> the buy-and-hold strategy by <span style='font-weight: bold;'>{abs(outperformance):.2f}%</span></p>
                """, unsafe_allow_html=True)
                
                # Show trades table
                st.markdown("<h4>Trade History</h4>", unsafe_allow_html=True)
                
                # Extract trade signals
                buy_signals, sell_signals = generate_trading_signals(stock_data)
                
                # Create trade history dataframe
                trades = pd.DataFrame(columns=['Date', 'Type', 'Price'])
                
                # Add buy trades
                buy_dates = buy_signals.dropna().index
                if not buy_dates.empty:
                    buy_df = pd.DataFrame({
                        'Date': buy_dates,
                        'Type': 'BUY',
                        'Price': buy_signals.dropna().values
                    })
                    trades = pd.concat([trades, buy_df])
                
                # Add sell trades
                sell_dates = sell_signals.dropna().index
                if not sell_dates.empty:
                    sell_df = pd.DataFrame({
                        'Date': sell_dates,
                        'Type': 'SELL',
                        'Price': sell_signals.dropna().values
                    })
                    trades = pd.concat([trades, sell_df])
                
                # Sort by date
                trades = trades.sort_values('Date')
                
                # Display trades
                if not trades.empty:
                    st.dataframe(trades.set_index('Date'), use_container_width=True)
                else:
                    st.info("No trades were generated during the backtesting period.")
        else:
            st.info("Click the button above to run the backtesting simulation.")
            
            # Show explanation of the backtest strategy
            with st.expander("How does the backtesting strategy work?"):
                st.markdown("""
                **Backtesting Strategy Logic**
                
                The backtesting system tests a strategy that combines multiple technical indicators:
                
                1. **Entry Signals**:
                   - RSI below 30 (oversold condition)
                   - MACD crosses above Signal Line
                   - 20-day MA crosses above 50-day MA
                   - Price touches or crosses below the lower Bollinger Band
                
                2. **Exit Signals**:
                   - RSI above 70 (overbought condition)
                   - MACD crosses below Signal Line
                   - 20-day MA crosses below 50-day MA
                   - Price touches or crosses above the upper Bollinger Band
                
                3. **Position Sizing**:
                   - Each entry signal uses all available capital to buy shares
                   - Each exit signal sells all shares
                
                4. **Performance Comparison**:
                   - The strategy performance is compared against a simple buy-and-hold approach
                   - Key metrics like total return, volatility, Sharpe ratio, and maximum drawdown are calculated
                
                **Note**: Past performance does not guarantee future results. This backtesting simulation is for educational purposes only.
                """)
    
    # Tab 5: Reports
    with tab5:
        st.markdown("<h2 class='sub-header'>Analysis Reports</h2>", unsafe_allow_html=True)
        
        # Technical Analysis Report
        with st.expander("Technical Analysis Summary", expanded=True):
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            
            # Get latest data for analysis
            latest = stock_data.iloc[-1]
            
            # Trend analysis
            if latest['Close'] > latest['MA50'] and latest['MA20'] > latest['MA50']:
                trend = "BULLISH"
                trend_color = "green"
            elif latest['Close'] < latest['MA50'] and latest['MA20'] < latest['MA50']:
                trend = "BEARISH"
                trend_color = "red"
            else:
                trend = "NEUTRAL"
                trend_color = "orange"
            
            # Momentum analysis
            if latest['RSI'] > 70:
                momentum = "OVERBOUGHT"
                momentum_color = "red"
            elif latest['RSI'] < 30:
                momentum = "OVERSOLD"
                momentum_color = "green"
            else:
                momentum = "NEUTRAL"
                momentum_color = "gray"
            
            # Volatility analysis
            bb_width = (latest['Upper_Band'] - latest['Lower_Band']) / latest['20MA']
            
            if bb_width > 0.1:
                volatility = "HIGH"
                volatility_color = "red"
            elif bb_width > 0.05:
                volatility = "MODERATE"
                volatility_color = "orange"
            else:
                volatility = "LOW"
                volatility_color = "green"
            
            # Volume analysis
            avg_volume = stock_data['Volume'].tail(20).mean()
            if latest['Volume'] > 1.5 * avg_volume:
                volume = "HIGH"
                volume_color = "red"
            elif latest['Volume'] < 0.5 * avg_volume:
                volume = "LOW"
                volume_color = "orange"
            else:
                volume = "AVERAGE"
                volume_color = "gray"
            
            # Overall signal
            signals = []
            
            if trend == "BULLISH":
                signals.append(1)
            elif trend == "BEARISH":
                signals.append(-1)
            else:
                signals.append(0)
                
            if momentum == "OVERSOLD":
                signals.append(1)
            elif momentum == "OVERBOUGHT":
                signals.append(-1)
            else:
                signals.append(0)
                
            if latest['MACD'] > latest['Signal_Line']:
                signals.append(1)
            else:
                signals.append(-1)
                
            if latest['Close'] < latest['Lower_Band']:
                signals.append(1)
            elif latest['Close'] > latest['Upper_Band']:
                signals.append(-1)
            else:
                signals.append(0)
            
            avg_signal = sum(signals) / len(signals)
            
            if avg_signal > 0.3:
                overall = "BUY"
                overall_color = "green"
            elif avg_signal < -0.3:
                overall = "SELL"
                overall_color = "red"
            else:
                overall = "HOLD"
                overall_color = "orange"
            
            # Display summary
            st.markdown(f"""
            <h3 style='text-align: center;'>Technical Analysis Summary for {stock_symbol}</h3>
            <p style='text-align: center;'>Last Updated: {stock_data.index[-1].strftime('%Y-%m-%d')}</p>
            """, unsafe_allow_html=True)
            
            # Create a summary table
            st.markdown(f"""
            <table style='width:100%;'>
                <tr>
                    <th style='padding:8px; border-bottom:1px solid #ddd;'>Indicator</th>
                    <th style='padding:8px; border-bottom:1px solid #ddd;'>Value</th>
                    <th style='padding:8px; border-bottom:1px solid #ddd;'>Signal</th>
                </tr>
                <tr>
                    <td style='padding:8px; border-bottom:1px solid #ddd;'>Trend (50-day MA)</td>
                    <td style='padding:8px; border-bottom:1px solid #ddd;'>{latest['MA50']:.2f}</td>
                    <td style='padding:8px; border-bottom:1px solid #ddd; color:{trend_color}; font-weight:bold;'>{trend}</td>
                </tr>
                <tr>
                    <td style='padding:8px; border-bottom:1px solid #ddd;'>RSI (14-day)</td>
                    <td style='padding:8px; border-bottom:1px solid #ddd;'>{latest['RSI']:.2f}</td>
                    <td style='padding:8px; border-bottom:1px solid #ddd; color:{momentum_color}; font-weight:bold;'>{momentum}</td>
                </tr>
                <tr>
                    <td style='padding:8px; border-bottom:1px solid #ddd;'>MACD</td>
                    <td style='padding:8px; border-bottom:1px solid #ddd;'>{latest['MACD']:.4f}</td>
                    <td style='padding:8px; border-bottom:1px solid #ddd; color:{"green" if latest["MACD"] > latest["Signal_Line"] else "red"}; font-weight:bold;'>
                        {"BULLISH" if latest["MACD"] > latest["Signal_Line"] else "BEARISH"}
                    </td>
                </tr>
                <tr>
                    <td style='padding:8px; border-bottom:1px solid #ddd;'>Bollinger Bands</td>
                    <td style='padding:8px; border-bottom:1px solid #ddd;'>Width: {bb_width:.4f}</td>
                    <td style='padding:8px; border-bottom:1px solid #ddd; color:{volatility_color}; font-weight:bold;'>{volatility} VOLATILITY</td>
                </tr>
                <tr>
                    <td style='padding:8px; border-bottom:1px solid #ddd;'>Volume</td>
                    <td style='padding:8px; border-bottom:1px solid #ddd;'>{latest['Volume']:,.0f}</td>
                    <td style='padding:8px; border-bottom:1px solid #ddd; color:{volume_color}; font-weight:bold;'>{volume}</td>
                </tr>
                <tr>
                    <td style='padding:8px; border-bottom:1px solid #ddd; font-weight:bold;'>OVERALL SIGNAL</td>
                    <td style='padding:8px; border-bottom:1px solid #ddd;'></td>
                    <td style='padding:8px; border-bottom:1px solid #ddd; color:{overall_color}; font-size:1.2em; font-weight:bold;'>{overall}</td>
                </tr>
            </table>
            """, unsafe_allow_html=True)
            
            # Additional insights
            st.markdown("<h4>Key Insights</h4>", unsafe_allow_html=True)
            
            insights = []
            
            # Trend insights
            if trend == "BULLISH":
                insights.append("• The stock is trading above its 50-day moving average, indicating a bullish trend.")
            elif trend == "BEARISH":
                insights.append("• The stock is trading below its 50-day moving average, indicating a bearish trend.")
            
            # RSI insights
            if momentum == "OVERBOUGHT":
                insights.append("• RSI is above 70, suggesting the stock may be overvalued and due for a pullback.")
            elif momentum == "OVERSOLD":
                insights.append("• RSI is below 30, suggesting the stock may be undervalued and due for a rebound.")
            
            # MACD insights
            if latest['MACD'] > latest['Signal_Line'] and latest['MACD'] > 0:
                insights.append("• MACD is above the signal line and positive, confirming bullish momentum.")
            elif latest['MACD'] < latest['Signal_Line'] and latest['MACD'] < 0:
                insights.append("• MACD is below the signal line and negative, confirming bearish momentum.")
            elif latest['MACD'] > latest['Signal_Line'] and latest['MACD'] < 0:
                insights.append("• MACD is above the signal line but still negative, suggesting potential trend reversal.")
            
            # Bollinger Bands insights
            if latest['Close'] > latest['Upper_Band']:
                insights.append("• Price is above the upper Bollinger Band, indicating overbought conditions.")
            elif latest['Close'] < latest['Lower_Band']:
                insights.append("• Price is below the lower Bollinger Band, indicating oversold conditions.")
            
            # Volume insights
            if volume == "HIGH" and trend == "BULLISH":
                insights.append("• High volume with rising price suggests strong buying interest.")
            elif volume == "HIGH" and trend == "BEARISH":
                insights.append("• High volume with falling price suggests strong selling pressure.")
            
            # Display insights
            for insight in insights:
                st.markdown(insight, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Fundamental Analysis Report
        with st.expander("Recent Performance Analysis"):
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            
            # Calculate performance metrics
            start_price = stock_data['Close'].iloc[0]
            end_price = stock_data['Close'].iloc[-1]
            price_change = end_price - start_price
            price_change_pct = (price_change / start_price) * 100
            
            # Calculate returns
            daily_returns = stock_data['Close'].pct_change().dropna()
            
            # Annualized return
            time_period = (stock_data.index[-1] - stock_data.index[0]).days / 365
            annualized_return = (((1 + price_change_pct / 100) ** (1 / time_period)) - 1) * 100
            
            # Volatility
            volatility = daily_returns.std() * np.sqrt(252) * 100
            
            # Calculate max drawdown
            rolling_max = stock_data['Close'].cummax()
            drawdown = (stock_data['Close'] / rolling_max - 1) * 100
            max_drawdown = drawdown.min()
            
            # Display performance summary
            st.markdown(f"""
            <h3 style='text-align: center;'>Performance Analysis for {stock_symbol}</h3>
            <p style='text-align: center;'>Period: {stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')} ({(stock_data.index[-1] - stock_data.index[0]).days} days)</p>
            """, unsafe_allow_html=True)
            
            # Create columns for key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div style='text-align: center;'>
                    <h4>Total Return</h4>
                    <p style='font-size: 1.8rem; font-weight: bold; color: {'green' if price_change_pct > 0 else 'red'};'>
                        {price_change_pct:.2f}%
                    </p>
                    <p>From ₹{start_price:.2f} to ₹{end_price:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style='text-align: center;'>
                    <h4>Annualized Return</h4>
                    <p style='font-size: 1.8rem; font-weight: bold; color: {'green' if annualized_return > 0 else 'red'};'>
                        {annualized_return:.2f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div style='text-align: center;'>
                    <h4>Maximum Drawdown</h4>
                    <p style='font-size: 1.8rem; font-weight: bold; color: red;'>
                        {max_drawdown:.2f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Create additional metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div style='text-align: center;'>
                    <h4>Volatility (Annualized)</h4>
                    <p style='font-size: 1.8rem; font-weight: bold;'>
                        {volatility:.2f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Calculate Sharpe ratio (assuming risk-free rate of 5.5%)
                risk_free_rate = 0.055
                sharpe_ratio = (daily_returns.mean() * 252 - risk_free_rate) / (daily_returns.std() * np.sqrt(252))
                
                st.markdown(f"""
                <div style='text-align: center;'>
                    <h4>Sharpe Ratio</h4>
                    <p style='font-size: 1.8rem; font-weight: bold; color: {'green' if sharpe_ratio > 1 else 'orange' if sharpe_ratio > 0 else 'red'};'>
                        {sharpe_ratio:.2f}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Plot the performance
            st.markdown("<h4>Price Performance</h4>", unsafe_allow_html=True)
            
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data['Close'],
                name="Close Price",
                line=dict(color='blue', width=2)
            ))
            
            # Add trend line (linear regression)
            x = np.array(range(len(stock_data))).reshape(-1, 1)
            y = stock_data['Close'].values
            
            # Use numpy's polyfit to create a linear regression trendline
            z = np.polyfit(x.flatten(), y, 1)
            p = np.poly1d(z)
            
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=p(x.flatten()),
                name=f"Trend (slope: {z[0]:.2f})",
                line=dict(color='red', width=1, dash='dash')
            ))
            
            # Add annotations for start and end points
            fig.add_annotation(
                x=stock_data.index[0],
                y=start_price,
                text=f"₹{start_price:.2f}",
                showarrow=True,
                arrowhead=1
            )
            
            fig.add_annotation(
                x=stock_data.index[-1],
                y=end_price,
                text=f"₹{end_price:.2f}",
                showarrow=True,
                arrowhead=1
            )
            
            # Customize layout
            fig.update_layout(
                title="Price Performance During Selected Period",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance comparison with benchmark (NIFTY 50)
            st.markdown("<h4>Benchmark Comparison (vs. NIFTY 50)</h4>", unsafe_allow_html=True)
            
            try:
                # Fetch benchmark data
                benchmark_data = fetch_stock_data("^NSEI", stock_data.index[0], stock_data.index[-1])
                
                if benchmark_data is not None and not benchmark_data.empty:
                    # Calculate normalized returns (starting at 100)
                    norm_stock = 100 * (stock_data['Close'] / stock_data['Close'].iloc[0])
                    norm_benchmark = 100 * (benchmark_data['Close'] / benchmark_data['Close'].iloc[0])
                    
                    # Create comparison plot
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=norm_stock,
                        name=stock_symbol,
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=benchmark_data.index,
                        y=norm_benchmark,
                        name="NIFTY 50",
                        line=dict(color='green', width=2)
                    ))
                    
                    # Customize layout
                    fig.update_layout(
                        title="Relative Performance (Base: 100)",
                        xaxis_title="Date",
                        yaxis_title="Normalized Value",
                        height=400,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate performance difference
                    stock_return = norm_stock.iloc[-1] - 100
                    benchmark_return = norm_benchmark.iloc[-1] - 100
                    
                    outperformance = stock_return - benchmark_return
                    
                    if outperformance > 0:
                        st.markdown(f"<p>{stock_symbol} <span style='color:green; font-weight:bold;'>OUTPERFORMED</span> NIFTY 50 by {outperformance:.2f} percentage points.</p>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<p>{stock_symbol} <span style='color:red; font-weight:bold;'>UNDERPERFORMED</span> NIFTY 50 by {abs(outperformance):.2f} percentage points.</p>", unsafe_allow_html=True)
                else:
                    st.warning("Could not fetch benchmark data for comparison.")
            except Exception as e:
                st.warning(f"Could not compare with benchmark: {e}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Trading Recommendations Report
        with st.expander("Trading Recommendations"):
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            
            st.markdown("<h3 style='text-align: center;'>Trading Recommendations</h3>", unsafe_allow_html=True)
            
            # Get TradingView recommendations if available
            if tradingview_data:
                rec = tradingview_data["RECOMMENDATION"]
                rec_color = "green" if "BUY" in rec else "red" if "SELL" in rec else "orange"
                
                st.markdown(f"""
                <div style='text-align: center; margin-bottom: 20px;'>
                    <h2 style='color: {rec_color};'>{rec}</h2>
                    <p>Based on TradingView technical analysis</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display more details in a table
                st.markdown(f"""
                <table style='width:100%;'>
                    <tr>
                        <th style='padding:8px; border-bottom:1px solid #ddd;'>Indicator</th>
                        <th style='padding:8px; border-bottom:1px solid #ddd;'>Value</th>
                    </tr>
                    <tr>
                        <td style='padding:8px; border-bottom:1px solid #ddd;'>Oscillators</td>
                        <td style='padding:8px; border-bottom:1px solid #ddd;'>{tradingview_data['OSCILLATORS']}</td>
                    </tr>
                    <tr>
                        <td style='padding:8px; border-bottom:1px solid #ddd;'>Moving Averages</td>
                        <td style='padding:8px; border-bottom:1px solid #ddd;'>{tradingview_data['MOVING_AVERAGES']}</td>
                    </tr>
                    <tr>
                        <td style='padding:8px; border-bottom:1px solid #ddd;'>Buy Signals</td>
                        <td style='padding:8px; border-bottom:1px solid #ddd;'>{tradingview_data['BUY']}</td>
                    </tr>
                    <tr>
                        <td style='padding:8px; border-bottom:1px solid #ddd;'>Neutral Signals</td>
                        <td style='padding:8px; border-bottom:1px solid #ddd;'>{tradingview_data['NEUTRAL']}</td>
                    </tr>
                    <tr>
                        <td style='padding:8px; border-bottom:1px solid #ddd;'>Sell Signals</td>
                        <td style='padding:8px; border-bottom:1px solid #ddd;'>{tradingview_data['SELL']}</td>
                    </tr>
                </table>
                """, unsafe_allow_html=True)
            
            # Technical analysis based recommendations
            st.markdown("<h4>Support and Resistance Levels</h4>", unsafe_allow_html=True)
            
            # Calculate support and resistance levels using recent highs and lows
            recent_data = stock_data.tail(30)
            
            # Find potential support levels (recent lows)
            lows = recent_data['Low'].sort_values()
            supports = []
            
            # Find distinct support levels with minimum 1% difference
            for low in lows:
                if not supports or low < supports[-1] * 0.99:
                    supports.append(low)
                    if len(supports) >= 3:
                        break
            
            # Find potential resistance levels (recent highs)
            highs = recent_data['High'].sort_values(ascending=False)
            resistances = []
            
            # Find distinct resistance levels with minimum 1% difference
            for high in highs:
                if not resistances or high > resistances[-1] * 1.01:
                    resistances.append(high)
                    if len(resistances) >= 3:
                        break
            
            # Display support and resistance levels
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<h5>Support Levels</h5>", unsafe_allow_html=True)
                for i, level in enumerate(supports):
                    st.markdown(f"<p>S{i+1}: ₹{level:.2f}</p>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<h5>Resistance Levels</h5>", unsafe_allow_html=True)
                for i, level in enumerate(resistances):
                    st.markdown(f"<p>R{i+1}: ₹{level:.2f}</p>", unsafe_allow_html=True)
            
            # Current price in relation to support/resistance
            last_price = stock_data['Close'].iloc[-1]
            
            # Find nearest support and resistance
            nearest_support = max([s for s in supports if s <= last_price], default=last_price * 0.95)
            nearest_resistance = min([r for r in resistances if r >= last_price], default=last_price * 1.05)
            
            # Calculate percentage distance to nearest levels
            dist_to_support = (last_price - nearest_support) / last_price * 100
            dist_to_resistance = (nearest_resistance - last_price) / last_price * 100
            
            st.markdown(f"""
            <p>Current price (₹{last_price:.2f}) is:
                <ul>
                    <li>{dist_to_support:.2f}% above nearest support (₹{nearest_support:.2f})</li>
                    <li>{dist_to_resistance:.2f}% below nearest resistance (₹{nearest_resistance:.2f})</li>
                </ul>
            </p>
            """, unsafe_allow_html=True)
            
            # Trading strategy recommendations
            st.markdown("<h4>Trading Strategies</h4>", unsafe_allow_html=True)
            
            # Generate strategy recommendations based on technical analysis
            strategies = []
            
            # Strategy 1: Trend following
            if trend == "BULLISH":
                strategies.append({
                    "name": "Trend Following",
                    "signal": "BUY",
                    "description": "The stock is in an uptrend. Consider buying on dips to the 20-day or 50-day moving average.",
                    "target": f"₹{nearest_resistance:.2f} (nearest resistance)",
                    "stop_loss": f"₹{nearest_support:.2f} (nearest support)",
                    "risk_reward": f"{dist_to_resistance/dist_to_support:.2f}",
                    "color": "green"
                })
            elif trend == "BEARISH":
                strategies.append({
                    "name": "Trend Following",
                    "signal": "SELL",
                    "description": "The stock is in a downtrend. Consider selling on rallies to the 20-day or 50-day moving average.",
                    "target": f"₹{nearest_support:.2f} (nearest support)",
                    "stop_loss": f"₹{nearest_resistance:.2f} (nearest resistance)",
                    "risk_reward": f"{dist_to_support/dist_to_resistance:.2f}",
                    "color": "red"
                })
            
            # Strategy 2: RSI based
            if momentum == "OVERSOLD":
                strategies.append({
                    "name": "RSI Reversal",
                    "signal": "BUY",
                    "description": "The RSI indicates oversold conditions. Consider a counter-trend buy for a short-term bounce.",
                    "target": f"₹{last_price * 1.05:.2f} (5% above current price)",
                    "stop_loss": f"₹{last_price * 0.98:.2f} (2% below current price)",
                    "risk_reward": "2.5",
                    "color": "green"
                })
            elif momentum == "OVERBOUGHT":
                strategies.append({
                    "name": "RSI Reversal",
                    "signal": "SELL",
                    "description": "The RSI indicates overbought conditions. Consider a counter-trend sell for a short-term pullback.",
                    "target": f"₹{last_price * 0.95:.2f} (5% below current price)",
                    "stop_loss": f"₹{last_price * 1.02:.2f} (2% above current price)",
                    "risk_reward": "2.5",
                    "color": "red"
                })
            
            # Strategy 3: Bollinger Bands
            if latest['Close'] < latest['Lower_Band']:
                strategies.append({
                    "name": "Bollinger Band Reversal",
                    "signal": "BUY",
                    "description": "Price is below the lower Bollinger Band, indicating potential overselling. Consider a buy with a target at the middle band.",
                    "target": f"₹{latest['20MA']:.2f} (middle band)",
                    "stop_loss": f"₹{latest['Close'] * 0.98:.2f} (2% below current price)",
                    "risk_reward": f"{(latest['20MA'] - latest['Close']) / (latest['Close'] - latest['Close'] * 0.98):.2f}",
                    "color": "green"
                })
            elif latest['Close'] > latest['Upper_Band']:
                strategies.append({
                    "name": "Bollinger Band Reversal",
                    "signal": "SELL",
                    "description": "Price is above the upper Bollinger Band, indicating potential overbuying. Consider a sell with a target at the middle band.",
                    "target": f"₹{latest['20MA']:.2f} (middle band)",
                    "stop_loss": f"₹{latest['Close'] * 1.02:.2f} (2% above current price)",
                    "risk_reward": f"{(latest['Close'] - latest['20MA']) / (latest['Close'] * 1.02 - latest['Close']):.2f}",
                    "color": "red"
                })
            
            # Display strategies
            if strategies:
                for strategy in strategies:
                    st.markdown(f"""
                    <div style='border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 15px;'>
                        <h5>{strategy["name"]}: <span style='color:{strategy["color"]};'>{strategy["signal"]}</span></h5>
                        <p>{strategy["description"]}</p>
                        <table style='width:100%;'>
                            <tr>
                                <th style='padding:4px; border-bottom:1px solid #ddd;'>Target</th>
                                <th style='padding:4px; border-bottom:1px solid #ddd;'>Stop Loss</th>
                                <th style='padding:4px; border-bottom:1px solid #ddd;'>Risk/Reward</th>
                            </tr>
                            <tr>
                                <td style='padding:4px; border-bottom:1px solid #ddd;'>{strategy["target"]}</td>
                                <td style='padding:4px; border-bottom:1px solid #ddd;'>{strategy["stop_loss"]}</td>
                                <td style='padding:4px; border-bottom:1px solid #ddd;'>{strategy["risk_reward"]}</td>
                            </tr>
                        </table>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No clear trading strategies available with the current market conditions. Consider waiting for clearer signals.")
            
            # Risk disclaimer
            st.markdown("""
            <div style='background-color: #ffffcc; padding: 10px; border-left: 5px solid #ffcc00; margin-top: 20px;'>
                <h5>⚠️ Risk Disclaimer</h5>
                <p style='font-size: 0.9rem;'>
                    The trading recommendations provided are for informational purposes only and should not be considered as financial advice.
                    Past performance is not indicative of future results. Always conduct your own research and consider consulting a licensed financial advisor before making investment decisions.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class='footer'>
        <p>SmartStock AI Predictor © 2023 | Powered by Streamlit, yfinance, and TensorFlow</p>
        <p style='font-size: 0.8rem;'>Data provided for educational purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)

# Run the main application
if __name__ == "__main__":
    main()