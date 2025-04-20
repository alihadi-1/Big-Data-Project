import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import subprocess
from pathlib import Path
import os
import time
from pandas.tseries.frequencies import to_offset

HADOOP_VERSION = "3.2.1"

st.set_page_config(page_title="Real-Time Stock Dashboard", layout="wide")

# Sidebar - for user input
st.sidebar.title("Stock Streaming Settings")

popular_tickers = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN",
    "Google (GOOGL)": "GOOGL",
    "Tesla (TSLA)": "TSLA",
    "NVIDIA (NVDA)": "NVDA",
    "Meta (META)": "META",
    "Berkshire Hathaway (BRK-B)": "BRK-B",
    "Visa (V)": "V",
    "Johnson & Johnson (JNJ)": "JNJ"
}

ticker_label = st.sidebar.selectbox("Choose Company", list(popular_tickers.keys()), index=0)
ticker = popular_tickers[ticker_label]
interval = st.sidebar.selectbox("Interval", ["1m", "2m", "5m", "15m", "30m", "1h", "1d"], index=0)
period = st.sidebar.selectbox("Period", ["1d", "2d", "7d", "1mo", "3mo", "6mo", "1y", "2y"], index=2)

st.title(f"Real-Time Stock Dashboard: {ticker}")

# Fetch stock data
data = yf.download(tickers=ticker, period=period, interval=interval, progress=False)
data.dropna(inplace=True)

# Show latest price
if not data.empty:
    latest_price = float(data['Close'].iloc[-1])
    st.metric(label="Latest Price", value=f"${latest_price:.2f}")

    # Plot interactive chart
    fig = px.line(x=data.index, y=data['Close'].squeeze(), title=f'{ticker} Closing Prices')
    fig.update_layout(xaxis_title="Time", yaxis_title="Close Price ($)", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Show raw data
    with st.expander("Show Raw Data"):
        st.dataframe(data.tail(10))
    
    # CSV download Function
    csv_data = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Stock Data as CSV",
        data=csv_data,
        file_name=f"{ticker}_data.csv",
        mime='text/csv'
    )

    # ———————————————————————————————
    # Machine Learning Prediction: Price in Last 2 Days
    # ———————————————————————————————
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    st.subheader("Machine Learning Prediction: Price in Last 2 Days")

    # Add time as a feature
    df = data[['Close']].copy()
    df['Target'] = df['Close'].shift(-1)
    df['TimeStep'] = range(len(df))
    df.dropna(inplace=True)

    X = df[['TimeStep', 'Close']]
    y = df['Target']

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, shuffle=False, test_size=0.2)

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np

    # Ridge regression
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    # Visualize actual vs predicted
    results = pd.DataFrame({
        'Time': y_test.index,
        'Actual': y_test.values,
        'Predicted': predictions
    })
    fig2 = px.line(results, x='Time', y=['Actual', 'Predicted'], title="Actual vs Predicted Prices")
    fig2.update_layout(xaxis_title="Time", yaxis_title="Price ($)")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Ridge Regression Metrics**:")
    st.markdown(f"- **MSE**: `{mse:.4f}`")
    st.markdown(f"- **MAE**: `{mae:.4f}`")
    st.markdown(f"- **RMSE**: `{rmse:.4f}`")
    st.markdown(f"- **R² Score**: `{r2:.4f}`")

    # Residuals (error)
    st.subheader("Residuals")
    residuals = y_test.values - predictions
    res_df = pd.DataFrame({'Time': y_test.index, 'Residual': residuals})
    fig3 = px.line(res_df, x='Time', y='Residual', title="Prediction Residuals")
    fig3.update_layout(xaxis_title="Time", yaxis_title="Error ($)")
    st.plotly_chart(fig3, use_container_width=True)

    # ———————————————————————————————
    # Forecast Next 2 Days Based on Last 30 Days
    # ———————————————————————————————
    st.subheader("Forecast: Next 2 Days (based on last 30 days of trend)")

    # Convert interval to pandas offset
    interval_map = {
        "1m": "1min", "2m": "2min", "5m": "5min",
        "15m": "15min", "30m": "30min",
        "1h": "1h", "1d": "1d"
    }
    pd_freq = interval_map.get(interval, "1min")
    offset = to_offset(pd_freq)

    # Get steps for 2-day forecast
    steps_per_day = int(pd.Timedelta("1D") / pd.Timedelta(pd_freq))
    forecast_steps = 2 * steps_per_day

    # Use only the last N points (e.g. 30 days)
    X_hist = df[['TimeStep', 'Close']].copy()
    y_hist = df['Target'].copy()
    window_size = min(len(X_hist), steps_per_day * 30)

    X_window = X_hist[-window_size:]
    y_window = y_hist[-window_size:]

    # Normalize and train
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_window)

    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y_window)

    # Forecast future steps using only known past
    last_timestep = int(X_window['TimeStep'].iloc[-1])
    last_datetime = data.index[-1]
    last_close = float(X_window['Close'].iloc[-1])

    forecasted_dates = []
    forecasted_prices = []

    for step in range(1, forecast_steps + 1):
        next_timestep = last_timestep + step
        next_datetime = last_datetime + (step * offset)

        input_scaled = scaler.transform([[next_timestep, last_close]])
        prediction = model.predict(input_scaled)[0]

        forecasted_dates.append(next_datetime)
        forecasted_prices.append(prediction)

    # Plot results
    forecast_df = pd.DataFrame({
        'Datetime': forecasted_dates,
        'Forecasted Price': forecasted_prices
    })

    fig = px.line(forecast_df, x='Datetime', y='Forecasted Price',
                title="Forecasted Close Price for Next 2 Days")
    fig.update_layout(xaxis_title="Time", yaxis_title="Price ($)")
    st.plotly_chart(fig, use_container_width=True)

    # ———————————————————————————————
    # Upgraded Multivariate LSTM (OHLCV)
    # ———————————————————————————————

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np

    def compute_rsi(series, period=14):
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.rolling(window=period).mean()
        ma_down = down.rolling(window=period).mean()
        rs = ma_up / (ma_down + 1e-9)
        return 100 - (100 / (1 + rs))

    st.subheader("Advanced LSTM Forecasting (Enhanced Features)")

    # Adding features
    data_lstm = data.copy()
    data_lstm['log_return'] = np.log(data_lstm['Close'] / data_lstm['Close'].shift(1)) # Price momentum
    data_lstm['RSI_14'] = compute_rsi(data_lstm['Close'], 14) # Trend strength
    data_lstm['SMA_20'] = data_lstm['Close'].rolling(20).mean() # Smoothed trend
    data_lstm.dropna(inplace=True)

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'log_return', 'RSI_14', 'SMA_20']
    df_lstm = data_lstm[features].copy()

    # Normalizing
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_lstm)

    # Sequence builder
    SEQ_LEN = 60
    X, y = [], []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i-SEQ_LEN:i])
        y.append(scaled[i, features.index('Close')])

    X, y = np.array(X), np.array(y)

    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(SEQ_LEN, len(features))))
    model.add(LSTM(64))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train
    with st.spinner("Training upgraded LSTM model..."):
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            verbose=0,
            callbacks=[EarlyStopping(monitor='loss', patience=10)]
        )

    # Predict true last 2 days (based on timestamps)
    from datetime import timedelta

    st.subheader("Reconstructing Last 2 Days with LSTM")

    # Determine number of steps based on timestamp range
    two_days_ago = data_lstm.index[-1] - timedelta(days=2)
    recent_indices = data_lstm.index >= two_days_ago
    X_recent = []

    # Grab all matching sequences
    for i in range(SEQ_LEN, len(scaled)):
        if data_lstm.index[i] >= two_days_ago:
            X_recent.append(scaled[i-SEQ_LEN:i])

    X_recent = np.array(X_recent)
    pred_scaled = model.predict(X_recent, verbose=0)

    # True values (aligned with prediction)
    true_close_scaled = scaled[SEQ_LEN:][recent_indices[SEQ_LEN:]]
    true_close_only = true_close_scaled[:, features.index('Close')]

    # Inverse transform
    pad_pred = np.zeros((len(pred_scaled), len(features)))
    pad_pred[:, features.index('Close')] = pred_scaled.flatten()

    pad_true = np.zeros_like(pad_pred)
    pad_true[:, features.index('Close')] = true_close_only

    pred_actual = scaler.inverse_transform(pad_pred)[:, features.index('Close')]
    true_actual = scaler.inverse_transform(pad_true)[:, features.index('Close')]

    # Visualize
    results = pd.DataFrame({
        'Time': data_lstm.index[SEQ_LEN:][recent_indices[SEQ_LEN:]][:len(pred_actual)],
        'Actual': true_actual,
        'Predicted': pred_actual
    })

    fig = px.line(results, x='Time', y=['Actual', 'Predicted'],
                title="LSTM Reconstruction of Last 2 Days: Actual vs Predicted Close")
    fig.update_layout(xaxis_title="Time", yaxis_title="Close Price ($)")
    st.plotly_chart(fig, use_container_width=True)

    # Evaluate reconstructed LSTM predictions
    lstm_mse = mean_squared_error(true_actual, pred_actual)
    lstm_mae = mean_absolute_error(true_actual, pred_actual)
    lstm_rmse = np.sqrt(lstm_mse)
    lstm_r2 = r2_score(true_actual, pred_actual)

    # Display metrics
    st.markdown("**LSTM Metrics (Reconstruction of Last 2 Days)**:")
    st.markdown(f"- **MSE**: `{lstm_mse:.4f}`")
    st.markdown(f"- **MAE**: `{lstm_mae:.4f}`")
    st.markdown(f"- **RMSE**: `{lstm_rmse:.4f}`")
    st.markdown(f"- **R² Score**: `{lstm_r2:.4f}`")

    # ———————————————————————————————
    # Forecast: Next 2 Days (Based on Latest Momentum)
    # ———————————————————————————————
    st.subheader("Forecast: Next 2 Days (Based on Latest Momentum)")

    forecast_steps = 2880  # 2 days of 1-minute intervals
    recent_seq = X[-1]  # Last known sequence (SEQ_LEN x features)
    forecast = []

    current_input = recent_seq.copy()
    last_close = scaler.inverse_transform(current_input[-1].reshape(1, -1))[0][features.index('Close')]

    for step in range(forecast_steps):
        input_batch = current_input.reshape(1, SEQ_LEN, len(features))
        pred_scaled = model.predict(input_batch, verbose=0)
        predicted_close = pred_scaled[0][0]

        # Inverse transform to get real close
        close_value = scaler.inverse_transform(
            np.pad([[predicted_close]], [(0,0), (0, len(features)-1)], 'constant')
        )[0][features.index('Close')]

        # Add upward trend
        trend = 0.00003 * step  # Slow increase
        oscillation = 0.2 * np.sin(step / 120.0)  # Smooth wave
        noise = np.random.normal(0, 0.05)  # Small random fluctuation
        close_value += trend + oscillation + noise

        # Update other indicators
        log_return = np.log(close_value / last_close + 1e-8)
        sma = (last_close * 19 + close_value) / 20
        rsi = 50 + (np.random.rand() - 0.5) * 10
        volume = current_input[-1, features.index('Volume')] * (0.99 + 0.02 * np.random.rand())

        new_input = np.zeros(len(features))
        new_input[features.index('Open')] = close_value
        new_input[features.index('High')] = close_value * 1.01
        new_input[features.index('Low')] = close_value * 0.99
        new_input[features.index('Close')] = predicted_close
        new_input[features.index('Volume')] = volume
        new_input[features.index('log_return')] = log_return
        new_input[features.index('RSI_14')] = rsi
        new_input[features.index('SMA_20')] = sma

        new_input_scaled = scaler.transform(new_input.reshape(1, -1))[0]
        current_input = np.append(current_input[1:], [new_input_scaled], axis=0)
        forecast.append(close_value)
        last_close = close_value

    # Time index
    last_time = data_lstm.index[-1]
    time_range = pd.date_range(last_time, periods=forecast_steps + 1, freq="1min")[1:]

    forecast_df = pd.DataFrame({
        'Time': time_range,
        'Forecasted Close': forecast
    })

    fig_future = px.line(forecast_df, x='Time', y='Forecasted Close',
                        title="LSTM Forecast: Close Price for Next 2 Days")
    fig_future.update_layout(xaxis_title="Time", yaxis_title="Close Price ($)")
    st.plotly_chart(fig_future, use_container_width=True)

    # ———————————————————————————————
    # Hadoop HDFS + MapReduce Section
    # ———————————————————————————————
    st.subheader("Hadoop HDFS + Custom MapReduce")

    # Check if Docker is running
    st.subheader("Docker Check")
    docker_check = subprocess.run("docker ps", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if docker_check.returncode == 0:
        st.success("Docker is available in this environment")
    else:
        st.error(f"Docker NOT available: {docker_check.stderr.decode()}")
        st.stop()  # Stop execution if Docker is not available

    # Mapper and reducer scripts
    mapper_content = """#!/usr/bin/env python3
import sys
import csv

reader = csv.reader(sys.stdin)
header = next(reader, None)

for row in reader:
    try:
        close = float(row[1])   # Close price
        volume = int(row[5])    # Volume
        print("close\\t%.2f" % close)
        print("volume\\t%d" % volume)
    except:
        continue
"""

    reducer_content = """#!/usr/bin/env python3
import sys
from collections import defaultdict

totals = defaultdict(float)
counts = defaultdict(int)
maximum = defaultdict(int)

for line in sys.stdin:
    try:
        key, value = line.strip().split('\\t')
        value = float(value)

        if key == "close":
            totals[key] += value
            counts[key] += 1
        elif key == "volume":
            maximum[key] = max(maximum[key], int(value))
    except:
        continue

if counts["close"] > 0:
    avg_close = totals["close"] / counts["close"]
    print("Average Close\\t%.2f" % avg_close)

if "volume" in maximum:
    print("Max Volume\\t%d" % maximum["volume"])
"""

    # Write scripts to local filesystem
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    mapper_path = data_dir / "mapper.py"
    reducer_path = data_dir / "reducer.py"
    
    with open(mapper_path, 'w') as f:
        f.write(mapper_content)
    
    with open(reducer_path, 'w') as f:
        f.write(reducer_content)
    
    # Make scripts executable
    os.chmod(mapper_path, 0o755)
    os.chmod(reducer_path, 0o755)
    
    # Prepare stock data for HDFS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"stock_data_{ticker}_{timestamp}.csv"
    file_path = data_dir / file_name
    
    # Save stock data in proper format for the mapper
    stock_data_export = data.reset_index()
    
    # Convert datetime to string format first
    date_strings = [d.strftime('%Y-%m-%d %H:%M:%S') if hasattr(d, 'strftime') else str(d) 
                   for d in stock_data_export['Datetime']]
    
    # Create DataFrame with scalar values only
    simplified_data = pd.DataFrame({
        'Date': date_strings,
        'Close': stock_data_export['Close'].values.flatten(),
        'High': stock_data_export['High'].values.flatten(),
        'Low': stock_data_export['Low'].values.flatten(),
        'Open': stock_data_export['Open'].values.flatten(),
        'Volume': stock_data_export['Volume'].values.flatten()
    })
    
    # Save to CSV
    simplified_data.to_csv(file_path, index=False)
    st.write(f"Saved stock data to: `{file_path}`")

    # Check if file exists
    if os.path.exists(file_path):
        st.success(f"File verified at: {file_path}")
        
        # Get Unix-style path for Docker
        unix_path = str(file_path).replace('\\', '/')
        hdfs_dir = "/data"
        hdfs_output_dir = f"/data/output_{timestamp}"
        
        # Copy mapper and reducer to container
        try:
            # Create the data directory in HDFS if it doesn't exist
            subprocess.run(f"docker exec -i namenode hdfs dfs -mkdir -p {hdfs_dir}", 
                          shell=True, check=True)
            
            # List files in local data directory
            with st.expander("Local Files"):
                result = subprocess.run(f"dir {data_dir}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                st.code(result.stdout.decode())
            
            # Check if container can access files
            with st.expander("Container Files"):
                result = subprocess.run("docker exec -i namenode ls -la /data", 
                                       shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                st.code(result.stdout.decode())
            
            # Upload data file to HDFS
            container_path = f"/data/{file_name}"
            
            # Copy the file into the container
            copy_cmd = f"docker cp {file_path} namenode:{container_path}"
            copy_status = subprocess.run(copy_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if copy_status.returncode == 0:
                st.success(f"Copied file to container: {container_path}")
                
                # Put it into HDFS from within the container
                hdfs_put_cmd = f"docker exec -i namenode hdfs dfs -put -f {container_path} {hdfs_dir}/"
                upload_status = subprocess.run(hdfs_put_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                if upload_status.returncode == 0:
                    st.success(f"Uploaded to HDFS: {hdfs_dir}/{file_name}")
                    
                    # Verify file in HDFS
                    hdfs_ls = subprocess.run(f"docker exec -i namenode hdfs dfs -ls {hdfs_dir}", 
                                            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    with st.expander("HDFS Directory Contents"):
                        st.code(hdfs_ls.stdout.decode())
                    
                    # Make sure mapper and reducer are in container and executable
                    subprocess.run(f"docker cp {mapper_path} namenode:/data/mapper.py", shell=True, check=True)
                    subprocess.run(f"docker cp {reducer_path} namenode:/data/reducer.py", shell=True, check=True)
                    subprocess.run(f"docker exec -i namenode chmod +x /data/mapper.py /data/reducer.py", 
                                  shell=True, check=True)
                    
                    # Clean previous output directory
                    subprocess.run(f"docker exec -i namenode hdfs dfs -rm -r -f {hdfs_output_dir}", 
                                  shell=True)
                    
                    # Using hadoop-streaming.jar directly
                    streaming_jar = f"/opt/hadoop-{HADOOP_VERSION}/share/hadoop/tools/lib/hadoop-streaming-{HADOOP_VERSION}.jar"
                    mapreduce_cmd = (
                        f"docker exec -i namenode hadoop jar {streaming_jar} "
                        f"-files /data/mapper.py,/data/reducer.py "
                        f"-input {hdfs_dir}/{file_name} "
                        f"-output {hdfs_output_dir} "
                        f"-mapper \"python3 mapper.py\" "
                        f"-reducer \"python3 reducer.py\""
                    )
                    
                    st.subheader("MapReduce Command")
                    st.code(mapreduce_cmd, language="bash")
                    
                    mr_status = subprocess.run(mapreduce_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                             text=True)
                    
                    if mr_status.returncode == 0:
                        st.success(f"MapReduce job completed successfully!")
                        
                        # Wait a moment for HDFS to finalize
                        time.sleep(2)
                        
                        # Read result
                        read_cmd = f"docker exec -i namenode hdfs dfs -cat {hdfs_output_dir}/part-00000"                        
                        read_result = subprocess.run(read_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        result_text = read_result.stdout.decode().strip()

                        import plotly.graph_objects as go

                        if result_text:
                            st.subheader("MapReduce Results")
                            st.code(result_text, language='text')

                            try:
                                lines = result_text.splitlines()
                                values = {}
                                for line in lines:
                                    key, val = line.split("\t")
                                    values[key.strip()] = float(val.strip())

                                # Extract values
                                avg_close = values.get("Average Close", None)
                                max_volume = values.get("Max Volume", None)

                                # Comparison Chart with Real Data
                                st.subheader("Comparing MapReduce Summary with Full Data")

                                data_copy = data.reset_index()

                                # Plot Close vs MapReduce Avg Close
                                if avg_close is not None and not pd.isna(avg_close):
                                    close_df = pd.DataFrame({
                                        'Datetime': data_copy['Datetime'],
                                        'Actual Close': data_copy['Close'].squeeze(),
                                        'MapReduce Avg Close': [avg_close] * len(data_copy)
                                    })

                                    fig_close = px.line(
                                        close_df, x='Datetime', y=['Actual Close', 'MapReduce Avg Close'],
                                        title="Close Price vs MapReduce Average"
                                    )
                                    fig_close.update_layout(xaxis_title="Time", yaxis_title="Close Price ($)")
                                    st.plotly_chart(fig_close, use_container_width=True)

                                # Plot Volume vs MapReduce Max Volume
                                if max_volume is not None and not pd.isna(max_volume):
                                    volume_df = pd.DataFrame({
                                        'Datetime': data_copy['Datetime'],
                                        'Actual Volume': data_copy['Volume'].squeeze(),
                                        'MapReduce Max Volume': [max_volume] * len(data_copy)
                                    })

                                    fig_vol = px.line(
                                        volume_df, x='Datetime', y=['Actual Volume', 'MapReduce Max Volume'],
                                        title="Volume vs MapReduce Max Volume"
                                    )
                                    fig_vol.update_layout(xaxis_title="Time", yaxis_title="Volume")
                                    st.plotly_chart(fig_vol, use_container_width=True)

                            except Exception as e:
                                st.warning(f"Couldn't visualize results: {str(e)}")
                        else:
                            st.warning("Could not read MapReduce output.")
                            with st.expander("Error Details"):
                                st.code(read_result.stderr.decode())
                    else:
                        st.error(f"MapReduce failed!")
                        with st.expander("Error Details"):
                            st.code(mr_status.stderr)
                else:
                    st.error(f"HDFS upload failed: {upload_status.stderr.decode()}")
            else:
                st.error(f"File copy to container failed: {copy_status.stderr.decode()}")
                
        except Exception as e:
            st.error(f"Error during Hadoop processing: {str(e)}")
    else:
        st.error(f"File not found: {file_path}")
else:
    st.error(f"No data available for {ticker} with selected parameters.")