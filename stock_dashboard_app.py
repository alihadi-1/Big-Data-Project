# stock_dashboard_app.py

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

HADOOP_VERSION = "3.2.1"

st.set_page_config(page_title="Real-Time Stock Dashboard", layout="wide")

# Sidebar - user input
st.sidebar.title("Stock Streaming Settings")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
interval = st.sidebar.selectbox("Interval", ["1m", "2m", "5m", "15m"])
period = st.sidebar.selectbox("Period", ["1d", "5d", "7d", "1mo"])

st.title(f"📈 Real-Time Stock Dashboard: {ticker}")

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

    # -------------------------------
    # Machine Learning: Price Prediction                                                            #? PREDICT NEXT 2 DAYS NOT PREVIOUS OR ALREADY EXISTING
    # -------------------------------
    df = data[['Close']].copy()
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    X = df[['Close']]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    st.subheader("📊 ML Prediction: Next Price")
    st.write(f"**Mean Squared Error (MSE):** `{mse:.4f}`")

    results = pd.DataFrame({
        'Time': y_test.index,
        'Actual': y_test.values,
        'Predicted': predictions
    })
    fig2 = px.line(results, x='Time', y=['Actual', 'Predicted'], title="Actual vs Predicted Prices")
    fig2.update_layout(xaxis_title="Time", yaxis_title="Price ($)")
    st.plotly_chart(fig2, use_container_width=True)

    # -------------------------------
    # Hadoop HDFS + MapReduce Section
    # -------------------------------
    st.subheader("🗃️ Hadoop HDFS + Custom MapReduce")

    # Check if Docker is running
    st.subheader("🔍 Docker Check")
    docker_check = subprocess.run("docker ps", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if docker_check.returncode == 0:
        st.success("✅ Docker is available in this environment")
    else:
        st.error(f"❌ Docker NOT available: {docker_check.stderr.decode()}")
        st.stop()  # Stop execution if Docker is not available

    # Prepare mapper and reducer scripts

                                                                                        #? FIX MAPPER
    mapper_content = """#!/usr/bin/env python3
import sys
import csv

reader = csv.reader(sys.stdin)
header = next(reader, None)

for row in reader:
    try:
        close = float(row["Close"])
        volume = int(row["Volume"])
    except:
        continue
"""
                                                                                        #? FIX REDUCER
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

        if key == "Close":
            totals[key] += value
            counts[key] += 1
        elif key == "volume":
            maximum[key] = max(maximum[key], int(value))
    except:
        continue

if counts["Close"] > 0:
    avg_close = totals["Close"] / counts["Close"]

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
    # Fix for ValueError: Data must be 1-dimensional
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

    # Debug: Check if file exists
    if os.path.exists(file_path):
        st.success(f"✅ File verified at: {file_path}")
        
        # Get Unix-style path for Docker
        unix_path = str(file_path).replace('\\', '/')
        hdfs_dir = "/data"
        hdfs_output_dir = f"/data/output_{timestamp}"
        
        # Copy mapper and reducer to container (if using bind mounts, this might be optional)
        try:
            # Create the data directory in HDFS if it doesn't exist
            subprocess.run(f"docker exec -i namenode hdfs dfs -mkdir -p {hdfs_dir}", 
                          shell=True, check=True)
            
            # Debug: List files in local data directory
            with st.expander("Local Files"):
                result = subprocess.run(f"dir {data_dir}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                st.code(result.stdout.decode())
            
            # Debug: Check if container can access files
            with st.expander("Container Files"):
                result = subprocess.run("docker exec -i namenode ls -la /data", 
                                       shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                st.code(result.stdout.decode())
            
            # Upload data file to HDFS
            # Fix path handling for Windows
            container_path = f"/data/{file_name}"
            
            # First copy the file into the container
            copy_cmd = f"docker cp {file_path} namenode:{container_path}"
            copy_status = subprocess.run(copy_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if copy_status.returncode == 0:
                st.success(f"✅ Copied file to container: {container_path}")
                
                # Now put it into HDFS from within the container
                hdfs_put_cmd = f"docker exec -i namenode hdfs dfs -put -f {container_path} {hdfs_dir}/"
                upload_status = subprocess.run(hdfs_put_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                if upload_status.returncode == 0:
                    st.success(f"✅ Uploaded to HDFS: {hdfs_dir}/{file_name}")
                    
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
                    
                    # Try using hadoop-streaming.jar directly (more reliable than mapred command)
                    streaming_jar = f"/opt/hadoop-{HADOOP_VERSION}/share/hadoop/tools/lib/hadoop-streaming-{HADOOP_VERSION}.jar"
                    mapreduce_cmd = (
                        f"docker exec -i namenode hadoop jar {streaming_jar} "
                        f"-files /data/mapper.py,/data/reducer.py "
                        f"-input {hdfs_dir}/{file_name} "
                        f"-output {hdfs_output_dir} "
                        f"-mapper \"python mapper.py\" "
                        f"-reducer \"python reducer.py\""
                    )
                    
                    st.subheader("MapReduce Command")
                    st.code(mapreduce_cmd, language="bash")
                    
                    mr_status = subprocess.run(mapreduce_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                             text=True)
                    
                    if mr_status.returncode == 0:
                        st.success(f"✅ MapReduce job completed successfully!")
                        
                        # Wait a moment for HDFS to finalize
                        time.sleep(2)
                        
                        # Read result
                        read_cmd = f"docker exec -i namenode hdfs dfs -cat {hdfs_output_dir}/part-00000"
                        read_result = subprocess.run(read_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        
                        if read_result.returncode == 0:
                            result_text = read_result.stdout.decode().strip()
                            st.subheader("MapReduce Results")
                            st.code(result_text, language='text')
                            
                            # Try to display as chart
                            try:
                                lines = result_text.splitlines()
                                values = {}
                                for line in lines:
                                    key, val = line.split("\t")
                                    values[key.strip()] = float(val.strip())
                                
                                df_summary = pd.DataFrame(values.items(), columns=["Metric", "Value"])
                                st.subheader("Results Visualization")
                                st.bar_chart(df_summary.set_index("Metric"))
                            except Exception as e:
                                st.warning(f"⚠️ Couldn't visualize results: {str(e)}")
                        else:
                            st.warning("⚠️ Could not read MapReduce output.")
                            with st.expander("Error Details"):
                                st.code(read_result.stderr.decode())
                                
                            # Try listing the output directory
                            ls_cmd = f"docker exec -i namenode hdfs dfs -ls {hdfs_output_dir}"
                            ls_result = subprocess.run(ls_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            st.code(ls_result.stdout.decode())
                    else:
                        st.error(f"❌ MapReduce failed!")
                        with st.expander("Error Details"):
                            st.code(mr_status.stderr)
                else:
                    st.error(f"❌ HDFS upload failed: {upload_status.stderr.decode()}")
            else:
                st.error(f"❌ File copy to container failed: {copy_status.stderr.decode()}")
                
        except Exception as e:
            st.error(f"❌ Error during Hadoop processing: {str(e)}")
    else:
        st.error(f"❌ File not found: {file_path}")
else:
    st.error(f"No data available for {ticker} with selected parameters.")