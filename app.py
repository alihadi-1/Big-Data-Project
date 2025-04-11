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
latest_price = float(data['Close'].iloc[-1])
st.metric(label="Latest Price", value=f"${latest_price:.2f}")

# Plot interactive chart
fig = px.line(x=data.index, y=data['Close'].squeeze(), title=f'{ticker} Closing Prices')
fig.update_layout(xaxis_title="Time", yaxis_title="Close Price ($)", height=500)
st.plotly_chart(fig, use_container_width=True)

# Show raw data
with st.expander("Show Raw Data"):
    st.dataframe(data.tail(10))

# Prepare file paths
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f"stock_data_{ticker}_{timestamp}.csv"

# Local directory (shared with Docker container)
local_data_dir = Path("data")
local_data_dir.mkdir(exist_ok=True)
local_file_path = local_data_dir / file_name
data.tail(1).to_csv(local_file_path)
st.write(f"Saved snapshot to `{local_file_path}`")

# -------------------------------
# Machine Learning: Price Prediction
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

hdfs_data_dir = "/data"
hdfs_output_dir = f"/data/output_{timestamp}"

st.subheader("🔍 Docker Check")
docker_check = subprocess.run("docker ps", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if docker_check.returncode == 0:
    st.success("✅ Docker is available in this environment")
else:
    st.error(f"❌ Docker NOT available: {docker_check.stderr.decode()}")

# Ensure /data exists in HDFS
subprocess.run(f"docker exec -i namenode hdfs dfs -mkdir -p {hdfs_data_dir}", shell=True)

# Upload file
upload_cmd = f"docker exec -i namenode hdfs dfs -put -f {hdfs_data_dir}/{file_name} {hdfs_data_dir}/"
upload_status = subprocess.run(upload_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

if upload_status.returncode == 0:
    st.success(f"✅ Uploaded to HDFS: {hdfs_data_dir}/{file_name}")
else:
    st.error(f"❌ Upload failed: {upload_status.stderr.decode()}")

# Run MapReduce
mapreduce_cmd = (
    f"docker exec -i namenode mapred streaming "
    f"-input {hdfs_data_dir} "
    f"-output {hdfs_output_dir} "
    f"-files /data/mapper.py,/data/reducer.py "
    f"-mapper mapper.py -reducer reducer.py"
)
mr_status = subprocess.run(mapreduce_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

if mr_status.returncode == 0:
    st.success(f"✅ MapReduce completed. Output: {hdfs_output_dir}")
else:
    st.error(f"❌ MapReduce failed: {mr_status.stderr.decode()}")

# Read result
read_cmd = f"docker exec -i namenode hdfs dfs -cat {hdfs_output_dir}/part-00000"
read_result = subprocess.run(read_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

if read_result.returncode == 0:
    result_text = read_result.stdout.decode().strip()
    st.code(result_text, language='bash')

    # Optional: Display as chart if output is structured
    if "Average Close" in result_text and "Max Volume" in result_text:
        lines = result_text.splitlines()
        values = {}
        for line in lines:
            key, val = line.split("\\t")
            values[key.strip()] = float(val.strip())

        df_summary = pd.DataFrame(values.items(), columns=["Metric", "Value"])
        st.bar_chart(df_summary.set_index("Metric"))
else:
    st.warning("⚠️ Could not read MapReduce output.")
