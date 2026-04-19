import serial
import cv2
import time
import glob
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tcn import TCN
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input


# AUTO DETECT ESP32 PORT
def get_serial_port():
    ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
    if not ports:
        print("ESP32 not found")
        exit()
    print(f"Using serial: {ports[0]}")
    return ports[0]

SERIAL_PORT = get_serial_port()

# SERIAL SETUP
ser = serial.Serial(SERIAL_PORT, 115200, timeout=1)
time.sleep(2)

# CAMERA SETUP
def init_camera():
    for i in [0, 1]:
        cap = cv2.VideoCapture(i)
        time.sleep(1)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera ready at index {i}")
                return cap
            cap.release()
    print("Camera not found")
    exit()

cap = init_camera()

# LOAD MODELS
effnet_model = load_model("effnetv2s_model.keras",compile=False)
tcn_model    = load_model("tcn_model.keras", compile= False, custom_objects={"TCN":TCN})

with open("scaler_tcn.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("meta_model.pkl", "rb") as f:
    meta_model = pickle.load(f)

with open("meta.pkl", "rb") as f:
    meta = pickle.load(f)

SEQ_LEN = meta["SEQ_LEN"]
#SEQ_LEN=15
TCN_FEATURES = meta["TCN_FEATURES"]
EFFNET_SIZE = meta["EFFNET_SIZE"]

THRESH_IMG = meta['EFFNET_THRESHOLD']
THRESH_TCN = meta['TCN_THRESHOLD']

print("Models Loaded")

# FEATURE ENGINEERING
SHORT_WIN = 5
LONG_WIN  = 30

def add_features(df):
    df = df.copy().sort_values('timestamp')

    for col in ['temperature', 'mq135', 'mq2']:
        df[f'{col}_delta'] = df[col].diff().fillna(0)
        df[f'{col}_accel'] = df[f'{col}_delta'].diff().fillna(0)

        df[f'{col}_roll_mean_s'] = df[col].rolling(SHORT_WIN, min_periods=1).mean()
        df[f'{col}_roll_std_s']  = df[col].rolling(SHORT_WIN, min_periods=1).std().fillna(0)

        df[f'{col}_baseline'] = df[col].rolling(LONG_WIN, min_periods=1).mean()
        df[f'{col}_dev_from_base'] = df[col] - df[f'{col}_baseline']

        df[f'{col}_slope'] = df[col].rolling(SHORT_WIN, min_periods=2).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
            raw=True
        ).fillna(0)

        df[f'{col}_cumsum_delta'] = df[f'{col}_delta'].cumsum()

    eps = 1e-6
    df['mq2_mq135_ratio'] = df['mq2'] / (df['mq135'] + eps)
    df['temp_x_mq2'] = df['temperature'] * df['mq2']
    df['temp_x_mq135'] = df['temperature'] * df['mq135']

    return df

# PREPROCESSING
def preprocess_image(frame):
    
    if isinstance(EFFNET_SIZE, tuple):
        size = EFFNET_SIZE
    else:
        size = (int(EFFNET_SIZE), int(EFFNET_SIZE))
    
    img = cv2.resize(frame, size)
    img = img.astype(np.float32)
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)

    
def preprocess_sensor(df):

    df = add_features(df)

    # ensure correct feature order
    df = df[TCN_FEATURES]

    # convert to numpy
    arr = df.values

    # scale (same as training logic)
    scaled = scaler.transform(arr)

    # reshape → (1, T, F)
    return np.expand_dims(scaled, axis=0)

# PREDICTION
def predict(frame, sensor_df):

    # Image prediction
    img = preprocess_image(frame)
    effnet_prob_raw = effnet_model.predict(img, verbose=0).flatten()[0]
    
    # CRITICAL FIX: Convert "Safe" probability into "Fire" probability
    effnet_prob = 1.0 - effnet_prob_raw

    # Sensor prediction
    sensor = preprocess_sensor(sensor_df)
    tcn_prob = tcn_model.predict(sensor, verbose=0).flatten()[0]

    # Meta fusion
    meta_input = np.array([[effnet_prob, tcn_prob]])

    # IMPORTANT: use predict (NOT predict_proba) to get hard 0 or 1
    final_pred = meta_model.predict(meta_input)[0]

    return final_pred, effnet_prob, tcn_prob


# SENSOR BUFFER
sensor_df = pd.DataFrame()


# SERIAL READ
def read_sensor():
    try:
        line = ser.readline().decode().strip()

        if "timestamp" in line:
            return None

        values = line.split(',')

        if len(values) != 7:
            return None

        return {
            "timestamp": float(values[0]),
            "temperature": float(values[1]),
            "humidity": float(values[2]),
            "mq135": float(values[3]),
            "mq2": float(values[4]),
            "flame": float(values[5]),
            "distance": float(values[6])
        }

    except:
        return None


# MAIN LOOP
print(f"SEQ_LEN:{SEQ_LEN}")
print("Running system...")

while True:
    row = read_sensor()

    if row is None:
        continue

    print("📡", row)

    # update buffer
    sensor_df = pd.concat([sensor_df, pd.DataFrame([row])], ignore_index=True)

    if len(sensor_df) > SEQ_LEN:
        sensor_df = sensor_df.iloc[-SEQ_LEN:]
    
    if cap is None or not cap.isOpened():
        print("Reinitializing camera...")
        cap = init_camera()
    
    # capture frame
    ret, frame = cap.read()

    if not ret:
        print("Camera error")
        continue

    # predict when buffer full
    if len(sensor_df) == SEQ_LEN:
        result, eff_p, tcn_p = predict(frame, sensor_df)

        
        print(f"EffNet: {eff_p:.3f} | TCN: {tcn_p:.3f}")
        print("Final Prediction:", result)

        if result == 1:
            print("FIRE ALERT")
        else:
            print("SAFE")

