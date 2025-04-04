from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pytrends.request import TrendReq
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.losses import MeanSquaredError  # Explicitly define the loss function
from sklearn.preprocessing import MinMaxScaler
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import os
from pathlib import Path
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pytrends
pytrends = TrendReq(hl="en-US", tz=480)

# Define directories for local storage
DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Retry strategy for Google Trends request
@retry(
    stop=stop_after_attempt(5),  # Retry up to 5 times
    wait=wait_exponential(multiplier=1, min=2, max=10),  # Exponential backoff (2s, 4s, 8s...)
    retry=retry_if_exception_type(Exception),  # Retry on any exception
)
def fetch_trends(search_terms, timeframe):
    """ Fetch Google Trends data with retry logic. """
    try:
        logger.info(f"Fetching trends for terms: {search_terms}, timeframe: {timeframe}")
        pytrends.build_payload(search_terms, cat=0, timeframe=timeframe, geo="", gprop="")
        data = pytrends.interest_over_time()
        
        if data.empty:
            raise ValueError("No data available, retrying...")  # Force retry if data is empty
        
        return data
    except Exception as e:
        logger.error(f"Error fetching trends: {str(e)}")
        raise ValueError(f"Error fetching trends: {str(e)}")

def validate_date(date_str):
    """ Validate and format date strings. """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {date_str}. Use YYYY-MM-DD.")

@app.get("/api/trends")
async def get_trends(
    keywords: str = Query(..., description="Comma-separated list of keywords"),
    start_date: str = None,
    end_date: str = None
):
    """
    Fetch trends data from Google Trends for multiple keywords.
    Example: /api/trends?keywords=pope,sick&start_date=2024-01-01&end_date=2024-12-31
    """
    search_terms = [term.strip() for term in keywords.split(",")]
    
    # Validate and construct timeframe
    if start_date and end_date:
        start_date = validate_date(start_date)
        end_date = validate_date(end_date)
        if start_date > end_date:
            raise HTTPException(status_code=400, detail="start_date must be earlier than end_date.")
        timeframe = f"{start_date} {end_date}"
    else:
        timeframe = "today 5-y"

    try:
        logger.info(f"Fetching trends for keywords: {search_terms}, timeframe: {timeframe}")
        interest_over_time = fetch_trends(search_terms, timeframe)

        response_data = {"dates": interest_over_time.index.strftime("%Y-%m-%d").tolist()}
        for keyword in search_terms:
            response_data[keyword] = interest_over_time[keyword].tolist()

        logger.info(f"Successfully fetched trends for keywords: {search_terms}")
        return response_data
    except Exception as e:
        logger.error(f"Failed to fetch trends: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch trends: {str(e)}")

@app.get("/api/predict")
async def predict_trend(
    keyword: str = Query(..., description="Keyword to predict"),
    start_date: str = None,
    end_date: str = None
):
    """
    Predict future trends for a single keyword using LSTM.
    Exclude data flagged as 'isPartial: true'.
    Example: /api/predict?keyword=python
    """
    # Validate and construct timeframe
    if start_date and end_date:
        start_date = validate_date(start_date)
        end_date = validate_date(end_date)
        if start_date > end_date:
            raise HTTPException(status_code=400, detail="start_date must be earlier than end_date.")
        timeframe = f"{start_date} {end_date}"
    else:
        timeframe = "today 12-m"  # Default to 12 months of data

    try:
        logger.info(f"Starting prediction for keyword: {keyword}, timeframe: {timeframe}")

        # Define file path for cached data
        file_path = Path(DATA_DIR) / f"{keyword}_{timeframe.replace(' ', '_')}.json"

        # Fetch data from Google Trends if not already cached
        if not file_path.exists():
            logger.info("Fetching data from Google Trends...")
            interest_over_time = fetch_trends([keyword], timeframe)

            # Convert index to strings to ensure JSON serialization works
            interest_over_time.index = interest_over_time.index.strftime("%Y-%m-%d")

            # Save fetched data to cache
            cached_data = interest_over_time.reset_index().to_dict(orient="records")
            with open(file_path, "w") as f:
                json.dump(cached_data, f)
                logger.info(f"Data successfully cached to {file_path}")
        else:
            logger.info("Data found in cache")

        # Read cached data
        with open(file_path, "r") as f:
            cached_data = json.load(f)

        # Convert cached data back into a DataFrame
        interest_over_time = pd.DataFrame(cached_data)
        interest_over_time["date"] = pd.to_datetime(interest_over_time["date"])  # Ensure 'date' is datetime
        interest_over_time.set_index("date", inplace=True)

        # Filter out partial data
        if "isPartial" in interest_over_time.columns:
            logger.info("Excluding data flagged as 'isPartial: true'")
            interest_over_time = interest_over_time[~interest_over_time["isPartial"]]
        
        # Extract series data
        if keyword not in interest_over_time.columns:
            raise HTTPException(status_code=400, detail=f"Keyword '{keyword}' not found in fetched data.")
        
        series = interest_over_time[keyword].values

        # Check for sufficient data
        past_days = 90  # Desired historical window in days
        future_days = 7  # Desired prediction horizon in days

        # Calculate the average time interval between data points
        time_intervals = np.diff(interest_over_time.index).astype('timedelta64[D]').astype(int)
        avg_interval = np.mean(time_intervals) if len(time_intervals) > 0 else 1  # Default to 1 day if no intervals
        logger.info(f"Average time interval between data points: {avg_interval:.2f} days")

        required_data_points = int((past_days + future_days - 1) / avg_interval)  # Adjust for time intervals
        if len(series) < required_data_points:
            logger.warning(f"Not enough data for keyword: {keyword}. Found {len(series)} data points, need {required_data_points}. Fetching 12 months of data.")
            timeframe = "today 12-m"  # Extend timeframe to 12 months
            interest_over_time = fetch_trends([keyword], timeframe)

            # Filter out partial data again
            if "isPartial" in interest_over_time.columns:
                interest_over_time = interest_over_time[~interest_over_time["isPartial"]]
            
            series = interest_over_time[keyword].values

            # Recalculate time intervals and required data points
            time_intervals = np.diff(interest_over_time.index).astype('timedelta64[D]').astype(int)
            avg_interval = np.mean(time_intervals) if len(time_intervals) > 0 else 1
            required_data_points = int((past_days + future_days - 1) / avg_interval)

            if len(series) < required_data_points:
                logger.error(f"Still not enough data after fetching 12 months. Found {len(series)} data points, need {required_data_points}.")
                raise HTTPException(status_code=400, detail=f"Not enough data to train the model (need {required_data_points} data points).")

        # Prepare data for LSTM
        X, y, scaler = prepare_data(series, past_days=required_data_points, future_days=future_days)

        # Train or load model
        model = train_lstm(X, y, keyword)

        # Predict next 7 days
        last_sequence = series[-required_data_points:].reshape(1, required_data_points, 1)
        prediction = model.predict(last_sequence).flatten()

        # Reverse scale to original values
        predicted_values = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()

        # Generate future dates
        future_dates = pd.date_range(start=interest_over_time.index[-1], periods=future_days + 1, freq=f"{avg_interval}D")[1:].strftime("%Y-%m-%d").tolist()

        logger.info(f"Prediction completed for keyword: {keyword}")
        return {
            "historical": series.tolist(),
            "predicted_dates": future_dates,
            "predicted_scores": predicted_values.tolist(),
        }
    except Exception as e:
        logger.error(f"Failed to predict trend: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to predict trend: {str(e)}")

def prepare_data(series, past_days=90, future_days=7):
    """ Prepare data for LSTM: scale, reshape, and create sequences. """
    scaler = MinMaxScaler(feature_range=(0, 1))
    series_scaled = scaler.fit_transform(series.reshape(-1, 1))

    X, y = [], []
    for i in range(len(series_scaled) - past_days - future_days + 1):
        X.append(series_scaled[i:i+past_days])
        y.append(series_scaled[i+past_days:i+past_days+future_days])

    return np.array(X), np.array(y), scaler

def train_lstm(X, y, keyword):
    """ Train LSTM model and save it. """
    model_path = Path(MODEL_DIR) / f"{keyword}.keras"  # Use native Keras format

    # Check if model already exists
    if model_path.exists():
        logger.info(f"Loading pre-trained model for keyword: {keyword}")
        return load_model(model_path)

    logger.info(f"Training new LSTM model for keyword: {keyword}")
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        LSTM(50, return_sequences=False),
        Dense(7)  # 7 days prediction
    ])
    model.compile(optimizer="adam", loss=MeanSquaredError())  # Explicitly define the loss function
    model.fit(X, y, epochs=50, batch_size=8, verbose=0)
    
    model.save(model_path)  # Save using native Keras format
    logger.info(f"Saved trained model for keyword: {keyword}")
    return model