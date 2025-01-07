import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec

class FeatureEngineering:
    """
    A class that performs feature engineering on the given data.
    Includes methods to sort data, generate embeddings, assign seasons, and more.
    """ 
    @classmethod  
    def apply_saved_arima_models(self, validation_df, model_file="arima_models.pkl", target_column="sellout", steps_ahead=3):
        """Apply saved ARIMA models (from a single .pkl file) to the validation set."""
        # Load the ARIMA models from the .pkl file
        with open(model_file, "rb") as f:
            arima_models = pickle.load(f)

        base_forecasts = []  # Store forecasts for each row
        unique_skus = validation_df['sku_name'].unique()

        for sku in tqdm(unique_skus, desc="Applying saved ARIMA models"):
            sku_data = validation_df[validation_df['sku_name'] == sku]
            time_series = sku_data[target_column].values

            if sku in arima_models:
                try:
                    # Use the saved ARIMA model to generate forecasts
                    arima_model = arima_models[sku]
                    forecast = arima_model.predict(n_periods=min(len(sku_data), steps_ahead))
                except Exception as e:
                    print(f"Failed to use ARIMA model for SKU {sku}: {e}")
                    forecast = [np.nan] * len(sku_data)  # Fallback to NaN forecasts
            else:
                print(f"No ARIMA model found for SKU {sku}, using Naive forecast.")
                forecast = [time_series[-1]] * min(len(sku_data), steps_ahead)

            # Expand forecasts to match the original data length for this SKU
            forecast = np.pad(forecast, (0, len(sku_data) - len(forecast)), constant_values=0)
            base_forecasts.extend(forecast)

        # Validate forecast length
        if len(base_forecasts) != len(validation_df):
            raise ValueError(f"Forecast length ({len(base_forecasts)}) does not match DataFrame length ({len(validation_df)})")

        # Add ARIMA forecasts to the validation DataFrame
        validation_df["arima_forecast"] = pd.Series(base_forecasts, dtype="float64")
        return validation_df




    
