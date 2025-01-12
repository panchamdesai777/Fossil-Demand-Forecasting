import joblib
import pandas as pd
import numpy as np
from joblib import load
from vecstack import StackingTransformer
from lightgbm import LGBMRegressor
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
# Change column display number during print
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)



class ModelPredictions:
    @classmethod
    def load_lightgbm_model_and_predict(self, model_pkl_file, data_df):
        """
        Load a LightGBM model from a .pkl file, generate predictions, 
        and store the predictions in a DataFrame.
        """

        # Load the LightGBM model from the .pkl file
        model = load(model_pkl_file)

        # Ensure the DataFrame contains the required feature columns
        feature_columns = [
            "leftover_inventory", "sellin_channel_1", "sellout", "sellin_channel_4",
            "sellin_channel_6", "sellin_channel_8", "sellin_channel_2", "sellin_channel_3",
            "sellout_channel_10", "sellin_channel_5", "ratio_inventory", "sellin_channel_7",
            "onhand_inventory_channel_10_lag_3", "sellout_channel_6", "sellout_channel_4",
            "sellout_channel_3", "onhand_inventory_channel_10", "onhand_inventory",
            "sellout_channel_10_lag_3", "onhand_inventory_channel_9", "leftover_inventory_lag_3",
            "sellout_channel_1", "year", "month", "sellout_channel_9", "onhand_inventory_channel_2",
            "sellout_channel_5", "onhand_inventory_lag_3", "sellin_channel_1_lag_3",
            "product_lifecycle_stage", "product_sku_embedding", "onhand_inventory_channel_4",
            "arima_forecast"
        ]

        # Extract feature values for prediction
        X = data_df[feature_columns].values

        # Generate predictions
        predictions = model.predict(X)
        predictions = np.expm1(predictions)

        # Add predictions to the DataFrame
        data_df['predictions'] = predictions
        return data_df
