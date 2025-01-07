import pandas as pd
import numpy as np
import pickle
import seaborn as sns
from lightgbm import LGBMRegressor
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import matplotlib.style as style # for styling the graphs
style.use('ggplot') # chosen style
plt.rc('xtick',labelsize=13) # to globally set the tick size
plt.rc('ytick',labelsize=13) # to globally set the tick size
# To print multiple outputs together
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
# Change column display number during print
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)



class ModelPredictions:
    @classmethod
    def load_lightgbm_model_and_predict(self,model_pkl_file, data_df):
        """
        Load a LightGBM model from a .pkl file, generate predictions, 
        and store the predictions in a DataFrame.

        Parameters:
        - model_pkl_file (str): Path to the .pkl file containing the LightGBM model.
        - data_df (pd.DataFrame): DataFrame containing the data for prediction.

        Returns:
        - pd.DataFrame: DataFrame with a new column `predictions` containing model predictions.
        """
        # Load the LightGBM model from the .pkl file
        with open(model_pkl_file, "rb") as file:
            model = pickle.load(file)
        
        # # Ensure the DataFrame contains the required feature columns
        # missing_columns = [col for col in feature_columns if col not in data_df.columns]
        # if missing_columns:
        #     raise ValueError(f"The following required columns are missing from the DataFrame: {missing_columns}")
        
        feature_columns = [
    "leftover_inventory",
    "sellin_channel_1",
    "sellout",
    "sellin_channel_4",
    "sellin_channel_6",
    "sellin_channel_8",
    "sellin_channel_2",
    "sellin_channel_3",
    "sellout_channel_10",
    "sellin_channel_5",
    "ratio_inventory",
    "sellin_channel_7",
    "onhand_inventory_channel_10_lag_3",
    "sellout_channel_6",
    "sellout_channel_4",
    "sellout_channel_3",
    "onhand_inventory_channel_10",
    "onhand_inventory",
    "sellout_channel_10_lag_3",
    "onhand_inventory_channel_9",
    "leftover_inventory_lag_3",
    "sellout_channel_1",
    "year",
    "month",
    "sellout_channel_9",
    "onhand_inventory_channel_2",
    "sellout_channel_5",
    "onhand_inventory_lag_3",
    "sellin_channel_1_lag_3",
    "product_lifecycle_stage",
    "product_sku_embedding",
    "onhand_inventory_channel_4",
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
        