import pandas as pd 
import pickle
from flask import Flask, request, jsonify, render_template
from Arima import FeatureEngineering
from Modeling import ModelPredictions
app = Flask(__name__)


# Serve the HTML form
@app.route("/")
def index():
    return render_template("fossil.html")  # Ensure your HTML file is named `index.html` and is placed in a `templates` folder.


@app.route("/fossil-demand-forecasting", methods=["POST"])
def fossil_demand_forecasting():
    # Fetching data from input request body
    try:
        data = request.get_json()
        sku_name = data.get("sku_name", "")
        year = data.get("year", "")
        month = data.get("month", "")
        CAT_GENDER_MEN = data.get("CAT_GENDER_MEN", "")
        CAT_GENDER_WOMEN = data.get("CAT_GENDER_WOMEN", "")
        CAT_GENDER_BOTH = data.get("CAT_GENDER_BOTH", "")
        
        #filter values 
        filter_values = {
            "sku_name": sku_name,
            "year": int(year),
            "month": int(month),
            "CAT_GENDER_MEN": int(CAT_GENDER_MEN),
            "CAT_GENDER_WOMEN": int(CAT_GENDER_WOMEN),
            "CAT_GENDER_BOTH": int(CAT_GENDER_BOTH)
        }
        print(filter_values)
        path  = "Feature-Engineering-Test.csv"
        arima_path = "arima_models.pkl"
        # Add Machine Learning Pipeline
        # load the test value table for which the prediction is required to be generated
        df = pd.read_csv(path)
        print(df)
        
        # Filter DataFrame
        filtered_df = df.copy()
        for key, value in filter_values.items():
            if value:
                filtered_df = filtered_df[filtered_df[key] == value]     
        feature_engineering = FeatureEngineering()
        filtered_df = feature_engineering.apply_saved_arima_models(filtered_df, arima_path, "sellout", 3)

        # Generating Final Predictions
        model_pkl_file = "lightgbm_model.pkl"
        light_gbm = ModelPredictions()
        final_df = light_gbm.load_lightgbm_model_and_predict(model_pkl_file, filtered_df)
        response = { "sku_name": final_df["sku_name"].iloc[0], "predictions": round(final_df["predictions"].iloc[0], 0)}

        # Create the statement
        statement = (
            f"For SKU '{response['sku_name']}', the total number of units required in the upcoming quarter is {response['predictions']}."
        )

        return jsonify({"statement":statement}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
      
      
    
    