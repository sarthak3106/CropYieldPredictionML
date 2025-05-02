import os

# Use the same path as in your original code
preprocessor_path = os.path.join('archive', 'preprocessor.pkl')
model_path = os.path.join('archive', 'model.pkl')

from flask import Flask, request, render_template, jsonify
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
import logging

# Basic logging setup
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    else:
        try:
            # SAFELY FETCH and VALIDATE all input fields
            fields = ['N', 'P', 'K', 'pH', 'rainfall', 'temperature', 'Area_in_hectares', 'State_Name', 'Crop_Type', 'Crop']
            inputs = {}

            logger.info("Processing form data")
            for field in fields:
                value = request.form.get(field)
                logger.info(f"Field {field}: {value}")
                
                if value is None or value.strip() == '':
                    error_msg = f"Error: {field} is required!"
                    logger.error(error_msg)
                    return render_template("form.html", error=error_msg)
                
                if field not in ['State_Name', 'Crop_Type', 'Crop']:  # Only numbers need float conversion
                    try:
                        inputs[field] = float(value)
                    except ValueError:
                        error_msg = f"Error: {field} must be a number!"
                        logger.error(error_msg)
                        return render_template("form.html", error=error_msg)
                else:
                    inputs[field] = value

            # Now Create CustomData object
            data = CustomData(
                N=inputs['N'],
                P=inputs['P'],
                K=inputs['K'],
                pH=inputs['pH'],
                rainfall=inputs['rainfall'],
                temperature=inputs['temperature'],
                Area_in_hectares=inputs['Area_in_hectares'],
                State_Name=inputs['State_Name'],
                Crop_Type=inputs['Crop_Type'],
                Crop=inputs['Crop']
            )

            logger.info("CustomData object created")
            new_data = data.get_data_as_dataframe()
            logger.info("DataFrame created")

            # Make prediction
            predict_pipeline = PredictPipeline()
            logger.info("Prediction pipeline created")
            
            pred = predict_pipeline.predict(new_data)
            logger.info(f"Prediction result: {pred}")

            # Calculate results
            production = round(pred[0], 2)
            yield_value = round(production / inputs['Area_in_hectares'], 2)

            final_result = f"Predicted Crop Production: {production} tons"
            yield_result = f"Predicted Yield: {yield_value} tons/hectare"

            logger.info(f"Final results: {final_result}, {yield_result}")
            return render_template("index.html", final_result=final_result, yield_result=yield_result)
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return render_template("form.html", error=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)