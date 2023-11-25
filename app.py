from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load your pre-trained models from pickle files
with open("rf_model.pkl", "rb") as file:
    rf_model = pickle.load(file)

with open("lgbm_model.pkl", "rb") as file:
    lgbm_model = pickle.load(file)

with open("svm_model.pkl", "rb") as file:
    svm_model = pickle.load(file)

# Flask App Code
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if request.method == 'POST':
        # Get input data from the form
        input_data = request.form.to_dict()
        input_data['patients_number_per_hour'] = float(input_data['patients_number_per_hour'])
        input_data['saturation'] = float(input_data['saturation'])
        input_data['length_of_stay_min'] = float(input_data['length_of_stay_min'])

# Convert boolean features
        bool_features = ["group_regional_ed",'sex_male', 'arrival_mode_private_ambulance', 'arrival_mode_private_vehicle',
                 'arrival_mode_public_ambulance', 'arrival_mode_walking', 'injury_yes',
                 'mental_pain_response', 'mental_unresponsive', 'mental_verbose_response',
                 'pain_yes', 'ktas_rn_non_emergency', 'disposition_admission_to_ward',
                 'disposition_death', 'disposition_discharge', 'disposition_surgery',
                 'disposition_transfer', 'ktas_expert_non_emergency', 'new_age_adult',
                 'new_age_mid_age', 'new_age_old', 'new_sbp_low', 'new_sbp_normal',
                 'new_dbp_low', 'new_dbp_normal', 'new_hr_low', 'new_hr_normal',
                 'new_rr_low', 'new_rr_normal', 'new_bt_low', 'new_bt_normal',
                 'new_nrs_pain_low_pain', 'new_nrs_pain_pain',
                 'new_ktas_duration_min_very_urgent', 'new_length_of_stay_min_non_urgent',
                 'new_length_of_stay_min_standard', 'new_length_of_stay_min_urgent',
                 'new_length_of_stay_min_very_urgent']

        for feature in bool_features:
            input_data[feature] = bool(input_data[feature])
        # Convert to DataFrame
        input_df = pd.DataFrame(input_data, index=[0])

        # Make predictions for all models
        rf_prediction = rf_model.predict(input_df)
        lgbm_prediction = lgbm_model.predict(input_df)
        svm_prediction = svm_model.predict(input_df)

        return render_template('index.html', rf_prediction=rf_prediction[0], lgbm_prediction=lgbm_prediction[0], svm_prediction=svm_prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
