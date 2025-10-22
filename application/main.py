import os
import threading
import joblib
import numpy as np
import pandas as pd
import io
from dotenv import load_dotenv
from flask import Flask
from flask import request
from ml.hotspotDetector.datasetLoader import DatasetLoader
from ml.hotspotDetector.preprocessing import Preprocessing
from ml.hotspotDetector.cityClustering import CityClustering
from ml.hotspotDetector.stateClustering import StateClustering
from ml.configloader import ConfigLoader
from ml.severityPrediction.manufacturingYearImputer import ManufacturingYearImputer
from ml.severityPrediction.personSexImputer import PersonSexImputer
from ml.severityPrediction.frequencySubsetEncoder import FrequencySubsetEncoder
from ml.severityPrediction.ordinalSubsetEncoder import OrdinalSubsetEncoder
from webview import WebView

load_dotenv()
FLASK_PORT = os.getenv('FLASK_PORT')

stop_event = threading.Event()
app = Flask(__name__)
# At the beginning we don't load any model, we wait for the selection
# of the classification model from the frontend to avoid un-useful overhead
_model = None
_require_conversion = None

def get_model():
    """ Loads the model from the joblib file into the main memory
    (even for future and concurrent accesses). Returns -1 if the
    joblib file isn't found, otherwise the model"""
    
    # Reads the model name from the config json file
    model_name = configloader.get_classificationModel()
    models_requiring_conversion = ['models/xgboost.joblib', 'models/lightgbm.joblib']
    if not os.path.exists(model_name):
        print("[ERR] Model file not found")
        return -1
    
    global _model
    global _require_conversion
    if _model is None:
        _model = joblib.load(model_name)
        _require_conversion = False
        print("[INFO] Joblib ended the loading of the model..")
        if model_name in models_requiring_conversion:
            _require_conversion = True
             
    return _model, _require_conversion

@app.route("/crashspot_stop", methods=["POST"])
def crashspot_stop():
    stop_event.set()
    shutdown = request.environ.get("werkzeug.server.shutdown")
    if shutdown is not None:
        shutdown()
    return {"status": "ok"}, 200

@app.route("/general_cause_of_accident", methods=["GET"])
def send_general_cause_of_accident():
    city = request.args.get("city")
    state = request.args.get("state")
    if city != None:
        causes_list = preprocessing.getGeneralCausesCity(city)
    if state != None:
        causes_list = preprocessing.getGeneralCausesState(state)
    return {"causes_list": causes_list}, 200

@app.route("/cities", methods=["GET"])
def send_cities():
    cities_list = preprocessing.getCities()
    return {"cities_list": cities_list}, 200

@app.route("/states", methods=["GET"])
def send_states():
    states_list = preprocessing.getStates()
    return {"states_list": states_list}, 200

@app.route("/clustering", methods=["POST"])
def clustering():
    payload = request.get_json()
    algorithm = payload.get("algorithm", "")
    if algorithm == "DBSCAN":
        cityClustering = CityClustering(preprocessed_df, 
                        payload["city"],
                        payload["cause"],
                        configloader.getKDistGraph(),
                        configloader.get_dbscanMinEps(),
                        configloader.get_dbscanStepEps(),
                        configloader.get_dbscanMinPtsArr())
        hopkins = cityClustering.getHopkins()
        knee = cityClustering.knee_heurstic_search()
        sts, cityClustering_perf = cityClustering.clustering_tuning()

        if sts == -1:
            response = {
                "sts": sts
            }
            return response, 200

        cityClustering.run(eps_km = cityClustering_perf.iloc[0,0],
                                minPts=cityClustering_perf.iloc[0,1])
        print(f"eps_km: {cityClustering_perf.iloc[0,0]}, minPts: {cityClustering_perf.iloc[0,1]}")
        cityClustering.attachLabels()
        cityClustering.add_victims_condition_rank()
        # We have to serialize both the dataframe and the clustering 
        # performances as a response to the frontend
        response = {
            "sts": sts,
            "hopkins": hopkins,
            "max_eps": knee,
            "dfLabelled": cityClustering.get_dfLabelled().to_json(),
            "clusteringPerf": cityClustering_perf.to_json()
        }
        return response, 200
    if algorithm == "OPTICS":
        max_eps = configloader.get_opticsMaxRadiusArr()[-1]
        stateClustering = StateClustering(preprocessed_df, 
                        payload["state"],
                        payload["cause"],
                        configloader.get_opticsMaxRadiusArr(),
                        configloader.get_opticsMinPtsArr(),
                        configloader.get_opticsXiArr())
        hopkins = stateClustering.getHopkins()
        sts, stateClustering_perf = stateClustering.clustering_tuning()

        if sts == -1:
            response = {
                "sts": sts
            }
            return response, 200

        stateClustering.run(minPts = stateClustering_perf.iloc[0,0],
                max_eps = stateClustering_perf.iloc[0,1],
                xi = stateClustering_perf.iloc[0,2])
        stateClustering.attachLabels()
        stateClustering.add_victims_condition_rank()
        response = {
            "sts": sts,
            "hopkins": hopkins,
            "max_eps": max_eps,
            "dfLabelled": stateClustering.get_dfLabelled().to_json(),
            "clusteringPerf": stateClustering_perf.to_json()
        }
        return response, 200

@app.route("/selected_features", methods=["GET"])
def send_selected_features():
    model, _ = get_model()
    if model == -1:
        return "Model not found", 500
    
    selected_features = model.named_steps["clf"].feature_names_in_
    # We return the features to the frontend as a list
    return {"selected_features": selected_features.tolist()}

@app.route("/classify", methods=["POST"])
def classify():
    model, require_conversion = get_model()
    if model == -1:
        return "Model not found", 500
    payload = request.get_json()
    
    X_test = pd.read_json(io.StringIO(payload['X_test_fs']))
    y_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    # np.arange generates a numpy array of INDEXES.
    # those indexes are as many as the number of predictions, so 
    # as many as the number of the elements of the test-set / real-data to predict

    # Here we are constructing an array in which the element i
    # contains a percentage about how much does the model believes
    # in its prediction.
    
    # Some conversion required to work with numpy arrays
    classes = model.classes_
    y_pred_idx = np.array([np.where(classes == c)[0][0] for c in y_pred])

    confidence = y_proba[np.arange(len(y_pred_idx)), y_pred_idx]

    # Checking if the classes need to be converted
    if require_conversion:
        mapping = {0: "Injured", 1: "Severe", 2: "Unharmed"}
        y_pred = pd.Series(y_pred).map(mapping).to_numpy()

    # Here we construct a dataframe which associate the prediction
    # and the confidence for each prediction that the model made

    predictions_df = pd.DataFrame({
        "predicted": y_pred,
        "confidence": confidence
    })

    response = {
        "prediction_df": predictions_df.to_json()
    }

    return response, 200
    

def run_flask():
    app.run(host="0.0.0.0", port=FLASK_PORT)

# We run Flask as a separated thread
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

configloader = ConfigLoader(os.path.join(os.path.dirname(__file__), "config.json"))
datasetloader = DatasetLoader(full_dataset_path=configloader.getFullDataset(),
                    clean_dataset_path=configloader.getCleanDataset())
cleaned_df = datasetloader.loadDataset()
preprocessing = Preprocessing(cleaned_df)
preprocessing.drop_columns(cols_to_keep=["latitude",
            "longitude", "cause_of_accident", "victims_conditions", "road_id", "km", "city", "state", "victims_condition"])
preprocessing.map_general_causes()
preprocessed_df = preprocessing.getDataframe()

# Automatically ends the process associated to the frontend at the
# end of life of the main program
with WebView() as webview:
    
    # We wait for the button to be pressed, so that the subprocess terminates naturally
    # When this happens, we exit from the application
    stop_event.wait()