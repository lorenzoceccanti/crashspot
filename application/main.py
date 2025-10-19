import os
import threading
from dotenv import load_dotenv
from flask import Flask
from flask import request
from ml.hotspotDetector.datasetLoader import DatasetLoader
from ml.hotspotDetector.preprocessing import Preprocessing
from ml.hotspotDetector.cityClustering import CityClustering
from ml.hotspotDetector.stateClustering import StateClustering
from ml.hotspotDetector.configloader import ConfigLoader
from webview import WebView

load_dotenv()
FLASK_PORT = os.getenv('FLASK_PORT')

stop_event = threading.Event()
app = Flask(__name__)

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