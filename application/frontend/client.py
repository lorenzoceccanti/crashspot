import os
import io
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import requests

class Client:
    def __init__(self, st):
        load_dotenv()
        FLASK_PORT = os.getenv('FLASK_PORT')
        FLASK_ADDRESS = os.getenv('FLASK_ADDRESS')
        self.api_address = f"http://{FLASK_ADDRESS}:{FLASK_PORT}"

        self.st = st
    
    def crashspot_stop(self):
        try:
            API_URL = self.api_address + "/crashspot_stop"
            requests.post(API_URL)
        except Exception as e:
            self.st.error(f"Error in closing the application: {e}")
    
    def get_general_cause_of_accident(self, param):
        try:
            API_URL = self.api_address + f"/general_cause_of_accident?{param}"
            response = requests.get(API_URL)
            return response.json()["causes_list"]
        except Exception as e:
            print(f"FLASK connection error: {e}")
    
    def get_cities(self):
        try:
            API_URL = self.api_address + "/cities"
            response = requests.get(API_URL)
            return response.json()["cities_list"]
        except Exception as e:
            print(f"FLASK connection error: {e}")
    
    def get_states(self):
        try:
            API_URL = self.api_address + "/states"
            response = requests.get(API_URL)
            return response.json()["states_list"]
        except Exception as e:
            print(f"FLASK connection error: {e}")
    
    def select_clustering_mode(self, algorithm, city, state, cause):
        """ Returns both the labelled dataframe and the clustering performances
        responses to the frontend"""
        try:
             API_URL = self.api_address + "/clustering"
             payload = {
                 "algorithm": algorithm,
                 "city": city if city is not None else "",
                 "state": state if state is not None else "",
                 "cause": cause
             }
             response = requests.post(API_URL, json=payload)
             data = response.json()
             
             sts = data["sts"]
             if sts == -1: # in this case the other fields are not returned by the backend, because no cluster exists
                 return sts, None, None, None, None
             hopkins = data["hopkins"]
             max_eps = data["max_eps"]
             df_labelled = pd.read_json(io.StringIO(data["dfLabelled"]))
             df_perf = pd.read_json(io.StringIO(data["clusteringPerf"]))

             return sts, hopkins, max_eps, df_labelled, df_perf
        except Exception as e:
            print(f"FLASK connection error: {e}")

    def get_features(self):
        """ Returns to the frontend the list of features selected by the model
        at fit time. Returns -1 if the model is not found, otherwise an
        ndarray containing the selected features"""
        try:
            API_URL = self.api_address + "/selected_features"
            response = requests.get(API_URL)
            if response.status_code == 500:
                return -1
            data = response.json()
            list = data["selected_features"]
            selected_features = np.array(list)
            return selected_features
        except Exception as e:
            print(f"FLASK connection error: {e}")
    
    def classify(self, X_test_fs):
        """ Sends the test set to the backend, with the desired features
        by the model. Returns a dataframe with a pandas index, predicted label
        and confidence value of the class predicted by the classifier"""
        try:
            API_URL = self.api_address + "/classify"
            payload = {
                "X_test_fs": X_test_fs.to_json()
            }
            response = requests.post(API_URL, json=payload)
            data = response.json()
            prediction_df = pd.read_json(io.StringIO(data['prediction_df']))
            return prediction_df
        except Exception as e:
            print(f"FLASK connection error: {e}")