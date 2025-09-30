import os
import io
import pandas as pd
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
             
             hopkins = data["hopkins"]
             max_eps = data["max_eps"]
             df_labelled = pd.read_json(io.StringIO(data["dfLabelled"]))
             df_perf = pd.read_json(io.StringIO(data["clusteringPerf"]))

             return hopkins, max_eps, df_labelled, df_perf
        except Exception as e:
            print(f"FLASK connection error: {e}")
            