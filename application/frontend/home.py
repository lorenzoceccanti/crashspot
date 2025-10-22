import streamlit as st
st.set_page_config(initial_sidebar_state="collapsed", layout="wide", page_title="CrashSpot")
import pandas as pd
import numpy as np
from client import Client
from globals import handler
from application.ml.severityPrediction.preprocessing import Preprocessing

def nav_to(url):
    nav_script = """
        <meta http-equiv="refresh" content="0; url='%s'">
    """ % (url)
    st.write(nav_script, unsafe_allow_html=True)

# Page title
st.markdown("<h1 style='text-align: center;'><em>CrashSpot</em></h1>", unsafe_allow_html=True)

client = Client(st)

# We have to exploit the onchange callback of the selectbox method
granularityOptions = st.selectbox(
    label = "Granularity selection:",
    options=('Hotspot Detection - City', 'Hotspot Detection - State', 'Accident severity prediction'),
    index=None,
    placeholder="Select:",
)

# If a choiche between city/state has already been made
match granularityOptions:
    case 'Hotspot Detection - City':
        cities_list = client.get_cities()
        citiesOptions = st.selectbox(
            label = "Pick a city:",
            options=cities_list,
            index = None,
            placeholder="Select.."
        )
        algorithm = "DBSCAN"
        stateOptions = None

        accident_causes_list = client.get_general_cause_of_accident(f"city={citiesOptions}")
        causeOfAccidentOptions = st.selectbox(
            label = "Cause of accident:",
            options=accident_causes_list,
            index = None,
            placeholder="Select.."
        )

    case 'Hotspot Detection - State':
        state_list = client.get_states()
        stateOptions = st.selectbox(
            label = "Pick a state:",
            options=state_list,
            index = None,
            placeholder="Select.."
        )
        algorithm = "OPTICS"
        citiesOptions = None

        accident_causes_list = client.get_general_cause_of_accident(f"state={stateOptions}")
        causeOfAccidentOptions = st.selectbox(
            label = "Cause of accident:",
            options=accident_causes_list,
            index = None,
            placeholder="Select.."
        )
    case 'Accident severity prediction':
        causeOfAccidentOptions = None
        # Load the test set file (without the classes, they represent real data
        # about incoming accidents)
        uploaded_file = st.file_uploader("Load a CSV file", type=["csv"])


if granularityOptions != None and causeOfAccidentOptions != None:
    if st.button("Discover hotspots", type="primary"):
        sts, hopkins, max_eps, df_labelled, df_performance = client.select_clustering_mode(algorithm, citiesOptions, stateOptions, causeOfAccidentOptions)
        handler.set_sts(sts)
        if sts != -1:
            handler.set_dataframe(df_labelled)
            handler.set_clustering_perf(df_performance)
            handler.set_hopkins_max_eps(hopkins, max_eps)
            if citiesOptions != None:
                handler.set_graph_title(f"{causeOfAccidentOptions} accidents distribution - City: {citiesOptions}")
            if stateOptions != None:
                handler.set_graph_title(f"{causeOfAccidentOptions} accidents distrubtion  - State: {stateOptions}")
            # Computation of the hotspot score
            handler.compute_hotspot_score()
        st.switch_page("pages/graph.py")

if granularityOptions != None and uploaded_file != None:
    if st.button("Predict", type="primary"):
        preprocessing = Preprocessing(uploaded_file)
        file_df = preprocessing.get_df()
        
        # Get the list of selected features (here call the backend) from the joblib
        # Here we have to send only the test set with the selected features as payload
        # As a response we'll get again the test set records along with the predictions
        # and the predict probas
        
        sel_features = client.get_features()
        # We expect to receive a ndarray, if we receive something else
        # it means that there was be a problem with the joblib model
        if not isinstance(sel_features, np.ndarray):
            st.error('Error. Model not found!')
        else:
            X_test_fs = file_df[sel_features]
            st.dataframe(X_test_fs)
            # X_test_fs is what we'll send as payload
            predictions_df = client.classify(X_test_fs)
            st.dataframe(predictions_df)


if st.button("Quit", type="primary"):
    nav_to("about:blank")
    client.crashspot_stop()
    st.stop()