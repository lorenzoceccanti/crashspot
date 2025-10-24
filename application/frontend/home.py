import os
import time
import streamlit as st
st.set_page_config(initial_sidebar_state="collapsed", layout="wide", page_title="CrashSpot")
import pandas as pd
import numpy as np
from client import Client
from globals import handler
from application.utility import Utility
from application.ml.severityPrediction.preprocessing import Preprocessing

def nav_to(url):
    nav_script = """
        <meta http-equiv="refresh" content="0; url='%s'">
    """ % (url)
    st.write(nav_script, unsafe_allow_html=True)

def get_absolute_path(relative_path):
    current_dir = os.path.dirname(__file__)
    path = os.path.normpath(os.path.join(current_dir, relative_path))
    return path

def read_confidence_threshold():
    path = get_absolute_path("../config.json")
    utility = Utility()
    return utility.read_json(path, mode="field", field="predict_proba_threshold")

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
        if isinstance(file_df, int):
            st.toast(":red[ERROR. Please provide a compliant dataset]")
            time.sleep(2)
            st.rerun()
        
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
            X_test = file_df
            X_test_fs = X_test[sel_features]
            feature_to_disp = ['date', 'week_day', 'hour', 'city',
                               'person_age', 'general_veichle_brand', 'type_of_accident']
            X_test_disp = X_test[feature_to_disp]
            # X_test_fs is what we'll send as payload
            predictions_df = client.classify(X_test_fs)
            output_df = X_test_disp.join(predictions_df, how='inner')
            
            # We save the dataframe in the Streamlit persitent memory (the session state)
            st.session_state["prediction_output_df"] = output_df
            # We save also as a session variable a list long as many are the tuple to show
            # containing the state of the selections that the user is making in the webpage
            # It's the memory state for what is rendered in the frontend
            st.session_state["selection_col"] = [""] * len(output_df)
            
            # We initialize some selections depending on the threshold for the model
            output_df["pre_selection"] = ""
            threshold = read_confidence_threshold()
            mask = output_df["confidence"] > threshold
            output_df.loc[mask & (output_df["predicted"] == "Unharmed"), "pre_selection"] = "No intervention"
            output_df.loc[mask & (output_df["predicted"] == "Severe"), "pre_selection"] = "Doctor"
            output_df.loc[mask & (output_df["predicted"] == "Injured"), "pre_selection"] = "Paramedics"

            st.session_state["selection_col"] = output_df["pre_selection"].tolist()
            # We create for later usage a dictionary which maps the pandas index and
            # the choice made for that instance
            st.session_state["selection_result"] = {}
            output_df.drop('pre_selection', axis=1, inplace=True)

if "prediction_output_df" in st.session_state:
    output_df = st.session_state["prediction_output_df"]

    display_df = output_df.copy()

    # This block of code guarantees a correct persistence of the ComboBox selection
    n = len(display_df)
    # If a selection has already been made, we display it
    sel_state = st.session_state.get("selection_col", [""] * n)
    # In particular it might happen that the selection made up so far 
    # are not complete (most of the times this happens)
    # At the next page rendering we adjust the list of choices to the correct
    # lenght: we add white spaces for the selections that are not done yet
    if len(sel_state) != n:
        sel_state = (sel_state[:n] + [""] * max(0, n - len(sel_state)))
    
    st.session_state["selection_col"] = sel_state

    CHOICES = ["", "No intervention", "Paramedics", "Doctor"]

    with st.form("selection_form", clear_on_submit=False):
        show_df = display_df.copy()
        show_df.insert(0, "selection", st.session_state["selection_col"])

        edited = st.data_editor(
            show_df,
            width='stretch',
            hide_index=False,
            column_config={
                "selection": st.column_config.SelectboxColumn(
                    "selection",
                    options=CHOICES,
                    default="",
                    help="Select a value",
                )
            },
            disabled=[c for c in show_df.columns if c != "selection"],
            key="editor_selection_in_form",
        )

        submitted = st.form_submit_button("Update choices")
    
    if submitted:
        st.session_state["selection_col"] = edited["selection"].tolist()
        st.rerun()

    if "selection_col" in st.session_state:
        sel_series = pd.Series(st.session_state["selection_col"], index=display_df.index)
        mask = sel_series.ne("")
        selected_indices = sel_series[mask].index.tolist()
        selected_values  = sel_series[mask].tolist()
        st.session_state["selection_result"] = {
            "indices": selected_indices,
            "values": selected_values,
        }


    if st.session_state.get("selection_result"):
        res = st.session_state["selection_result"]
        display_df.insert(0, "selection", pd.Series(st.session_state["selection_col"], index=display_df.index))
        combobox_df = display_df.loc[res["indices"]]
        path = get_absolute_path(f"../../choices/") # path of the folder
        if not os.path.exists(path):
            os.makedirs(path)
        path += f"/{uploaded_file.name}" # path of the folder becomes the path of the file
        st.toast(f":green[Updates saved into]: ./choices/{uploaded_file.name}")
        combobox_df.to_csv(path, index=False)

if st.button("Quit", type="primary"):
    nav_to("about:blank")
    client.crashspot_stop()