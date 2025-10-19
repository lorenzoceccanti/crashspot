import streamlit as st
st.set_page_config(initial_sidebar_state="collapsed", layout="wide", page_title="CrashSpot")
import pandas as pd
from client import Client
from globals import handler

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
    options=('City', 'State'),
    index=None,
    placeholder="Select:",
)

# If a choiche between city/state has already been made
match granularityOptions:
    case 'City':
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

    case 'State':
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


if granularityOptions != None and causeOfAccidentOptions != None:
    if st.button("Discover hotspots", type="primary"):
        sts, hopkins, max_eps, df_labelled, df_performance = client.select_clustering_mode(algorithm, citiesOptions, stateOptions, causeOfAccidentOptions)
        handler.set_sts(sts)
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

if st.button("Quit", type="primary"):
    nav_to("about:blank")
    client.crashspot_stop()
