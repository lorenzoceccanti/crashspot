import streamlit as st
from globals import handler
from client import Client
import plotly.express as px
import pandas as pd

def nav_to(url):
    nav_script = """
        <meta http-equiv="refresh" content="0; url='%s'">
    """ % (url)
    st.write(nav_script, unsafe_allow_html=True)

def plot_map(data: pd.DataFrame, text: str, clusters_stats):
    cont_scale = [
        [0.0,  "lightgrey"],
        [0.2,  "green"],
        [0.4,  "limegreen"],
        [0.6,  "yellow"],
        [0.75, "orange"],
        [1.0,  "red"]
    ]
    fig = px.scatter_map(
        data,
        lat="latitude",
        lon="longitude",
        color = data["label"].map(clusters_stats.set_index("label")["hotspot_score"]),
        color_continuous_scale=cont_scale
    )
    fig.update_layout(
        mapbox_style="carto-positron",
        title=f"Map plot ({text})"
    )
    return fig

st.set_page_config(page_title="Clustering results")
client = Client(st)

if handler.get_sts() != -1:
    st.title("CrashSpot - Clustering results")
    if handler.get_sts() == 1:
        st.text("Attention. The algorithm detected an high presence of outliers. Results may not be reliable.")
    st.text("Best parameters fit:")
    st.dataframe(handler.get_clustering_perf())

    hop, me = handler.get_hopkins_max_eps()
    st.text(f"Hopkins index: {hop:.3f}")
    st.text(f"Max eps advised (according to heuristic): {me:.3f}")

    plot = plot_map(handler.get_dataframe(), text=handler.get_graph_title(), clusters_stats=handler.get_clusters_stats())
    st.plotly_chart(plot)
else:
    st.title("No clusters.")

if st.button("Go back", type="primary"):
    st.switch_page("home.py")

if st.button("Quit app", type="primary"):
    nav_to("about:blank")
    client.crashspot_stop()