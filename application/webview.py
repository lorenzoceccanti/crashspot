import os
from dotenv import load_dotenv
import subprocess
import sys
class WebView:

    def __init__(self):
        load_dotenv()
        STREAMLIT_PORT = os.getenv('STREAMLIT_PORT')
        """ Launches the frontend process responsible for visualizing the data"""
        app_path = os.path.join(os.path.dirname(__file__),"./frontend/home.py")
        self.port = str(STREAMLIT_PORT)
        # Launching Streamlit as a subprocess
        # -m is needed to launch the streamlit as a library installed
        # in the current environment
        self.proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", app_path, "--server.port", self.port])
    

    # We transform the WebView class in a Context manager class, so as an object that is instantiated with the
    # construct with

    # The enter method is executed when we enter inside the with block
    # It returns the reference to the resource initialized
    def __enter__(self):
        print(f"Streamlit server started on port {self.port}")
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        # We check that the process was launched and that is still in execution
        # before trying to end it
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()