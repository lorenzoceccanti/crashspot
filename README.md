<h2 align="center"> <b> <i> CrashSpot </i> </b> </h2>

<p align="center"> Project for DMML course @ DII UNIPI </p>

## Minimum requirements:

- A conda environment hosting a Python 3.11.7 instance
- The package listed in the <code>requirements</code> file. Once you have created a conda Python 3.11.7 environment run:
      <code>pip install -r requirements.txt</code>

## Suggested requirements
- In order to correctly visualize the graphs, if you are running the notebooks on Visual Studio Code make sure to have installed the <code> Jupyter Notebook Renderers </code> extension by Microsoft.

## Deployment instructions

- It's required to import an .env file in the root of the project, including the set of links pointing to the source datasets.
Contact Lorenzo Ceccanti via email for obtaining such env file.

- It's required to import an .env file to be included into ./application. The structure of that env file is the following:
<code>FLASK_PORT = your_flask_port
FLASK_ADDRESS = "a_flask_address"
STREAMLIT_PORT = your_streamlit_port</code>

- To retrieve the dataset run the <code> download_dataset </code> script and check for the logs.




Readme under construction.

Author: Lorenzo Ceccanti
