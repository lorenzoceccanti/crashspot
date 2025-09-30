import json
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

class Utility:
    
    def read_json(self, path: str, mode="all", field=""):
        """
        Reads a JSON file at the specified path.
        
        Args:
            path: the fully-qualified path name for the json file

            mode: Optional field. If mode == "all", then the whole json
            document is read (default). If mode == "field", then only the
            specific field of the JSON document will be read and returned
            as a string.

        Returns:
            If mode == "all", a dictionary containing all the fields of the json document
            If mode == "field", the value associated to field specified by field.
            -1 otherwise.
        """
        try:
            with open(path, 'r') as file:
                data = json.load(file)
            if mode == "field":
                return data[field] if field in data else -1
            elif mode == "all":
                return data if data else -1
            else:
                return -1
        except FileNotFoundError:
            print(f"[ERR]: File {path} was not found.")
            return -1
    
    def getEarthRadius():

        """Returns the Earth radius in kilometers"""

        return 6371.0
    
    def hopkins(data_frame, sampling_size):
        """ Adaptation of the hopkins method from pyclustend
        in which a prior conversion to radians and haversine distance
        is used
        
        Original code: https://pyclustertend.readthedocs.io/en/latest/_modules/pyclustertend/hopkins.html#hopkins

        """

        if type(data_frame) == np.ndarray:
            data_frame = pd.DataFrame(data_frame)
        
        # Sample n observations from D : P
        if sampling_size > data_frame.shape[0]:
            raise Exception(
            'The number of sample of sample is bigger than the shape of D')
        
        data_frame = np.radians(data_frame)
        data_frame_sample = data_frame.sample(n=sampling_size)

        # Get the distance to their neirest neighbors in D : X

        tree = BallTree(data_frame, leaf_size=2, metric='haversine')
        dist, _ = tree.query(data_frame_sample, k=2)
        data_frame_sample_distances_to_nearest_neighbours = dist[:, 1]

        # Randomly simulate n points with the same variation as in D : Q.

        max_data_frame = data_frame.max()
        min_data_frame = data_frame.min()

        uniformly_selected_values_0 = np.random.uniform(min_data_frame[0], max_data_frame[0], sampling_size)
        uniformly_selected_values_1 = np.random.uniform(min_data_frame[1], max_data_frame[1], sampling_size)

        uniformly_selected_observations = np.column_stack((uniformly_selected_values_0, uniformly_selected_values_1))
        if len(max_data_frame) >= 2:
            for i in range(2, len(max_data_frame)):
                uniformly_selected_values_i = np.random.uniform(min_data_frame[i], max_data_frame[i], sampling_size)
                to_stack = (uniformly_selected_observations, uniformly_selected_values_i)
                uniformly_selected_observations = np.column_stack(to_stack)
        
        uniformly_selected_observations_df = pd.DataFrame(uniformly_selected_observations)

        # Get the distance to their neirest neighbors in D : Y

        tree = BallTree(data_frame, leaf_size=2, metric='haversine')
        dist, _ = tree.query(uniformly_selected_observations_df, k=1)
        uniformly_df_distances_to_nearest_neighbours = dist

        # return the hopkins score

        x = sum(data_frame_sample_distances_to_nearest_neighbours)
        y = sum(uniformly_df_distances_to_nearest_neighbours)

        if x + y == 0:
            raise Exception('The denominator of the hopkins statistics is null')
        
        return x / (x + y)[0]
