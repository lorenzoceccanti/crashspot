import os
import pandas as pd
import numpy as np

class DatasetLoader:
    """ The DatasetLoader class performs all the IO operations related to csv files"""
    
    def __init__(self, full_dataset_path, clean_dataset_path):
        self.full_dataset_path = full_dataset_path
        self.clean_dataset_path = clean_dataset_path
    
    def produce_clean_dataset(self):
        """ Produces the cleaned version of the brasilian aggregated dataset"""
        # The motivation for the following processing steps can be find in the dataCleaning notebook
        full_dataset_df = pd.read_csv(self.full_dataset_path)
        
        full_dataset_df.rename(columns={"ignored": "unharmed"}, inplace=True)
        full_dataset_df.rename(columns={"inverse_data": "date"}, inplace=True)
        full_dataset_df.rename(columns={"wheather_condition": "weather_condition"}, inplace=True)

        full_dataset_df["road_id"] = full_dataset_df["road_id"].apply(lambda x: str(int(x)) if not pd.isna(x) else None)

        full_dataset_df = full_dataset_df.dropna(subset=['police_station', 'regional', 'road_id', 'km'])

        full_dataset_df.replace('Not informed', np.nan).dropna(subset=['type_of_accident'])

        full_dataset_df.drop(columns = ['road_delineation'], inplace=True)

        df_invalid = full_dataset_df[
            (full_dataset_df['latitude'] > 90) | (full_dataset_df['latitude'] < -90) |
            (full_dataset_df['longitude'] > 180) | (full_dataset_df['longitude'] < -180)
        ]

        self.clean_dataset_df = full_dataset_df.drop(df_invalid.index)
        self.clean_dataset_df.to_csv(self.clean_dataset_path, index=False, encoding='utf-8')
    
    def loadDataset(self):
        """ Loads the dataset pointed by the path in the json configuration file into
        a Pandas Dataframe
        
        Args:
            jsonName: The full path of the JSON configuration file.

        Returns:
            A Pandas Dataframe containing the whole dataset
        """
        
        # Checking the correctness of the config.json file
        if self.clean_dataset_path == -1 and self.full_dataset_path == 1:
            print("(1) Invalid JSON format. Exiting.")
            return
        
        if not os.path.exists(self.clean_dataset_path):
            # If the clean dataset doesn't exist, the pre-processing has to be performed
            if self.full_dataset_path != -1:
                if not os.path.exists(self.full_dataset_path):
                    print("No dataset found. Exiting.")
                    return
                print("Performing pre-processing...")
                # The method works on the implicit class object
                self.produce_clean_dataset()
                return self.clean_dataset_df
            else:
                print("(2) Invalid JSON format. Exiting")
                return
        else:
            self.clean_dataset_df = pd.read_csv(self.clean_dataset_path)
        return self.clean_dataset_df