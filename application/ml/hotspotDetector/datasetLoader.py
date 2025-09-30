import os
import pandas as pd

class DatasetLoader:
    """ The DatasetLoader class performs all the IO operations related to csv files"""
    
    def __init__(self, full_dataset_path, clean_dataset_path):
        self.full_dataset_path = full_dataset_path
        self.clean_dataset_path = clean_dataset_path
    
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
            else:
                print("(2) Invalid JSON format. Exiting")
                return
        else:
            self.clean_dataset_df = pd.read_csv(self.clean_dataset_path)
        return self.clean_dataset_df