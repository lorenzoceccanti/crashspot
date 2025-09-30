import numpy as np
from sklearn import metrics
from utility import Utility

class Geoclustering:

    def __init__(self, df, accidentCause):
        """ Creates a new instance of the class Geoclustering
         During the instantiation, the dataframe inside the implict object
         is created as the projection of the attributes latitude and longitude only.
        Args:
            self: A reference to the current object
            df: The pruned dataframe
        """
        self.df_labelled = df.copy()
        self.df_coordsGPS = df[["latitude", "longitude"]].copy()
        self.accidentCause = accidentCause
        # I require also:
        # df_coordsGPSLabeled
        # arr_Labels

    def get_dfCoordsGPS(self):
        return self.df_coordsGPS
    
    def get_arrRadians(self):
        return self.arr_Radians
    
    def add_victims_condition_rank(self):
        """ Adds a new column victims_conditions_rank to the labelled dataframe
        The rank is 0 for the accidents without victims, 1 for accidents
        with injured victims and 2 for accidents with dead victims
        """
        victims_condition_map = {
            "Without victims": 0,
            "With injured victims": 1,
            "With dead victims": 2
        }
        self.df_labelled['victims_condition_rank'] = self.df_labelled['victims_condition'].map(victims_condition_map)

    def get_dfLabelled(self):
        """ Retutns the dataframe ready to be serialized
        Only latitude, longitude, label, victims_condition, road_id and km
        are kept in the dataframe """
        return self.df_labelled[['latitude', 'longitude', 'victims_condition', 'victims_condition_rank', 'label', 'road_id', 'km']]
    
    def calculateRadians(self, format='np'):
        """ Creates a new inner object containing the coordinates in radians
        to be passed to the clustering algorithm
        The format parameter, if equal to pd forces the method to return a copy
        of the object as a dataframe"""
        self.arr_Radians = np.radians(self.df_coordsGPS).to_numpy()
        if format == 'pd':
            return np.radians(self.df_coordsGPS)
    
    def getHopkins(self):
        """ Returns the Hopkins index for the data to be clustered
        A score tending to 0 expresses high clustering tendency."""
        X = self.df_coordsGPS.copy()
        dim = X.shape[0]
        if dim < 50:
            n_samples = dim
        else:
            n_samples = 50
        # The conversion is necessary since hopkins() expects an ndarray as first parameter
        H = Utility.hopkins(X.to_numpy(), n_samples)
        return H
        
    
    def get_numberOfClusters(self):
        """ Returns the number of clusters found by the clustering algorithm.
        """
        return len(set(self.arr_Labels[self.arr_Labels >= 0]))
    
    def get_coreOutlierRatio(self):
        """ Returns the ratio between the number of core points and the numbers
        of outliers found by the clustering algorithm"""
        
        mask = self.arr_Labels != -1
        count_core_points = np.sum(mask)
        total_points = np.size(mask)
        count_outliers = total_points - count_core_points

        # Protection against division by zero
        # It may happen to have 0 outliers
        # In that case we set the ratio equal to 10k
        if count_outliers == 0:
            return 10000
        
        return count_core_points/count_outliers
    
    def get_davies_bouldin_index(self):
        """ Returns the Davies-Boudlin Index found by the clustering algorithm"""
        
        mask = self.arr_Labels != -1
        data_filtered = self.arr_Radians[mask]
        labels_filtered = self.arr_Labels[mask]
        
        return metrics.davies_bouldin_score(data_filtered, labels_filtered)
    def get_silouette_coefficent(self):
        """ Returns the Silouette Coefficent found by the clustering algorithm"""

        mask = self.arr_Labels != -1
        data_filtered = self.arr_Radians[mask]
        labels_filtered = self.arr_Labels[mask]

        return metrics.silhouette_score(data_filtered, labels_filtered, metric='haversine')
    
    def get_calinski_index(self):
        """ Returns the Calinski-Harabasz index found by the clustering algorithm"""

        mask = self.arr_Labels != -1
        data_filtered = self.arr_Radians[mask]
        labels_filtered = self.arr_Labels[mask]

        return metrics.calinski_harabasz_score(data_filtered, labels_filtered)

    def attachLabels(self):
        """ Creates a dataframe in which the GPS coordinates and the labels
        (different from -1) are juxtaposed"""
        self.df_labelled['label'] = self.arr_Labels
        self.df_labelled = self.df_labelled.query('label != -1')