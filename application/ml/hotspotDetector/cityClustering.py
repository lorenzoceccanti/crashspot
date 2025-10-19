import numpy as np
import pandas as pd
from .geoclustering import Geoclustering # searches the module in the same folder
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from kneed import KneeLocator
from utility import Utility

class CityClustering(Geoclustering):
    """
    Class that implements the hotspot location for specified city
    """

    def __init__(self, df, cityName, accidentCause, k, minEps, stepEps, minPtsArr):

        """ Creates a new instance of the class CityClustering
        Args:
            self: A reference to the current object
            df: A pre-processed dataframe containing accident events records
            cityName: The name of the city to be considered in the analysis
        """

        # With this statement we are able to call the constructor of the class Geoclustering
        # We are redefining the constructor for Cityclustering
        df_city = df.query(f"city == '{cityName}' and general_cause_of_accident == '{accidentCause}'").copy()
        super().__init__(df_city, accidentCause)
        self.cityName = cityName
        self.k = k
        self.minEps = minEps
        self.stepEps = stepEps
        self.minPtsArr = minPtsArr
    
    def knee_heurstic_search(self):

        """ Estimates heuristically the Eps parameters. For each object (lat, lon)
        computes the distance with its k-th nearest neighbor, then orders the obtained
        distances in non-increasing order in order to construct the K-dist graph. Looks
        for the knee on the K-dist graph 
        
        Args:
            self: A reference to the current object
        
        Returns:
            An estimation for the value of Eps, None if any knee isn't found
        """

        Geoclustering.calculateRadians(self)
        coords_rad = self.get_arrRadians()
        # For each object, we consider its distance with the k-th nearest neighbor
        neigh = NearestNeighbors(n_neighbors=self.k, metric="haversine")
        neigh.fit(coords_rad)

        # The kneighbors method of neighb object returns the distance as the first
        # parameter and the index of the nearest point as second parameter
        # However, we are not interested in the second parameter
        
        distances_rad, _ = neigh.kneighbors(coords_rad)
        
        # the k-1 column in distances is the distance with the k-th nearest neighbor
        # : is to consider the distance between EACH point and that k-th neighbor
        # We consider a decreasing order, that is not implemented in numpy

        kth_rad = distances_rad[:,self.k-1]
        kth_km = kth_rad * Utility.getEarthRadius()

        kth_km_sorted = np.sort(kth_km)[::-1]

        # Looking for the knee of the K-distance graph
        x = np.arange(1, len(kth_km_sorted)+1)
        y = kth_km_sorted
        knee = KneeLocator(x, y, curve="convex", direction="decreasing")
        eps = y[knee.knee] if knee.knee is not None else None

        self.maxEps = eps
        return eps
    
    def run(self, eps_km, minPts):
        """ Performs DBSCAN clustering algorithm
        Args:
            self: A reference to the current object
            eps: The Eps parameter of DBSCAN expressed in km
            minPts: The minPts parameter of DBSCAN

        Returns:
            A dictionary containing the performance metrics 
            of the clustering algorithm or "No clusters" if no more than
            1 cluster is found
        """
        dbscan = DBSCAN(eps = eps_km/Utility.getEarthRadius(), 
                        min_samples=minPts, metric='haversine')
        dbscan.fit(self.get_arrRadians())
        
        self.arr_Labels = dbscan.labels_
        n_clusters = Geoclustering.get_numberOfClusters(self)
        if n_clusters <= 1:
            return "No clusters"
        dbscan_performance = {
            'eps_km': eps_km,
            'min_samples': minPts,
            'core_outlier_ratio': Geoclustering.get_coreOutlierRatio(self),
            'number_of_clusters': n_clusters,
            'davies_bouldin_index': Geoclustering.get_davies_bouldin_index(self),
            'silouette_coefficent': Geoclustering.get_silouette_coefficent(self),
            'calinski_index': Geoclustering.get_calinski_index(self)
        }
        return dbscan_performance

    def clustering_tuning(self):

        """ Selects the best combination of parameters to use.
        Args:
            self: A reference to the current object
        Returns:
            A dataframe containing the best 3 combination of paramters to use
        """
        arr_tuning_results = []

        eps_to_test = []
        for eps in np.arange(self.minEps, self.maxEps, self.stepEps):
            eps_to_test.append(eps)

        for eps in eps_to_test:
            for minPts in self.minPtsArr:
                perf = self.run(eps, minPts)
                if isinstance(perf, str) == False:
                    arr_tuning_results.append(perf)
        
        # Selecting the best performances
        tuning_results_df = pd.DataFrame(arr_tuning_results)
        tuning_results_df = tuning_results_df.sort_values(
            by = ["core_outlier_ratio", "silouette_coefficent", "davies_bouldin_index", "calinski_index"],
            ascending = [True, False, True, False]
        )
        df_returned = tuning_results_df.query("core_outlier_ratio > 1.4")[:3]
        if df_returned.shape[0] < 3:
            return -1, tuning_results_df[-3:]
        else:
            return 0, df_returned