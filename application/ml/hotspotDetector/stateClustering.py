from .geoclustering import Geoclustering
from sklearn.cluster import OPTICS
from utility import Utility
import pandas as pd

class StateClustering(Geoclustering):
    """
    Class that implements the hotspot location for specified city
    """

    def __init__(self, df, stateName, accidentCause, maxRadiusArr, minPtsArr, xiArr):
        # We are overriding the constructor of Geoclustering
        df_state = df.query(f"state == '{stateName}' and general_cause_of_accident == '{accidentCause}'").copy()
        super().__init__(df_state, accidentCause)
        self.stateName = stateName
        self.maxRadiusArr = maxRadiusArr
        self.minPtsArr = minPtsArr
        self.xiArr = xiArr
    
    def run(self, max_eps, minPts, xi):
        """" Performs OPTICS clustering algorithm"""

        # If we use OPTICS, we cannot maintain multiple accidents characterized by the *exact* 
        # couple (lat, lon)! In that case, 
        # we would obtain 0 in the computation of the reachability distance. 
        # This is one of the known issue of the OPTICS algorithm. 
        # For this reason, to fit OPTICS I'll pass a dataframe without duplicates.
        coords = Geoclustering.calculateRadians(self, format = 'pd')

        # aggr contains a tuple of this kind
        # lat | lon | count
        aggr = (coords
            .groupby(["latitude","longitude"], as_index=False)
            .size().rename(columns={"size":"count"}))
        np_coords = aggr[["latitude", "longitude"]].to_numpy()
        
        # min_samples defines the minimum density: how many neighbors are required to consider a point as a core poin
        optics = OPTICS(min_samples=minPts, max_eps=max_eps/Utility.getEarthRadius(), metric='haversine', xi=xi)
        optics.fit(np_coords)

        aggr["label"] = optics.labels_
        # Re-assigning the labels to the original data
        gps_labeled = coords.merge(
            aggr[["latitude","longitude","label","count"]],
            on=["latitude","longitude"],
            how="left"
        )

        self.arr_Labels = gps_labeled['label'].to_numpy()
        
        n_clusters = Geoclustering.get_numberOfClusters(self)
        if n_clusters <= 1:
            return "No clusters"
        optics_performance = {
            'min_samples': minPts,
            'max_radius': max_eps,
            'xi': xi,
            'core_outlier_ratio': Geoclustering.get_coreOutlierRatio(self),
            'number_of_clusters': n_clusters,
            'davies_bouldin_index': Geoclustering.get_davies_bouldin_index(self),
            'silouette_coefficent': Geoclustering.get_silouette_coefficent(self),
            'calinski_index': Geoclustering.get_calinski_index(self)
        }
        return optics_performance
    
    def clustering_tuning(self):
        """ Selects the best combination of parameters to use.
        Args:
            self: A reference to the current object
        Returns:
            A dataframe containing the best 3 combination of paramters to use
        """

        arr_tuning_results = []

        for max_eps in self.maxRadiusArr:
            for minPts in self.minPtsArr:
                for xi in self.xiArr:
                    perf = self.run(max_eps, minPts, xi)
                    if isinstance(perf, str) == False:
                        arr_tuning_results.append(perf)
        
        if len(arr_tuning_results) == 0:
            return -1, None

        # Selecting the best performances
        tuning_results_df = pd.DataFrame(arr_tuning_results)
        tuning_results_df = tuning_results_df.sort_values(
            by = ["core_outlier_ratio", "silouette_coefficent", "davies_bouldin_index", "calinski_index"],
            ascending = [True, False, True, False]
        )
        df_returned = tuning_results_df.query("core_outlier_ratio > 1.4")[:3]
        # Sometimes it may happen that it's not possible to obtain a good core_outlier_ratio
        # In that case we inform the frontend and we return the top 3 highest metrics in terms
        # of descending core_outlier_ratios
        if df_returned.shape[0] < 3:
            return 1, tuning_results_df[-3:]
        else:
            return 0, df_returned