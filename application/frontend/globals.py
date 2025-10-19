import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# Contains the handler object to be shared among the pages

class Handler:
    def __init__(self):
        pass
    def set_dataframe(self, df):
        self.df = df
    def set_clustering_perf(self, df):
        self.df_perf = df
    def set_sts(self, sts):
        self.sts = sts
    def set_graph_title(self, title):
        self.graph_title = title
    def set_hopkins_max_eps(self, hopkins, max_eps):
        self.hopkins = hopkins
        self.max_eps = max_eps
    def compute_hotspot_score(self, alpha=0.2, beta=0.8):
        """ This method produces a new dataframe, clusters_stats,
        containing for each cluster: the number of accidents, the sum
        of numerical victims condition ranks, the hotspot score so an index
        ranging from 0 and 1 suggesting """
        
        group = self.df.groupby("label")
        # This is a series containing the number of accidents
        # events for each cluster
        accidents_per_label = group.size().reset_index(name="accidents_per_label")
        # Here we produce a column in which we sum the numerical ranks associated
        # to the victim_condition. Higher sums suggests "bad" clusters
        sum_victims_condition_rank = group["victims_condition_rank"].sum().reset_index(name="sum_victims_condition_rank")
        clusters_stats = pd.merge(accidents_per_label, sum_victims_condition_rank, on="label")
        
        # Here we compute the hotspot score
        clusters_stats['hotspot_score'] = alpha*clusters_stats['accidents_per_label'] + beta*clusters_stats['sum_victims_condition_rank']
        scaler = MinMaxScaler()

        clusters_stats["hotspot_score"] = scaler.fit_transform(clusters_stats[["hotspot_score"]])
        self.clusters_stats = clusters_stats
    def get_dataframe(self):
        return self.df
    def get_clustering_perf(self):
        return self.df_perf
    def get_clusters_stats(self):
        return self.clusters_stats
    def get_sts(self):
        return self.sts
    def get_graph_title(self):
        return self.graph_title
    def get_hopkins_max_eps(self):
        return self.hopkins, self.max_eps

handler = Handler()