from utility import Utility

class ConfigLoader:
    """ It's responsible to parsing the fields on the JSON configuration file"""

    def __init__(self, path):
        """ Parses the fields on the JSON configuration file
            Args:
                path: The relative path of the json configuration file
        """
        self.utility = Utility()
        json = self.utility.read_json(path)
        self.full_dataset  = json.get("full_dataset", -1)
        self.clean_dataset = json.get("clean_dataset", -1)
        self.k_distGraph   = json.get("k_distGraph", -1)
        self.dbscan_minEps = json.get("dbscan_minEps", -1)
        self.dbscan_stepEps = json.get("dbscan_stepEps", -1)
        self.dbscan_minPtsArr = json.get("dbscan_minPtsArr", -1)
        self.optics_maxRadiusArr = json.get("optics_maxRadiusArr", -1)
        self.optics_minPtsArr = json.get("optics_minPtsArr", -1)
        self.optics_xiArr = json.get("optics_xiArr", -1)
    
    def getFullDataset(self):
        return self.full_dataset
    
    def getCleanDataset(self):
        return self.clean_dataset
    
    def getKDistGraph(self):
        return self.k_distGraph
    
    def get_dbscanMinEps(self):
        return self.dbscan_minEps
    
    def get_dbscanStepEps(self):
        return self.dbscan_stepEps
    
    def get_dbscanMinPtsArr(self):
        return self.dbscan_minPtsArr
    
    def get_opticsMaxRadiusArr(self):
        return self.optics_maxRadiusArr
    
    def get_opticsMinPtsArr(self):
        return self.optics_minPtsArr
    
    def get_opticsXiArr(self):
        return self.optics_xiArr