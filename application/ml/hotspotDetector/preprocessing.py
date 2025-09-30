class Preprocessing:
    """ Class performing dimensionality, numerosity and class reduction """

    def __init__(self, df):
        """ Creates a new instance of the class Preprocessing
            Args:
                self: A reference to the current object
                df: The dataframe to be pre-processed
        """
        self.df = df.copy()
    
    def drop_columns(self, cols_to_keep):
        """ Drops all the columns but for those ones specified in cols_to_keep
            Args:
                self: A reference to the current object containing the dataframe
                cols_to_keep: A set of str containing the name of the attributes to keep
        """
        # We extract from the dataframe the list with all the columns names and we extract
        # them as a Python set object
        all_cols_names = set(self.df.columns.to_list())
        # The difference between the sets all_cols_names and cols_to_keep gives us
        # the set of the columns to drop
        cols_to_drop = all_cols_names.difference(cols_to_keep)
        # The pandas drop method wants a list, not a set.
        return self.df.drop(columns=list(cols_to_drop), inplace=True)
    
    def map_general_causes(self):
        """ The method works on the df belonging to the current Preprocessing instance
        in order to produce a new attribute general_cause_of_accident that encapsulates
        all the specific causes into wider ones in order to reduce the class numbers.
            Args:
                self: A reference to the current object containing the dataframe
        """

        # cause_mapping is a dictionary containing the association between the
        # specific causes we have (as value) and the general cause we want to assign to them
        cause_mapping = {
            "Brake slam": ["Abrupt use of the car's brake"],
            "Minor traffic offense": ["Absence of sinalization",
                                      "Disobedience to laws of transit by the pedestrian",
                                      "car's on sidewalk"],
            "Traffic offense": ["Driver broke the laws of transit", "Irregular access",
                                "Lane change maneuver",
                                "Stopping at a prohibited place",
                                "The driver passed the next car improperly",
                                "Traffic with a motorcycle (or similar) between lanes",
                                "Acessing the road without seeing the presence of other vehicles"],
            "Major traffic offense": ["Disrespecting the intersection", 
                                      "Driver changed the lane illegally",
                                      "Driver disrespected the red traffic light",
                                      "Driver was in the opposite direction",
                                      "Driving on the breakdown lane",
                                      "Prohibited conversion"],
            "Driver distraction": ["Driver using cellphone",
                                 "Driver was sleeping",
                                 "Driver's lack of reaction",
                                 "Driver's lack of attention to conveyance"],
            "Road defect":  ["Inadequate sinalization of the road",
                             "Curvy road", "No breakdown lanes", "Other flaws/problems in the road",
                             "Poor ilumination (of the road)",
                             "Road's defect",
                             "Roads with holes without cement",
                             "Sinking or ondulation in the pavement",
                             "Slippery track",
                             "Uneven breakdown lane",
                             "Unlevel track",
                             "Urban area without appropriate pedestrian walking"],
            "Road condition": ["Accumulation of water on the road", "Fog",
                               "Natural phenomena",
                               "Obstacle in the road",
                               "Oil accumulation on the road",
                               "Rain",
                               "Road had lots of sand/wreckage",
                               "Road works (in maintenance)",
                               "Static object on the drainage gate",
                               "Visibility restriction"],
            "Alcohol": ["Alcohol and/or drug ingestion by the pedestrian", "Alcohol consumption",
                        "Alcohol ingestion by the driver"],
            "Drugs": ["Driver was using drugs"],
            "Driver behavior": ["External fight"],
            "Animals": ["Animals on the road"],
            "Veichle not human fault": ["Car's brake problem", 
                               "Car's suspension system with problems", 
                               "Deficiency of vehicle's sinalization/ilumination system",
                               "Electrical or mechanical flaws",
                               "Mechanical loss/defect of vehicle"],
            "Veichle human fault": ["Excessive load/cargo", "Excessive use of the car's tire"],
            "Driver health": ["Cardiac attack", "Driver had a cardiac attack"],
            "Safe distance": ["Disrespect of safe distance from the next car",
                              "Driver failed to keep distance from the vehicle in front"],
            "High speed": ["Incompatible velocity"],
            "Pedestrian involved": ["Pedestrian was crossing the road outside of the crosswalk",
                                    "Pedestrian was walking in the road",
                                    "Pedestrian's lack of attention",
                                    "Unexpected pedestrian entry"]
        }

        # Since for Pandas it's more convenient to have the specific causes as key, we reverse the mapping of the dictionary
        reverse_mapping = {specific: general 
                   for general, specifics in cause_mapping.items() 
                   for specific in specifics}
        
        self.df["general_cause_of_accident"] = self.df["cause_of_accident"].map(reverse_mapping)

    def getDataframe(self):
        """ Getter method for the Preprocessing object"""
        return self.df
    
    def getGeneralCausesCity(self, city):
        """ Starting from the preprocessed dataset, returns a list containing the general causes of accident
        for a specific city"""
        df_sel = self.df.query(f"city == '{city}'")
        # Notice: the na values we drop are not so relevant, and they derive from all the
        # secondaries cause of accidents for which there is a very very small amount of events
        # For the sake of the convenience, these options are not even returned to the frontend
        return df_sel['general_cause_of_accident'].drop_duplicates().dropna().to_list()

    def getGeneralCausesState(self, state):
        """ Starting from the preprocessed dataset, returns a list containing the general causes of accident
        for a specific state"""
        df_sel = self.df.query(f"state == '{state}'")
        # Notice: the na values we drop are not so relevant, and they derive from all the
        # secondaries cause of accidents for which there is a very very small amount of events
        # For the sake of the convenience, these options are not even returned to the frontend
        return df_sel['general_cause_of_accident'].drop_duplicates().dropna().to_list()
    
    def getCities(self):
        """ Starting from the preprocessed dataset, returns a list containing the cities"""
        return list(self.df["city"].unique())
    
    def getStates(self):
        """ Starting from the preprocessed dataset, returns a list containing the states"""
        return list(self.df["state"].unique())
