text = """"
*** RESET CLASSIFICATION RESULTS ***
*** 0) RESET ALL
*** 1) RandomForest
*** 2) AdaBoost
*** 3) XGBoost
*** 4) LightGBM
*************************************
Make your choice:
"""
selection = input(text)
tokens = selection.split() # an array of selections

actions = {
    "0": lambda: print("Resetting all ..."),
    "1": lambda: print("Resetting RandomForest ..."),
    "2": lambda: print("Resetting AdaBoost ..."),
    "3": lambda: print("Resetting XGBoost ..."),
    "4": lambda: print("Resetting LightGBM ..."),
}

for token in tokens:
    action = actions.get(token)
    if action:
        action()
    else:
        print(f"Not valid selection: {token}")