import os
import shutil

text = """"
*** RESET CLASSIFICATION RESULTS ***
*** 0) RESET ALL
*** 1) RandomForest
*** 2) AdaBoost
*** 3) XGBoost
*** 4) LightGBM
*** 99) Exit - NO ACTIONS
*************************************
Make your choice:
"""

def clean_result_directory(algorithmName):
    dirname = f"./notebooks/reports_{algorithmName}"
    if not os.path.exists(dirname):
        print(f"[{algorithmName}] Nothing to remove.")
    else:
        print(f"Resetting {algorithmName} ..")
        shutil.rmtree(dirname)

def clean_all():
    clean_result_directory("RF")
    clean_result_directory("ADA")
    clean_result_directory("XG")
    clean_result_directory("LGB")

selection = input(text)
tokens = selection.split() # an array of selections

actions = {
    "0": lambda: clean_all(),
    "1": lambda: clean_result_directory("RF"),
    "2": lambda: clean_result_directory("ADA"),
    "3": lambda: clean_result_directory("XG"),
    "4": lambda: clean_result_directory("LGB"),
    "99": lambda: print("Exiting ...")
}

if "99" in tokens:
    action = actions.get("99")
    action()
else:
    for token in tokens:
        action = actions.get(token)
        if action:
            action()
        else:
            print(f"Not valid selection: {token}")