import os
import zipfile
import gdown
from dotenv import dotenv_values

def load_env():
    env = dotenv_values(".env")
    prefix_raw = "BRASIL_RAW_" # years from 2017 to 2013
    prefix_extra = "BRASIL_EXTRA_" # years 2007-2016, 2024, from 01/25 to 08/25
    prefix_aggr = "BRASIL_AGGR_"
    
    links_brasil_raw = [(k,v) for k, v in env.items() if k.startswith(prefix_raw)]
    links_brasil_extra = [(k,v) for k, v in env.items() if k.startswith(prefix_extra)]

    link_brasil_aggr = [(k,v) for k, v in env.items() if k.startswith(prefix_aggr)]
    
    return links_brasil_raw, links_brasil_extra, link_brasil_aggr

def unzip(from_folder, to_folder):
    zip_names = [f for f in os.listdir(from_folder) if os.path.isfile(os.path.join(from_folder, f))]
    to_folder = from_folder + "/" + to_folder

    for zip in zip_names:
        with zipfile.ZipFile(from_folder + "/" + zip, "r") as zip_ref:
            zip_ref.extractall(to_folder)
        os.remove(from_folder + "/" + zip)

def download(link_list, folder):

    dir = "./dataset/" + folder
    if not os.path.exists(dir):
        os.makedirs(dir)

    for link in link_list:
        try:
            url = link[1]
            fileName = link[0].split(folder + "_")[1].lower() + ".zip"

            save_path_zip = dir + "/" + fileName

            if not os.path.exists(save_path_zip):
                gdown.download(url, save_path_zip, quiet=False, fuzzy=True)
        except Exception as e:
            print(f"[{folder}] Dataset download error. Relaunch the script.")
            return False
    return True

links_brasil_raw, links_brasil_extra, link_brasil_aggr = load_env()

no_download_errors1 = download(links_brasil_raw, folder="BRASIL_RAW")
no_download_errors2 = download(links_brasil_extra, folder="BRASIL_EXTRA")
no_download_errors3 = download(link_brasil_aggr, folder="BRASIL_AGGR")

if no_download_errors1 and no_download_errors2 and no_download_errors3:
    unzip("./dataset/BRASIL_RAW", "acidentes")
    unzip("./dataset/BRASIL_EXTRA", "acidentes")
    unzip("./dataset/BRASIL_AGGR", ".")
