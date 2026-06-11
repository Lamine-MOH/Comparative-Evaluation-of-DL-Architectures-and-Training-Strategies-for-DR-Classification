import os
import shutil
import zipfile
import kagglehub
import gdown
import pandas as pd

def dataset_download(dataset_name, data_path="./Data/"):
    if not dataset_name in ["Aptos", "IDRiD", "DDR", "Messidor-2"]:
        print("Invalid Dataset name")
        return
    
    print(f"Downloading {dataset_name} Dataset")
    
    # Aptos Dataset
    if dataset_name == "Aptos":        
        path = kagglehub.dataset_download("mariaherrerot/aptos2019")
        
    # IDRiD Dataset
    elif dataset_name == "IDRiD":        
        file_id = "https://drive.google.com/file/d/1QY2jrzOLf787qH1PrRnohTycAor-UibD/view?usp=sharing"
        zip_path = os.path.join(data_path, "IDRiD_Grading.zip")
        
        # Download
        gdown.download(file_id, output=zip_path,quiet=False)
        
        # Unzip
        print(f"Extracting file {zip_path} ...")
        
        path = os.path.join(data_path, "IDRiD_RAW/")
        with zipfile.ZipFile(zip_path, 'r') as ref:
            ref.extractall(path)
        
        path = os.path.join(path, "B.Disease Grading/")
                
    # DDR Dataset
    if dataset_name == "DDR":
        path = kagglehub.dataset_download("mariaherrerot/ddrdataset")
        
    # Messidor-2 Dataset
    if dataset_name == "Messidor-2":
        path = kagglehub.dataset_download("mariaherrerot/messidor2preprocess")
        
    return path

def dataset_prepare(dataset_name, download_path, save_path="./Data/", data_path="./Data/"):
    dataset_path = os.path.join(save_path, dataset_name)
    images_path = os.path.join(dataset_path, "Images/") 
    
    # Ensure the target image directory exists before copying
    os.makedirs(images_path, exist_ok=True)
    
    print("Preparing Dataset Files...")
    
    # Aptos dataset
    if dataset_name == "Aptos":
        # Merge Images
        shutil.copytree(os.path.join(download_path, "train_images/train_images/"), images_path, dirs_exist_ok=True)
        shutil.copytree(os.path.join(download_path, "test_images/test_images/"), images_path, dirs_exist_ok=True)
        shutil.copytree(os.path.join(download_path, "val_images/val_images/"), images_path, dirs_exist_ok=True)

        # Merge Labels
        df_train = pd.read_csv(os.path.join(download_path, 'train_1.csv'))
        df_test = pd.read_csv(os.path.join(download_path, 'test.csv'))
        df_val = pd.read_csv(os.path.join(download_path, 'valid.csv'))

        # Concatenate all labels
        df = pd.concat([df_train, df_test, df_val], ignore_index=True) 

        # add img extension
        df["file_name"] = df["id_code"].astype(str) + ".png" 

        # Save labels
        df.to_csv(os.path.join(dataset_path, "labels.csv"), index=False)
        
    # IDRiD dataset
    elif dataset_name == "IDRiD":
        # Merge Images
        shutil.copytree(os.path.join(download_path, "1. Original Images/a. Training Set"), images_path, dirs_exist_ok=True)
        shutil.copytree(os.path.join(download_path, "1. Original Images/b. Testing Set"), images_path, dirs_exist_ok=True)
        
        # Merge Labels
        df_train = pd.read_csv(os.path.join(download_path, "2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"))
        df_test = pd.read_csv(os.path.join(download_path, "2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"))
        
        # Concat training and testing
        df = pd.concat([df_train, df_test], ignore_index=True)
        df = df.rename(columns={'Image name': 'id_code'})
        df = df.rename(columns={'Retinopathy grade': 'diagnosis'})
        
        # add img extension
        df["file_name"] = df["id_code"].astype(str) + ".jpg" 
        
        # Save to csv
        df.to_csv(os.path.join(dataset_path, "labels.csv"), index=False)

    # DDR dataset
    elif dataset_name == "DDR":
        # Copy images
        shutil.copytree(os.path.join(download_path, "DR_grading/DR_grading/"), images_path, dirs_exist_ok=True)

        # Merge Labels
        df = pd.read_csv(os.path.join(download_path, 'DR_grading.csv'))

        # add img extension
        df["file_name"] = df["id_code"].astype(str) + "" 

        # save to csv
        df.to_csv(os.path.join(dataset_path, "labels.csv"), index=False)
    
    # Messidor-2 dataset
    elif dataset_name == "Messidor-2":
        # Merge Images
        shutil.copytree(os.path.join(download_path, "messidor-2/messidor-2/preprocess/"), images_path, dirs_exist_ok=True)

        # Merge Labels
        df = pd.read_csv(os.path.join(download_path, 'messidor_data.csv'))

        # add img extension
        df["file_name"] = df["id_code"].astype(str) + "" 

        # save to csv
        df.to_csv(os.path.join(dataset_path, "labels.csv"), index=False)
        
    return dataset_path