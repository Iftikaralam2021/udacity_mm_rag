import os
import pandas as pd

#

def combine_csv_files(directory,output_filename):


    # Initialize an empty DataFrame to store the combined data
    combined_data = pd.DataFrame()

    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            # Read the CSV file into a DataFrame
            data = pd.read_csv(file_path)
            # Append the data to the combined_data DataFrame
            combined_data = pd.concat([combined_data,data], axis=0, ignore_index=True)

    # Write the combined data to a new CSV file
    combined_data = combined_data.assign(row_number=range(len(combined_data)))
    # combined_data["row_number"]-=1
    combined_data["Price"] = combined_data["Price"].str.replace("$","").str.replace(",","").astype(float)
    combined_data["House Size(sqft)"]=combined_data["House Size"].str.replace("sqft","").str.replace(",","").astype(float)
    combined_data["images"]="./images/image_"+combined_data["row_number"].astype(str)+".png"
    print(combined_data.head())
    combined