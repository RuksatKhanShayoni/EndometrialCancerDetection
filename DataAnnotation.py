import os                      # for file and folder operations
import pandas as pd            # for creating CSV

# dataset root directory
root_dir = "/Users/ruksatkhanshayoni/Downloads/7306361/histopathological image dataset for ET"

image_paths = []
labels = []

# iterate each main folders
for main_folder in os.listdir(root_dir):
    main_path = os.path.join(root_dir, main_folder)  # construct full path to main folder

    if os.path.isdir(main_path):  # check if it's a folder
        # iterate each item inside the main folder and check if there is a subfolder or images
        for sub in os.listdir(main_path):
            sub_path = os.path.join(main_path, sub)  # construct full path to subfolder

            if os.path.isdir(sub_path):  # if subfolder exists
                for file in os.listdir(sub_path):  # iterate images in subfolder
                    if file.lower().endswith('.jpg'):  # get image files
                        image_paths.append(os.path.join(sub_path, file))  # store image path
                        labels.append(f"{main_folder}_{sub}")  # labeling
            else:
                # if there is no subfolder
                if sub.lower().endswith('.jpg'):
                    image_paths.append(os.path.join(main_path, sub))  # store image path
                    labels.append(main_folder)  # labeling

# create dataframe from the image paths and labels
df = pd.DataFrame({'image_path': image_paths, 'label': labels})

# save the dataframe to CSV file
df.to_csv('annotated_labels.csv', index=False)  # index=False prevents writing row numbers

# print the annotated labels
print(f"Annotated {len(df)} images and saved to 'annotated_labels.csv'")
 