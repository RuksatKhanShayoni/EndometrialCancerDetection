import os
import shutil              # For deleting directories
import pandas as pd          # For reading and writing CSV files
import random                     # For randomly select images
from PIL import Image           # For reading and saving images
from torchvision import transforms  # For defining image augmentation technique
from tqdm import tqdm             # For showing progress bar

# Configuration
train_csv = 'train_split.csv'       # Path to original training CSV file
output_dir = 'augmented_train/'      # Folder to save augmented images
target_per_class = 5000              # Number of total images per class

# Clean previous output if it exists
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)             # Delete the output directory
os.makedirs(output_dir, exist_ok=True)    # Create the output directory again

# Define augmentation techniques
augment_transform = transforms.Compose([
    transforms.Resize((224, 224)),                  # Resize all images to 224x224 as ResNet input size
    transforms.RandomHorizontalFlip(),               # Flip image horizontally
    transforms.RandomRotation(20),                   # Random rotation 20 degrees
    transforms.ColorJitter(brightness=0.3,            # Random brightness
                           contrast=0.3,            # Random contrast
                           saturation=0.3,          # Random saturation
                           hue=0.05),                   # Random hue
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))  # Randomly translate image in height and width
])


# Load training CSV
df = pd.read_csv(train_csv)          # Read the training split CSV into a pandas DataFrame
class_groups = df.groupby('label')     # Group image paths by their class label
augmented_data = []


# Loop each class
for label, group in class_groups:
    print(f"\nProcessing class: {label}")

    images = group['image_path'].tolist()  # Get list of image paths

    # Create a separate output folder for each class
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)


    # Copy original images
    for img_path in images:
        try:
            img_name = os.path.basename(img_path)             # Get file name
            save_path = os.path.join(label_dir, img_name)     # Path output folder

            if not os.path.exists(save_path):                # Avoid re-copying if the file already exists
                image = Image.open(img_path).convert('RGB')
                image = image.resize((224, 224))
                image.save(save_path)
                augmented_data.append({'image_path': save_path, 'label': label})  # Save image path and label

        except Exception as e:
            print(f"Failed to copy {img_path}: {e}")          # Print error if any image fails to process


    # Create augmented images

    # Count how many images are already present
    current_count = len([f for f in os.listdir(label_dir) if f.lower().endswith('.jpg')])
    to_generate = target_per_class - current_count  # How many new images needed
    print(f"Generating {to_generate} new images for class {label}")

    idx = 0  # track how many images created

    while idx < to_generate:
        img_path = random.choice(images)  # Randomly select an original image to augment

        try:
            image = Image.open(img_path).convert('RGB')        # convert to RGB
            new_img = augment_transform(image)                   # Apply augmentation
            aug_name = f"aug_{idx}_{os.path.basename(img_path)}" # name augmented image
            save_path = os.path.join(label_dir, aug_name)        # Output path

            new_img.save(save_path)                              # Save the augmented image
            augmented_data.append({'image_path': save_path, 'label': label})  # Save its path and label
            idx += 1

        except Exception as e:
            print(f"Error on image {img_path}: {e}")             # If augmentation fails skip that image


# Save the CSV with all data
aug_df = pd.DataFrame(augmented_data)           # Convert list of dictionaries to DataFrame
aug_df.to_csv('augmented_train_split.csv', index=False)  # Save to CSV for training

print("\nAugmentation complete!")
