from sklearn.model_selection import train_test_split
import pandas as pd

# load saved annotations
df = pd.read_csv('annotated_labels.csv')

# split train, val and test
train_val_df, test_df = train_test_split(
    df,
    test_size=0.15, # 15% for test set
    stratify=df['label'], # maintain label balance in each split
    random_state=42 # ensures reproducibility of split
)

# Split into train and validation
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.15 / 0.85,  # 15% for val
    stratify=train_val_df['label'], # maintain label balance in each split
    random_state=42 # ensures reproducibility of split
)

# Save splits to CSV
train_df.to_csv('train_split.csv', index=False)
val_df.to_csv('val_split.csv', index=False)
test_df.to_csv('test_split.csv', index=False)


# Step 5: Print how many samples are in each split
print(f"Splitting Complete.....")
print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

