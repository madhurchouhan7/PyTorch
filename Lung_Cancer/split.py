import os
import splitfolders 

input_folder = "F:\Machine Learning\PyTorch\Lung_Cancer\data"

output_folder = "F:\Machine Learning\PyTorch\Lung_Cancer\Final_Split_Data"

print("Splitting data...")

splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.8, .1, .1), group_prefix=None)

print(f"Success! Your data is ready at: {output_folder}")
print("Check your folder structure. It should look like this:")
print(f"{output_folder}\\train\\Normal")
print(f"{output_folder}\\train\\Malignant")