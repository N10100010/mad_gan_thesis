
import os
import zipfile

# Define the root experiments folder
experiments_folder = "/home/stud/n/nr063/mounted_home/mad_gan_thesis/code/experiments"

# Define the output zip file name
output_zip = "./data_creation_jsons.zip"

# Collect JSON files from folders containing "_DataCreation_"
json_files = []
for root, dirs, files in os.walk(experiments_folder):
    if "_DataCreation_" in os.path.basename(root):
        json_files.extend([os.path.join(root, f) for f in files if f.endswith(".json")])

# Create a zip archive and add the collected JSON files
with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
    for json_file in json_files:
        zipf.write(json_file, os.path.relpath(json_file, experiments_folder))

print(f"Created zip archive: {output_zip} with {len(json_files)} JSON files.")

