import os

directory = "./"

for filename in os.listdir(directory):
    if filename.endswith(".json"):
        parts = filename.split("_")
        if len(parts) >= 5:
            new_filename = "_".join(parts[:4] + parts[5:])
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
