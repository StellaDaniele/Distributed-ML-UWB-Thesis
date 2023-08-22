import os
import shutil

folder_path = '../Simulation/Logs/'

json_file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.json')]

for json_file_name in json_file_names:
    common_prefix = json_file_name.rsplit('_', 1)[0]  # Common prefix until the last underscore
    folder_name = os.path.join(folder_path, common_prefix)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    source_file_path = os.path.join(folder_path, json_file_name)
    destination_file_path = os.path.join(folder_name, json_file_name)
    shutil.move(source_file_path, destination_file_path)
