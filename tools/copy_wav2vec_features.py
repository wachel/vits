import os
import shutil

def copy(from_path, to_path, endswith):
    for root, dirs, files in os.walk(from_path):
        for file in files:
            if file.endswith(endswith):
                file_path = os.path.join(root, file)
                target_path = file_path.replace(from_path, to_path)
                target_dir = os.path.split(target_path)[0]
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                shutil.copyfile(file_path, target_path)

copy('data', 'data2', '.w2v_hm.pt')