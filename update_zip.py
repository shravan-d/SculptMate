import zipfile
import os

dirs_to_disregard = [".git", "assets", ".github", "checkpoints", "__pycache__"]

def update_zip(zip_file, folder_path):
    with zipfile.ZipFile(zip_file, 'a', compression=zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            disregard = False
            for dir in dirs_to_disregard:
                if dir in root:
                    disregard = True
                    break
            if disregard:
                continue
            for file in files:
                file_path = os.path.join(root, file)
                arcname = "SculptMate/"+os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname=arcname)


folder_path = '../SculptMate/'
zip_file = '../SculptMate.zip'
update_zip(zip_file, folder_path)
