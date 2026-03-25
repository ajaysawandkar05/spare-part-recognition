import os

# Root folder containing all material folders
root_folder = r"C:\Users\ajays\Desktop\Spare_Part_Recognition\data\hwl_images"

for subdir, dirs, files in os.walk(root_folder):
    for file in files:
        file_path = os.path.join(subdir, file)

        # Check if NOT a PNG file
        if not file.lower().endswith(".png"):
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")