import subprocess
import os
import csv
import glob
import time
from PIL import Image

# --- Configuration ---
# CRITICAL: Use 'blender.exe', NOT 'blender-launcher.exe'.
# CRITICAL: Change for your own directory for those files, for some reason relative directory don't work
BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
BLEND_FILE = r"C:\Users\Owner\Documents\own_final_project\chessboard_generation\chess-set.blend"
SCRIPT_FILE =  r"C:\Users\Owner\Documents\own_final_project\chessboard_generation\chess_position_api_v2.py"
RESOLUTION = 2400  # Resolution defined here to use in crop calculations


def generate_dataset():
 
    input_dir_raw = input("Enter the directory path containing the CSV file: ").strip().strip('"')
    input_dir = os.path.abspath(input_dir_raw)

    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print(f"Error: No .csv file found in {input_dir}")
        return

    csv_path = csv_files[0]
    print(f"Found CSV: {csv_path}")

    output_dir = os.path.join(input_dir, "generated_images")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output Directory: {output_dir}")

    print("\nThe Blender script generates 3 views. Which one do you want to keep?")
    print("1. overhead (Keeps '1_overhead.png') - Cropped Center")
    print("2. east     (Keeps '2_east.png')     - Cropped Right Side")
    print("3. west     (Keeps '3_west.png')     - Cropped Left Side")

    view_map = {
        "1": ("1_overhead.png", "overhead"),
        "overhead": ("1_overhead.png", "overhead"),
        "2": ("2_east.png", "east"),
        "east": ("2_east.png", "east"),
        "3": ("3_west.png", "west"),
        "west": ("3_west.png", "west")
    }

    choice = input("Enter choice: ").strip().lower()
    target_source_file, view_type = view_map.get(choice, ("1_overhead.png", "overhead"))
    print(f"Selected view: {view_type} (Source: {target_source_file})")

    generated_files = ["1_overhead.png", "2_east.png", "3_west.png"]

    #Iterate through CSV
    with open(csv_path, 'r') as f:

        reader = csv.DictReader(f)
        print(1)
        for i, row in enumerate(reader):
            try:
                frame_id = int(row['from_frame'])
                fen = row['fen']
            except KeyError:
                print("Error: CSV must have 'from_frame' and 'fen' columns.")
                return

            final_filename = f"frame_{frame_id:06d}.jpg"
            final_path = os.path.join(output_dir, final_filename)

            if os.path.exists(final_path):
                print(f"Skipping frame {frame_id}, file exists in file.")
                continue

            print(f"Generating frame {frame_id}...")

            #  Blender Command
            cmd = [
                BLENDER_EXE,
                BLEND_FILE,
                "--background",
                "--python", SCRIPT_FILE,
                "--",
                "--fen", fen,
                "--resolution", str(RESOLUTION),
                "--output_path", output_dir,
                "--view", "white",
                "--view_angle", view_type
            ]

            if i == 0:
                print(f"\n[DEBUG] Running command:\n{' '.join(cmd)}\n")

           
            try:

                result = subprocess.run(cmd,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)

            except subprocess.CalledProcessError as e:
                    print(f"Blender failed for frame {frame_id}.")
                    print(e.stderr[-3000:])
                    continue
            
            # check that Blender generated the image
            source_path = os.path.join(output_dir, target_source_file)

            print("Expected path:", repr(source_path))
            print("File exists:", os.path.isfile(source_path))
            print("Files in directory:", os.listdir(output_dir))

            if not os.path.isfile(source_path):
                print(f"Error: Blender did not generate {source_path}")
                print("Blender STDOUT:", result.stdout[-3000:])
                print("Blender STDERR:", result.stderr[-3000:])
                continue

            # crop and save image
            try:
                with Image.open(source_path) as img:
                    width, height = img.size

                    if view_type == "overhead":
                        left = width * 0.4
                        top = height * 0.4
                        right = width * 0.6
                        bottom = height * 0.6

                    elif view_type == "west":
                        top = height * 0.4
                        bottom = height * 0.6
                        left = width / 16
                        right = width / 16 + 2 * (width / 10)

                    elif view_type == "east":
                        top = height * 0.4
                        bottom = height * 0.6
                        right = width - width / 16
                        left = right - 2 * (width / 10)

                    crop_box = (int(left + 34), int(top + 34), int(right - 34), int(bottom - 34))

                    cropped_img = img.crop(crop_box)
                    cropped_img.convert("RGB").save(final_path, "JPEG")

                print(f"Saved successfully: {final_path}")

                # Cleanup temporary Blender renders
                for fname in generated_files:
                    temp_path = os.path.join(output_dir, fname)
                    if os.path.isfile(temp_path):
                        os.remove(temp_path)

            except Exception as e:
                print(f"Error processing image for frame {frame_id}: {e}")

    print("\nGeneration process complete.")


if __name__ == "__main__":
    generate_dataset()
