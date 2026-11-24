"""
This file generates synthetic chessboard images using blender.

The image generation process is:
1. Asks the user to select a folder containing a csv file with fen positions.
2. Creates an output folder for the generated images.
3. Lets the user choose one of three camera views: overhead, east, west.
4. Reads each fen position from the csv file.
5. Runs blender to arrange the chess pieces and render the selected camera view.
6. Crops the rendered image to the chessboard area.
7. Saves the cropped image as a jpg using its frame number.
8. Removes temporary blender images after successful processing.

Images that already exist are skipped to avoid generating them again.

"""
import subprocess
import os
import csv
import glob
import time
from PIL import Image


BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
BLEND_FILE = r"C:\Users\Owner\Documents\own_final_project\chessboard_generation\chess-set.blend"
SCRIPT_FILE =  r"C:\Users\Owner\Documents\own_final_project\chessboard_generation\chess_position_api_v2.py"
RESOLUTION = 2400 


def generate_dataset():
 
    input_dir_raw = input("Enter the directory path containing the csv file: ").strip().strip('"')
    input_dir = os.path.abspath(input_dir_raw)

    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print(f"Error: No .csv file found in {input_dir}")
        return

    csv_path = csv_files[0]
    print(f"Found csv: {csv_path}")

    output_dir = os.path.join(input_dir, "generated_images")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    print("\nThe blender script generates 3 views. Which one do you want to keep?")
    print("1. for overhead - Cropped Center")
    print("2. for east - Cropped Right Side")
    print("3. for west - Cropped Left Side")

    view_map = {
        "1": ("1_overhead.png", "overhead"),
        "2": ("2_east.png", "east"),
        "3": ("3_west.png", "west"),
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
                print("Error: csv must have 'from_frame' and 'fen' columns.")
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
            
            # checks that blender generated the image
            source_path = os.path.join(output_dir, target_source_file)

            print("Expected path:", repr(source_path))
            print("File exists:", os.path.isfile(source_path))
            print("Files in directory:", os.listdir(output_dir))

            if not os.path.isfile(source_path):
                print(f"Error: Blender did not generate {source_path}")
                print("Blender STDOUT:", result.stdout[-3000:])
                print("Blender STDERR:", result.stderr[-3000:])
                continue

            # crops and save image
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

                # cleanup temporary Blender renders
                for fname in generated_files:
                    temp_path = os.path.join(output_dir, fname)
                    if os.path.isfile(temp_path):
                        os.remove(temp_path)

            except Exception as e:
                print(f"Error processing image for frame {frame_id}: {e}")

    print("\nGeneration process complete.")


if __name__ == "__main__":
    generate_dataset()
