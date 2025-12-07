import cv2
import os
import csv
import time
import numpy as np
import random 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(SCRIPT_DIR, "benchmark")

RESOURCE_DIR = os.path.join(BASE_PATH, "resources")

HAAR_MODELS = [
    "haarcascade_russian_plate_number.xml",
    "haarcascade_license_plate_rus_16stages.xml" 
]

DATASET1_PATH = os.path.join(BASE_PATH, "images", "dataset", "dataset1")
DATASET2_PATH = os.path.join(BASE_PATH, "images", "dataset", "dataset2", "Diverse-LPDv2", "images")

OUTPUT_CSV_BASE = os.path.join(BASE_PATH, "csvs")
OUTPUT_IMG_BASE = os.path.join(BASE_PATH, "images", "processed_plates")

def get_image_type(root_folder, filename, dataset_origin):
    """
    @summary
    Defines the 'Type' of the image based on the Dataset origin.

    Args:
        root_folder (str): The root folder of the image.
        filename (str): The name of the image file.
        dataset_origin (str): The origin of the dataset ("DS1" or "DS2").
    Returns:
        str: The type label of the image.
    """
    if dataset_origin == "DS1":
        try:
            rel_path = os.path.relpath(root_folder, DATASET1_PATH)
            type_label = rel_path.split(os.sep)[0]
            if type_label == ".": return "Unknown_DS1"
            return type_label
        except:
            return "Unknown_DS1"
            
    elif dataset_origin == "DS2":
        return "Diverse-LPDv2"
    
    return "Unknown"

def process_dataset(detector, dataset_path, dataset_id, csv_writer, output_img_folder, model_tag):
    """
    @summary
    Processes a dataset folder, detects license plates, and logs results to CSV.

    Args:
        detector (cv2.CascadeClassifier): The Haar Cascade detector.
        dataset_path (str): Path to the dataset folder.
        dataset_id (str): Identifier for the dataset (e.g., "DS1", "DS2").
        csv_writer (csv.DictWriter): CSV writer object to log results.
        output_img_folder (str): Folder to save processed images.
        model_tag (str): Tag identifying the model used.

    Returns:
        None
    """
    if not os.path.exists(dataset_path):
        print(f"⚠️ WARNING: Dataset folder not found: {dataset_path}")
        return

    print(f"Scanning Dataset: {dataset_id}...")

    for root, dirs, files in os.walk(dataset_path):
        for filename in files:
            if not filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                continue

            img_type = get_image_type(root, filename, dataset_id)
            
            image_path = os.path.join(root, filename)
            frame = cv2.imread(image_path)
            
            if frame is None:
                continue

            start_time = time.perf_counter_ns()
            
            plates = detector.detectMultiScale(
                frame,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(30, 30),
                flags=0
            )
            
            end_time = time.perf_counter_ns()
            duration_ns = end_time - start_time
            
            count = len(plates)
            
            first_x, first_y, first_w, first_h = 0, 0, 0, 0
            
            if count > 0:
                for i, (x, y, w, h) in enumerate(plates):
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    if i == 0:
                        first_x, first_y, first_w, first_h = x, y, w, h
           
            unique_filename = f"{model_tag}_{img_type}_{dataset_id}_{filename}"
            cv2.imwrite(os.path.join(output_img_folder, unique_filename), frame)

            csv_writer.writerow({
                'file': filename,
                'model': model_tag,      
                'type': img_type,
                'dataset': dataset_id,
                'detected_count': count,
                'time_ns': duration_ns,
                'first_x': first_x,
                'first_y': first_y,
                'first_w': first_w,
                'first_h': first_h
            })

def main():
    
    print(f"🐍 OpenCV version in Python: {cv2.__version__}")

    for xml_file in HAAR_MODELS:
        xml_path = os.path.join(RESOURCE_DIR, xml_file)
        
        print("-" * 60)
        print(f"🚀 Starting Benchmark for Model: {xml_file}")
        
        if not os.path.exists(xml_path):
            print(f"❌ Error: XML not found at {xml_path}")
            continue

        detector = cv2.CascadeClassifier(xml_path)
        if detector.empty():
            print(f"❌ Error loading detector (corrupted file?).")
            continue

        model_tag = "UNKNOWN"
        if "16stages" in xml_file:
            model_tag = "16STAGES"
        elif "russian" in xml_file:
            model_tag = "RUS"

        model_name = xml_file.replace(".xml", "")
        
        run_id = random.randint(10000, 99999)
        
        model_csv_folder = os.path.join(OUTPUT_CSV_BASE, model_name)
        os.makedirs(model_csv_folder, exist_ok=True)
        
        csv_filename = f"results_{model_tag}_{run_id}.csv"
        csv_path = os.path.join(model_csv_folder, csv_filename)
        
        img_folder = os.path.join(OUTPUT_IMG_BASE, model_name)
        os.makedirs(img_folder, exist_ok=True)

        print(f"Saving CSV in: {model_csv_folder}")
        print(f"File: {csv_filename}")

        with open(csv_path, mode='w', newline='') as f:
            fieldnames = ['file', 'model', 'type', 'dataset', 'detected_count', 'time_ns', 'first_x', 'first_y', 'first_w', 'first_h']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            process_dataset(detector, DATASET1_PATH, "DS1", writer, img_folder, model_tag)
            process_dataset(detector, DATASET2_PATH, "DS2", writer, img_folder, model_tag)

        print(f"Finished for {xml_file}.")

if __name__ == "__main__":
    main()