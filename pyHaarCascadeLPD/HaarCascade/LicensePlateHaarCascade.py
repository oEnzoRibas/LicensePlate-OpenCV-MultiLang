import cv2
import os
import csv
import time
import numpy as np
import random 

print(f"🐍 Versão do OpenCV em Python: {cv2.__version__}")
# --- CONFIGURAÇÃO DE CAMINHOS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(SCRIPT_DIR, "benchmark")

# Pasta onde estão os XMLs
RESOURCE_DIR = os.path.join(BASE_PATH, "resources")

# Lista dos modelos Haar Cascade para testar
HAAR_MODELS = [
    "haarcascade_russian_plate_number.xml",
    "haarcascade_license_plate_rus_16stages.xml" 
]

# Caminhos dos Datasets
DATASET1_PATH = os.path.join(BASE_PATH, "images", "dataset", "dataset1")
DATASET2_PATH = os.path.join(BASE_PATH, "images", "dataset", "dataset2", "Diverse-LPDv2", "images")

# Caminhos de saída base
OUTPUT_CSV_BASE = os.path.join(BASE_PATH, "csvs")
OUTPUT_IMG_BASE = os.path.join(BASE_PATH, "images", "processed_plates")

def get_image_type(root_folder, filename, dataset_origin):
    """
    Define o 'Tipo' da imagem baseado na origem do Dataset.
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

# Adicionei o parametro model_tag
def process_dataset(detector, dataset_path, dataset_id, csv_writer, output_img_folder, model_tag):
    """
    Processa uma pasta inteira (e subpastas) usando o detector fornecido.
    """
    if not os.path.exists(dataset_path):
        print(f"⚠️ Aviso: Pasta do dataset não encontrada: {dataset_path}")
        return

    print(f"   📂 Varrendo Dataset: {dataset_id}...")

    for root, dirs, files in os.walk(dataset_path):
        for filename in files:
            if not filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                continue

            # 1. Definir o TIPO antes de processar
            img_type = get_image_type(root, filename, dataset_id)
            
            image_path = os.path.join(root, filename)
            frame = cv2.imread(image_path)
            
            if frame is None:
                continue

            # 2. Medição de Tempo (Detect)
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
            
            # 3. Desenhar e Salvar Dados
            first_x, first_y, first_w, first_h = 0, 0, 0, 0
            
            if count > 0:
                for i, (x, y, w, h) in enumerate(plates):
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    if i == 0:
                        first_x, first_y, first_w, first_h = x, y, w, h
            
            # Salvar imagem processada (Adicionado model_tag no nome)
            unique_filename = f"{model_tag}_{img_type}_{dataset_id}_{filename}"
            cv2.imwrite(os.path.join(output_img_folder, unique_filename), frame)

            # Escrever no CSV (Adicionado model)
            csv_writer.writerow({
                'file': filename,
                'model': model_tag,        # Nova Coluna
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
    # Loop Principal: Um ciclo para cada XML
    for xml_file in HAAR_MODELS:
        xml_path = os.path.join(RESOURCE_DIR, xml_file)
        
        print("-" * 60)
        print(f"🚀 Iniciando Benchmark para o modelo: {xml_file}")
        
        if not os.path.exists(xml_path):
            print(f"❌ Erro: XML não encontrado em {xml_path}")
            continue

        # Carrega o detector
        detector = cv2.CascadeClassifier(xml_path)
        if detector.empty():
            print(f"❌ Erro ao carregar o detector (arquivo corrompido?).")
            continue

        # --- LÓGICA DE IDENTIFICAÇÃO DO MODELO (Igual ao C++) ---
        model_tag = "UNKNOWN"
        if "16stages" in xml_file:
            model_tag = "16STAGES"
        elif "russian" in xml_file:
            model_tag = "RUS"

        model_name = xml_file.replace(".xml", "")
        
        # 1. Gera número aleatório para o ID da execução
        run_id = random.randint(10000, 99999)
        
        # 2. Define pasta específica para o CSV deste modelo
        model_csv_folder = os.path.join(OUTPUT_CSV_BASE, model_name)
        os.makedirs(model_csv_folder, exist_ok=True)
        
        # 3. Define nome do arquivo com a TAG e número aleatório
        # Ex: results_RUS_12345.csv
        csv_filename = f"results_{model_tag}_{run_id}.csv"
        csv_path = os.path.join(model_csv_folder, csv_filename)
        
        # Pasta de imagens também separada por modelo
        img_folder = os.path.join(OUTPUT_IMG_BASE, model_name)
        os.makedirs(img_folder, exist_ok=True)

        print(f"   📁 Salvando CSV em: {model_csv_folder}")
        print(f"   📄 Arquivo: {csv_filename}")

        # Adicionei 'model' no fieldnames
        with open(csv_path, mode='w', newline='') as f:
            fieldnames = ['file', 'model', 'type', 'dataset', 'detected_count', 'time_ns', 'first_x', 'first_y', 'first_w', 'first_h']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # Processa Datasets
            process_dataset(detector, DATASET1_PATH, "DS1", writer, img_folder, model_tag)
            process_dataset(detector, DATASET2_PATH, "DS2", writer, img_folder, model_tag)

        print(f"✅ Finalizado para {xml_file}.")

if __name__ == "__main__":
    main()