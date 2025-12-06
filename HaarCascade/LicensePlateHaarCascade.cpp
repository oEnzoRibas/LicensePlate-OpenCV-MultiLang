#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <random>

// Namespaces para facilitar a leitura
using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// --- CONFIGURAÇÃO DE CAMINHOS ---
const string BASE_PATH = "../benchmark"; 

const string RESOURCE_DIR = BASE_PATH + "/resources";
const string DATASET1_PATH = BASE_PATH + "/images/dataset/dataset1";
const string DATASET2_PATH = BASE_PATH + "/images/dataset/dataset2/Diverse-LPDv2/images";
const string OUTPUT_CSV_BASE = BASE_PATH + "/csvs";
const string OUTPUT_IMG_BASE = BASE_PATH + "/images/processed_plates";

// Lista dos modelos Haar Cascade
const vector<string> HAAR_MODELS = {
    "haarcascade_russian_plate_number.xml",
    "haarcascade_license_plate_rus_16stages.xml"
};

// --- FUNÇÃO AUXILIAR: Get Image Type ---
string getImageType(const string& root_folder, const string& filename, const string& dataset_origin) {
    if (dataset_origin == "DS1") {
        try {
            fs::path rootPath(root_folder);
            fs::path ds1Path(DATASET1_PATH);
            
            // Cria caminho relativo
            fs::path relPath = fs::relative(rootPath, ds1Path);
            
            // Pega o primeiro componente do caminho (ex: Brazil)
            string typeLabel = relPath.begin()->string();
            
            if (typeLabel == "." || typeLabel.empty()) return "Unknown_DS1";
            return typeLabel;
        } catch (...) {
            return "Unknown_DS1";
        }
    } else if (dataset_origin == "DS2") {
        return "Diverse-LPDv2";
    }
    return "Unknown";
}

// --- FUNÇÃO AUXILIAR: Process Dataset ---
void processDataset(CascadeClassifier& detector, const string& dataset_path, const string& dataset_id, ofstream& csv_writer, const string& output_img_folder) {
    
    if (!fs::exists(dataset_path)) {
        cout << "⚠️ Aviso: Pasta do dataset nao encontrada: " << dataset_path << endl;
        return;
    }

    cout << "   📂 Varrendo Dataset: " << dataset_id << "..." << endl;

    // Itera recursivamente (equivalente a os.walk)
    for (const auto& entry : fs::recursive_directory_iterator(dataset_path)) {
        if (!entry.is_regular_file()) continue;

        string filename = entry.path().filename().string();
        string ext = entry.path().extension().string();
        
        // Converte extensão para lowercase para verificação
        for (auto& c : ext) c = tolower(c);
        if (ext != ".jpg" && ext != ".png" && ext != ".jpeg" && ext != ".bmp") continue;

        // 1. Definir o TIPO
        string img_type = getImageType(entry.path().parent_path().string(), filename, dataset_id);

        Mat frame = imread(entry.path().string());
        if (frame.empty()) continue;

        // 2. Medição de Tempo
        auto start_time = chrono::high_resolution_clock::now();

        std::vector<Rect> plates;
        detector.detectMultiScale(
            frame, 
            plates, 
            1.1, // scaleFactor
            3,   // minNeighbors
            0,   // flags
            Size(30, 30) // minSize
        );

        auto end_time = chrono::high_resolution_clock::now();
        long long duration_ns = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

        int count = plates.size();

        // 3. Desenhar e Salvar Dados
        int first_x = 0, first_y = 0, first_w = 0, first_h = 0;

        if (count > 0) {
            for (size_t i = 0; i < plates.size(); i++) {
                rectangle(frame, plates[i], Scalar(255, 0, 0), 2);
                if (i == 0) {
                    first_x = plates[i].x;
                    first_y = plates[i].y;
                    first_w = plates[i].width;
                    first_h = plates[i].height;
                }
            }
        }

        // Salvar imagem processada
        string unique_filename = img_type + "_" + dataset_id + "_" + filename;
        string output_path = output_img_folder + "/" + unique_filename;
        imwrite(output_path, frame);

        // Escrever no CSV
        csv_writer << filename << ","
                   << img_type << ","
                   << dataset_id << ","
                   << count << ","
                   << duration_ns << ","
                   << first_x << ","
                   << first_y << ","
                   << first_w << ","
                   << first_h << endl;
    }
}

int main() {
    // Configura gerador de números aleatórios
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distr(10000, 99999);

    // Garante que diretórios base existem
    fs::create_directories(OUTPUT_CSV_BASE);
    fs::create_directories(OUTPUT_IMG_BASE);

    for (const auto& xml_file : HAAR_MODELS) {
        string xml_path = RESOURCE_DIR + "/" + xml_file;
        cout << xml_path << endl;
        cout << string(60, '-') << endl;
        cout << "🚀 Iniciando Benchmark para o modelo: " << xml_file << endl;

        if (!fs::exists(xml_path)) {
            cout << "❌ Erro: XML nao encontrado em " << xml_path << endl;
            continue;
        }

        CascadeClassifier detector;
        if (!detector.load(xml_path)) {
            cout << "❌ Erro ao carregar o detector (arquivo corrompido?)." << endl;
            continue;
        }

        // Remove extensão .xml para usar no nome da pasta
        string model_name = xml_file;
        size_t lastindex = model_name.find_last_of("."); 
        if (lastindex != string::npos) model_name = model_name.substr(0, lastindex);

        // 1. Gera ID aleatório
        int run_id = distr(gen);

        // 2. Define pastas
        string model_csv_folder = OUTPUT_CSV_BASE + "/" + model_name;
        fs::create_directories(model_csv_folder);

        string img_folder = OUTPUT_IMG_BASE + "/" + model_name;
        fs::create_directories(img_folder);

        string csv_filename = "results_" + to_string(run_id) + ".csv";
        string csv_path = model_csv_folder + "/" + csv_filename;

        cout << "   📁 Salvando CSV em: " << model_csv_folder << endl;
        cout << "   📄 Arquivo: " << csv_filename << endl;

        ofstream csvFile(csv_path);
        if (!csvFile.is_open()) {
            cout << "❌ Erro ao criar arquivo CSV." << endl;
            continue;
        }

        // Cabeçalho CSV
        csvFile << "file,type,dataset,detected_count,time_ns,first_x,first_y,first_w,first_h" << endl;

        // Processa Datasets
        processDataset(detector, DATASET1_PATH, "DS1", csvFile, img_folder);
        processDataset(detector, DATASET2_PATH, "DS2", csvFile, img_folder);

        csvFile.close();
        cout << "✅ Finalizado para " << xml_file << "." << endl;
    }

    return 0;
}