#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <random>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;


const string BASE_PATH = "../benchmark"; 

const string RESOURCE_DIR = BASE_PATH + "/resources";
const string DATASET1_PATH = BASE_PATH + "/images/dataset/dataset1";
const string DATASET2_PATH = BASE_PATH + "/images/dataset/dataset2/Diverse-LPDv2/images";
const string OUTPUT_CSV_BASE = BASE_PATH + "/csvs";
const string OUTPUT_IMG_BASE = BASE_PATH + "/images/processed_plates";

const vector<string> HAAR_MODELS = {
    "haarcascade_russian_plate_number.xml",
    "haarcascade_license_plate_rus_16stages.xml"
};

/*
@brief Determines the image type based on its path and dataset origin.
@param root_folder: The root folder of the image.
@param filename: The name of the image file.
@param dataset_origin: Identifier for the dataset origin.

@return A string representing the image type.
*/
string getImageType(const string& root_folder, const string& filename, const string& dataset_origin) {
    if (dataset_origin == "DS1") {
        try {
            fs::path rootPath(root_folder);
            fs::path ds1Path(DATASET1_PATH);
            
            fs::path relPath = fs::relative(rootPath, ds1Path);
            
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

/*
@brief Processes a dataset by detecting license plates and logging results.
@param detector: The CascadeClassifier for license plate detection.
@param dataset_path: The path to the dataset.
@param dataset_id: Identifier for the dataset.
@param csv_writer: The output CSV file stream.
@param output_img_folder: The folder to save processed images.
@param model_tag: Tag representing the model used.

@return void
*/
void processDataset(CascadeClassifier& detector, const string& dataset_path, const string& dataset_id, ofstream& csv_writer, const string& output_img_folder, const string& model_tag) {
    
    if (!fs::exists(dataset_path)) {
        cout << "⚠️ WARNING: Dataset folder not found: " << dataset_path << endl;
        return;
    }

    cout << "Scanning Dataset: " << dataset_id << "..." << endl;
    for (const auto& entry : fs::recursive_directory_iterator(dataset_path)) {
        if (!entry.is_regular_file()) continue;

        string filename = entry.path().filename().string();
        string ext = entry.path().extension().string();
        
        for (auto& c : ext) c = tolower(c);
        if (ext != ".jpg" && ext != ".png" && ext != ".jpeg" && ext != ".bmp") continue;

        string img_type = getImageType(entry.path().parent_path().string(), filename, dataset_id);

        Mat frame = imread(entry.path().string());
        if (frame.empty()) continue;

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

        string unique_filename = model_tag + "_" + img_type + "_" + dataset_id + "_" + filename;
        string output_path = output_img_folder + "/" + unique_filename;
        imwrite(output_path, frame);

        csv_writer << filename << ","
                   << model_tag << ","
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
    
    std::cout << "OpenCV version in C++: " << CV_VERSION << std::endl;
    
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distr(10000, 99999);

    fs::create_directories(OUTPUT_CSV_BASE);
    fs::create_directories(OUTPUT_IMG_BASE);

    for (const auto& xml_file : HAAR_MODELS) {
        string xml_path = RESOURCE_DIR + "/" + xml_file;
        cout << xml_path << endl;
        cout << string(60, '-') << endl;
        cout << "🚀 Starting Benchmark for model: " << xml_file << endl;

        if (!fs::exists(xml_path)) {
            cout << "Error: XML not found at " << xml_path << endl;
            continue;
        }

        CascadeClassifier detector;
        if (!detector.load(xml_path)) {
            cout << "Error loading detector (corrupted file?)." << endl;
            continue;
        }

        string model_tag = "UNKNOWN";
        if (xml_file.find("16stages") != string::npos) {
            model_tag = "16STAGES";
        } else if (xml_file.find("russian") != string::npos) {
            model_tag = "RUS";
        }

        string model_name_folder = xml_file;
        size_t lastindex = model_name_folder.find_last_of("."); 
        if (lastindex != string::npos) model_name_folder = model_name_folder.substr(0, lastindex);

        int run_id = distr(gen);

        string model_csv_folder = OUTPUT_CSV_BASE + "/" + model_name_folder;
        fs::create_directories(model_csv_folder);

        string img_folder = OUTPUT_IMG_BASE + "/" + model_name_folder;
        fs::create_directories(img_folder);

        string csv_filename = "results_" + model_tag + "_" + to_string(run_id) + ".csv";
        string csv_path = model_csv_folder + "/" + csv_filename;

        cout << " Saving CSV in: " << model_csv_folder << endl;
        cout << " File: " << csv_filename << endl;

        ofstream csvFile(csv_path);
        if (!csvFile.is_open()) {
            cout << "Error creating CSV file." << endl;
            continue;
        }

        csvFile << "file,model,type,dataset,detected_count,time_ns,first_x,first_y,first_w,first_h" << endl;

        processDataset(detector, DATASET1_PATH, "DS1", csvFile, img_folder, model_tag);
        processDataset(detector, DATASET2_PATH, "DS2", csvFile, img_folder, model_tag);

        csvFile.close();
        cout << "Finished for " << xml_file << "." << endl;
    }

    return 0;
}