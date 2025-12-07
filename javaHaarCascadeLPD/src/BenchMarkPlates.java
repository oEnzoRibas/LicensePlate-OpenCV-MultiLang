import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Locale;
import java.util.Random;
import java.util.stream.Stream;

public class BenchMarkPlates {

    // --- CONFIGURAÇÃO DE CAMINHOS ---
    // Ajuste este BASE_PATH para apontar para a sua pasta benchmark real
    static String BASE_PATH = "src/benchmark";

    static String RESOURCE_DIR = BASE_PATH + "/resources";
    static String DATASET1_PATH = BASE_PATH + "/images/dataset/dataset1";
    static String DATASET2_PATH = BASE_PATH + "/images/dataset/dataset2/Diverse-LPDv2/images";
    static String OUTPUT_CSV_BASE = BASE_PATH + "/csvs";
    static String OUTPUT_IMG_BASE = BASE_PATH + "/images/processed_plates";

    static String[] HAAR_MODELS = {
            "haarcascade_russian_plate_number.xml",
            "haarcascade_license_plate_rus_16stages.xml"
    };

    public static void main(String[] args) throws IOException {
        // Carrega a DLL nativa do OpenCV. 
        // Se der erro aqui, adicione -Djava.library.path=... na configuração de Run do IntelliJ
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        System.out.println("⚙️ Versão do OpenCV Java: " + Core.VERSION);

        Random random = new Random();

        // Garante diretórios base
        new File(OUTPUT_CSV_BASE).mkdirs();
        new File(OUTPUT_IMG_BASE).mkdirs();

        for (String xmlFile : HAAR_MODELS) {
            String xmlPath = RESOURCE_DIR + "/" + xmlFile;

            System.out.println("-".repeat(60));
            System.out.println("🚀 Iniciando Benchmark para o modelo: " + xmlFile);

            if (!new File(xmlPath).exists()) {
                System.out.println("❌ Erro: XML não encontrado em " + xmlPath);
                continue;
            }

            CascadeClassifier detector = new CascadeClassifier(xmlPath);
            if (detector.empty()) {
                System.out.println("❌ Erro ao carregar o detector.");
                continue;
            }

            // Identificação do Modelo (Tag)
            String modelTag = "UNKNOWN";
            if (xmlFile.contains("16stages")) modelTag = "16STAGES";
            else if (xmlFile.contains("russian")) modelTag = "RUS";

            // Nome da pasta (remove extensão)
            String modelNameFolder = xmlFile.replace(".xml", "");

            // 1. Gera ID e Pastas
            int runId = 10000 + random.nextInt(90000);

            String modelCsvFolder = OUTPUT_CSV_BASE + "/" + modelNameFolder;
            new File(modelCsvFolder).mkdirs();

            String imgFolder = OUTPUT_IMG_BASE + "/" + modelNameFolder;
            new File(imgFolder).mkdirs();

            String csvFilename = "results_" + modelTag + "_" + runId + ".csv";
            String csvPath = modelCsvFolder + "/" + csvFilename;

            System.out.println("   📁 Salvando CSV em: " + modelCsvFolder);
            System.out.println("   📄 Arquivo: " + csvFilename);

            try (PrintWriter writer = new PrintWriter(new FileWriter(csvPath))) {
                // Cabeçalho CSV
                writer.println("file,model,type,dataset,detected_count,time_ns,first_x,first_y,first_w,first_h");

                processDataset(detector, DATASET1_PATH, "DS1", writer, imgFolder, modelTag);
                processDataset(detector, DATASET2_PATH, "DS2", writer, imgFolder, modelTag);
            }

            System.out.println("✅ Finalizado para " + xmlFile);
        }
    }

    private static void processDataset(CascadeClassifier detector, String datasetPathStr, String datasetId, PrintWriter writer, String outputImgFolder, String modelTag) {
        File datasetDir = new File(datasetPathStr);
        if (!datasetDir.exists()) {
            System.out.println("⚠️ Aviso: Dataset não encontrado: " + datasetPathStr);
            return;
        }

        System.out.println("   📂 Varrendo Dataset: " + datasetId + "...");

        try (Stream<Path> paths = Files.walk(Paths.get(datasetPathStr))) {
            paths.filter(Files::isRegularFile).forEach(path -> {
                String filename = path.getFileName().toString();
                String lowerName = filename.toLowerCase();

                if (!lowerName.endsWith(".jpg") && !lowerName.endsWith(".png") && !lowerName.endsWith(".jpeg") && !lowerName.endsWith(".bmp")) {
                    return;
                }

                // 1. Definir o TIPO
                String imgType = getImageType(path, datasetId);

                // Carrega Imagem
                Mat frame = Imgcodecs.imread(path.toString());
                if (frame.empty()) return; // Falha ao ler

                try {
                    // 2. Medição de Tempo
                    long startTime = System.nanoTime();

                    MatOfRect plates = new MatOfRect();
                    detector.detectMultiScale(
                            frame,
                            plates,
                            1.1, // scaleFactor
                            3,   // minNeighbors
                            0,   // flags
                            new Size(30, 30), // minSize
                            new Size() // maxSize
                    );

                    long endTime = System.nanoTime();
                    long durationNs = endTime - startTime;

                    Rect[] platesArray = plates.toArray();
                    int count = platesArray.length;

                    // 3. Dados para CSV
                    int firstX = 0, firstY = 0, firstW = 0, firstH = 0;

                    if (count > 0) {
                        for (int i = 0; i < count; i++) {
                            Rect r = platesArray[i];
                            Imgproc.rectangle(frame, r, new Scalar(255, 0, 0), 2);
                            if (i == 0) {
                                firstX = r.x; firstY = r.y; firstW = r.width; firstH = r.height;
                            }
                        }
                    }

                    // Salvar Imagem
                    String uniqueFilename = modelTag + "_" + imgType + "_" + datasetId + "_" + filename;
                    Imgcodecs.imwrite(outputImgFolder + "/" + uniqueFilename, frame);

                    // Escrever CSV (Locale.US para usar ponto decimal se precisar, embora aqui sejam inteiros/longs)
                    writer.printf(Locale.US, "%s,%s,%s,%s,%d,%d,%d,%d,%d,%d%n",
                            filename, modelTag, imgType, datasetId, count, durationNs,
                            firstX, firstY, firstW, firstH);

                } finally {
                    // CRUCIAL: Libera memória nativa imediatamente para não estourar a RAM
                    frame.release();
                }
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static String getImageType(Path filePath, String datasetId) {
        if ("DS1".equals(datasetId)) {
            try {
                // Lógica relativa ao DATASET1_PATH
                Path dsRoot = Paths.get(DATASET1_PATH);
                Path relative = dsRoot.relativize(filePath);
                // Pega o primeiro nome da pasta (ex: Brazil)
                if (relative.getNameCount() > 0) {
                    return relative.getName(0).toString();
                }
            } catch (Exception e) {
                return "Unknown_DS1";
            }
            return "Unknown_DS1";
        } else if ("DS2".equals(datasetId)) {
            return "Diverse-LPDv2";
        }
        return "Unknown";
    }
}