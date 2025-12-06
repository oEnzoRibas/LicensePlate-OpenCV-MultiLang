cmake_minimum_required(VERSION 3.10)
project(PlateBenchmarkCpp)

# Define o padrão C++17 (Obrigatório para usar std::filesystem)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# Procura o OpenCV instalado no sistema
set(OpenCV_DIR "C:/opt/opencv_mingw")  

find_package(OpenCV REQUIRED)

# Mostra no terminal onde o OpenCV foi encontrado (para debug)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Inclui os cabeçalhos do OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Cria o executável
add_executable(benchmark_plates LicensePlateHaarCascade.cpp)

# Linka as bibliotecas do OpenCV ao executável
target_link_libraries(benchmark_plates ${OpenCV_LIBS})