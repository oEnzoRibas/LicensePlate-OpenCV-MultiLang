mkdir build
cd build
cmake ..
cmake --build . --config Release
.\Release\benchmark_plates.exe

rm -r * # (Estando DENTRO da pasta build)


cmake -DCMAKE_BUILD_TYPE=Release ..