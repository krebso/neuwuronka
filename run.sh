#!/bin/bash
## change this file to your needs

echo "Adding some modules"

module add cmake
module add gcc

rm -rf build
mkdir build
cd build
cmake ..

echo "Compiling..."

make


echo "Running..."

./neuwuronka

cd -

## dont forget to use compiler optimizations (e.g. -O3 or -Ofast)
# g++ -Wall -std=c++17 -O3 src/main.cpp src/file2.cpp -o network


## use nice to decrease priority in order to comply with aisa rules
## https://www.fi.muni.cz/tech/unix/computation.html.en
## especially if you are using multiple cores
# nice -n 19 ./network
