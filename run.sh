#!/bin/bash
## change this file to your needs

echo "Adding modules..."

module add cmake
module add gcc

echo "Preparing build directory..."

rm -rf build
mkdir build
cd build
cmake .. > /dev/null

echo "Compiling..."

make > /dev/null


echo "Running..."

./nn

cd -
