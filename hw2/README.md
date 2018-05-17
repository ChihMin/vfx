## Prerequest

- OpenCV 3.4
- Compiler support C++11 & OpenMP

## How to build?

```
# Build my code
cd code/
mkdir build
cd build/
cmake .. -DCMAKE_CXX_FLAGS="-std=c++11 -fopenmp"
make
```

## Prepare

Images in a folder containing pano.txt, and the format of pano.txt is listed below:

pano.txt:
XXXXX-1.jpg focal-length-1
XXXXX-2.jpg focal-length-2
...
...

## How to run?

```
# This command will emit output.jpg, which is stitching image
# The command format: ./EXEC ${DATA_FORDER}

./DisplayImage ../../data/images
```
