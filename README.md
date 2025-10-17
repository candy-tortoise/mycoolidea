# GPU-Accelerated FHE Logistic Regression
CUDA-accelerated fully homomorphic encryption logistic regression training and testing using quadratic gradient

## Requirements
-  [CMake](https://cmake.org/)
-  [FIDESlib](https://github.com/CAPS-UMU/FIDESlib/) with its patched v.1.2.4 of [OpenFHE](https://github.com/openfheorg/openfhe-development)

## Compilation
To compile the project, follow these steps:

  - Clone this repository.
  - Generate the build files with CMake.
  ```bash
  cmake -S . -B build
  ```
  - Build the project.
  ```bash
  cmake --build build
  ```
  - Run the code.
  ```bash
  ./build/log-reg
  ```

## Notes

Tested on an Ubuntu 24.03.4 LTS machine with an NVIDIA L40S 48GB GPU and gcc/g++ version 13.3.0, nvcc version 12.9
