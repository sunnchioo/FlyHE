## Introduction

FlyHE is a **high-performance CUDA-based Fully Homomorphic Encryption (FHE) library** designed to accelerate privacy-preserving computation on GPUs. The library focuses on efficient GPU implementations of both **SIMD-FHE** and **SISD-FHE**, while also providing **scheme conversion** to enable mutual conversion between different FHE schemes.

Currently, FlyHE supports the following FHE schemes:
- **SIMD-FHE**: CKKS  
- **SISD-FHE**: TFHE  

By leveraging GPU parallelism, FlyHE aims to significantly improve the computational efficiency of homomorphic operations, making it suitable for research-oriented and experimental scenarios involving high-throughput encrypted computation.  
This library is **research-oriented in nature**, primarily intended for academic research and experimental evaluation rather than production deployment. It is under active development and will be continuously updated with new features, optimizations, and supported schemes.

---

## Usage

```bash
mkdir build
cd build
cmake ..
make -j
```


