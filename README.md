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

## References

- Cheon, J. H., Kim, A., Kim, M., & Song, Y. (2017).
  *Homomorphic Encryption for Arithmetic of Approximate Numbers*.

- Chillotti, I., Gama, N., Georgieva, M., & Izabachène, M. (2016).
  *TFHE: Fast Fully Homomorphic Encryption over the Torus*.

- Lu, W. J., Huang, Z., Hong, C., Ma, Y., & Qu, H. (2021, May). *PEGASUS: bridging polynomial and non-polynomial evaluations in homomorphic encryption*.


## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/sunnchioo/FlyHE/blob/main/LISENCE) file for details.

Some files contain the modified code from the [Phantom](https://github.com/encryptorion-lab/phantom-fhe) version licensed under MIT License.

Some files contain the modified code from the [Opera](https://github.com/hku-systems/Opera) version licensed under [MIT License](https://github.com/hku-systems/Opera/blob/main/LICENSE).

Some files contain the modified code from the [TFHEpp](https://github.com/virtualsecureplatform/TFHEpp), which is licensed under the [Apache License 2.0](https://github.com/virtualsecureplatform/TFHEpp/blob/master/LICENSE).
