# itesol

:warning: This is work in progress software currently in an early design phase.

A C++20 header-only library to implement iterative eigensolvers ([power method](https://en.wikipedia.org/wiki/Power_iteration), [Lanczos algorithm](https://en.wikipedia.org/wiki/Lanczos_algorithm), …).
By heavily relying on template metaprogramming we try to interface with various different linear algebra backends (Eigen, Blaze, BLAS/LAPACK, CUDA) and build algorithms from simple elementary building blocks that can be easily implemented for a new custom backend. 
A Python module is also worked on.

# Roadmap

## Algorithms

- [x] Power method
- [ ] Lanczos
- [ ] Implicitly restarted Lanczos
- [ ] Arnoldi iterations
- [ ] ARPACK wrapper

## Backends

- [x] Eigen3
- [ ] Blaze
- [ ] CUDA
- [x] BLAS/LAPACK (experimental)
