# RecursiveFactorization

[![Github Action CI](https://github.com/YingboMa/RecursiveFactorization.jl/workflows/CI/badge.svg)](https://github.com/YingboMa/RecursiveFactorization.jl/actions)
[![codecov](https://codecov.io/gh/YingboMa/RecursiveFactorization.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/YingboMa/RecursiveFactorization.jl)

---

`RecursiveFactorization.jl` is a package that collects various recursive matrix
factorization algorithms.

random change

#### Implemented Algorithms:

- Sivan Toledo's recursive left-looking LU algorithm. DOI:
  [10.1137/S0895479896297744](https://epubs.siam.org/doi/10.1137/S0895479896297744)

#### Usage:

`RecursiveFactorization` does not export any functions.

```julia
julia> using RecursiveFactorization

julia> A = rand(5, 5);

julia> RecursiveFactorization.lu(A); # out-of-place

julia> RecursiveFactorization.lu!(copy(A)); # in-place

julia> RecursiveFactorization.lu!(copy(A), Vector{Int}(undef, size(A, 2))); # in-place w/ pivoting vector
```

#### Performance:

For small to medium sized matrices, it is beneficial to use
`RecursiveFactorization` over `OpenBLAS`. The benchmark script is available in
`perf/lu.jl`

![lubench](https://user-images.githubusercontent.com/17304743/81491200-555b1a80-9259-11ea-95c1-ae98b36f3779.png)
