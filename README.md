# RecursiveFactorization

[![Build Status](https://travis-ci.org/YingboMa/RecursiveFactorization.jl.svg?branch=master)](https://travis-ci.org/YingboMa/RecursiveFactorization.jl)
[![codecov](https://codecov.io/gh/YingboMa/RecursiveFactorization.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/YingboMa/RecursiveFactorization.jl)

---

`RecursiveFactorization.jl` is a package that collects various recursive matrix factorization algorithms.

#### Implemented Algorithms:

- Sivan Toledo's recursive left-looking LU algorithm. DOI: [10.1137/S0895479896297744](https://epubs.siam.org/doi/10.1137/S0895479896297744)

#### Performance:

For medium sized mat
```julia
julia> using BenchmarkTools

julia> import LinearAlgebra, RecursiveFactorization

julia> A = rand(40, 40);

julia> @btime RecursiveFactorization.lu($A);
  23.762 μs (19 allocations: 26.50 KiB)

julia> @btime LinearAlgebra.lu($A);
  63.846 μs (3 allocations: 13.05 KiB)

julia> A = rand(80, 80);

julia> @btime RecursiveFactorization.lu($A);
  91.609 μs (42 allocations: 102.95 KiB)

julia> @btime LinearAlgebra.lu($A);
  420.470 μs (4 allocations: 50.83 KiB)
```
