# TinySparse.jl

[![Build Status](https://travis-ci.org/mcovalt/TinySparse.jl.svg?branch=master)](https://travis-ci.org/mcovalt/TinySparse.jl)

`TinySparse.jl` is a Julia package for further compressing sparse matrices. Operations on `TinySparseMat`'s are deferred until the matrix operates on a vector (see [LinearMaps.jl](https://github.com/Jutho/LinearMaps.jl)).

## Requirements
* Julia 0.5 and up
* GCC installed (Linux or macOS)
* **Windows is unsupported at this time.**

## Instalation
```julia
julia> Pkg.add("TinySpare")
julia> Pkg.test("TinySparse")
```

## What is TinySparse.jl?
`TinySparse.jl` compresses sparse matrices. Indexes are compressed using [TinyInt.jl](https://github.com/mcovalt/TinyInt.jl) and only unique non-zero entries are stored.

Consider a gradient operator.
```julia
using TinySparse
nx, ny, nz = (100, 100, 100)
h = 0.5
G1 = spdiagm((fill(1.0, nx), fill(-1.0, nx)), (0, -1), nx+1, nx)
G2 = spdiagm((fill(1.0, ny), fill(-1.0, ny)), (0, -1), ny+1, ny)
G3 = spdiagm((fill(1.0, nz), fill(-1.0, nz)), (0, -1), nz+1, nz)
Gx = (1/h)*kron(speye(nz), kron(speye(ny), G1))
Gy = (1/h)*kron(speye(nz), kron(G2, speye(nx)))
Gz = (1/h)*kron(G3, kron(speye(ny), speye(nx)))
gradientOp = vcat(Gx, Gy, Gz)
tinyGradOp = pack(gradientOp)
Base.summarysize(gradientOp)
# 104000048
Base.summarysize(tinyGradOp)
# 22079893
```
Multiplication speed is sligthly slower than Julia's default method.
```julia
@time A_mul_B!(out, gradientOp, x)
# 0.014913 seconds (4 allocations: 160 bytes)
@time A_mul_B!(out, tinyGradOp, x)
# 0.018533 seconds (4 allocations: 160 bytes)
```

*These times are from a quad-core Intel® Core™ i7-4790 CPU @ 3.60GHz*

## Functions
Function              | Description
--------------------- | ------------
`pack(x)`             | Compresses sparse matrix.
`+, *, A_mul_B!...`   | All operations supported in [LinearMaps.jl](https://github.com/Jutho/LinearMaps.jl) works here.

## Notes
* **The compressed sparse matrix is immutable.** Once a compressed sparse matrix has been made, it's elements cannot be changed.
* **Speed decreases as more unique values are present.** Currently, only two or so unique values are competetive with Julia's ordinary sparse matrix vector multiplication. This is likely to change in the future (see next section).

## To Do
* **Implement CSC format.** Currently COO sparse matrix format is implemented. If we change this to CSC, we can reduce the amount of memory calls to near the ordinary level of memory calls. This should speed up sparse matrix vector multiplication. It may even produce faster multiplication due to [cache-locality](https://en.wikipedia.org/wiki/Locality_of_reference) of the unique non-zero elements.
* **Implement SIMD operations.** Through some testing, I've noticed gathering multiplication inputs into some linear buffer, performing SIMD operations on them, then scattering the results into the output vector offers some speed improvements over just serially accessesing the irregular memory pattern of the vectors. It would be nice to implement this.
