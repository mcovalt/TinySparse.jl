module TinySparse
    # Required modules
    using TinyInt
    using LinearMaps
    # Overloaded functions
    import Base: size, nnz, issymmetric, ishermitian, isposdef, A_mul_B!, *, At_mul_B!, Ac_mul_B!, Ac_mul_B
    import TinyInt: pack
    # Type definition and overloaded functions
    include("tinysparsemat.jl")
    # Function to pack sparse matrices
    include("compress.jl")

    export TinySparseMat, pack
end