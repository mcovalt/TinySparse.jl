function pack{T1<:Number,T2<:Integer}(X::SparseMatrixCSC{T1,T2}; symmetric::Bool=false,
                                                                 hermitian::Bool=false,
                                                                 posdef::Bool=false)
    # Packs a sparse array
    _nnz = nnz(X)
    dims = (X.m, X.n)
    rowind = Vector{UInt32}(_nnz)
    colind = Vector{UInt32}(_nnz)
    nzind = Vector{UInt32}(_nnz)
    rowind_wblock = Vector{UInt32}(128)
    colind_wblock = Vector{UInt32}(128)
    nzind_wblock = Vector{UInt32}(128)
    unique_nzval = unique(X.nzval)
    # Fill row and column index
    ind = 1
    for colptr_ind = 1:length(X.colptr)-1
        @inbounds for i = X.colptr[colptr_ind]:X.colptr[colptr_ind+1]-1
            rowind[ind] = X.rowval[ind]
            colind[ind] = colptr_ind
            ind += 1
        end
    end
    # Fill non-zero index
    unique_nzvals_dict = Dict{Float64,UInt32}(zip(unique_nzval, 1:length(unique_nzval)))
    for i = 1:_nnz
        @inbounds nzind[i] = unique_nzvals_dict[X.nzval[i]]
    end
    # Compress integers
    rowind = pack(rowind)
    colind = pack(colind)
    nzind = pack(nzind)
    return TinySparseMat{T1}(f, fc, X.m, X.n, _nnz, dims, rowind, colind, nzind, unique_nzval, rowind_wblock, colind_wblock, nzind_wblock, symmetric, hermitian, posdef)
end

# Multiplication with vector
function f{T<:Number}(y::Vector{T}, A::TinySparseMat{T}, x::Vector{T})
    fill!(y, 0.0)
    nchunks = cld(A._nnz, 128)
    for chunk = 1:nchunks - 1
        unpack!(A.rowind, A.rowind_wblock, chunk)
        unpack!(A.colind, A.colind_wblock, chunk)
        unpack!(A.nzind, A.nzind_wblock, chunk)
        for i::Int64 = 1:128
            @inbounds y[A.rowind_wblock[i]] += x[A.colind_wblock[i]]*A.nzval[A.nzind_wblock[i]]
        end
    end
    unpack!(A.rowind, A.rowind_wblock, nchunks)
    unpack!(A.colind, A.colind_wblock, nchunks)
    unpack!(A.nzind, A.nzind_wblock, nchunks)
    for i = 1:A._nnz-(nchunks-1)*128
        @inbounds y[A.rowind_wblock[i]] += x[A.colind_wblock[i]]*A.nzval[A.nzind_wblock[i]]
    end
end

# Conjugate transpose multiplication with vector
function fc{T<:Number}(y::Vector{T}, A::TinySparseMat{T}, x::Vector{T})
    fill!(y, 0.0)
    nchunks = cld(A._nnz, 128)
    for chunk = 1:nchunks - 1
        unpack!(A.rowind, A.rowind_wblock, chunk)
        unpack!(A.colind, A.colind_wblock, chunk)
        unpack!(A.nzind, A.nzind_wblock, chunk)
        for i = 1:128
            @inbounds y[A.colind_wblock[i]] += x[A.rowind_wblock[i]]*A.nzval[A.nzind_wblock[i]]'
        end
    end
    unpack!(A.rowind, A.rowind_wblock, nchunks)
    unpack!(A.colind, A.colind_wblock, nchunks)
    unpack!(A.nzind, A.nzind_wblock, nchunks)
    for i = 1:A._nnz-(nchunks-1)*128
        @inbounds y[A.colind_wblock[i]] += x[A.rowind_wblock[i]]*A.nzval[A.nzind_wblock[i]]'
    end
end