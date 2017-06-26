immutable TinySparseMat{T} <: AbstractLinearMap{T}
    f::Function
    fc::Function
    M::Int
    N::Int
    _nnz::Int
    dims::Tuple{Int, Int}
    rowind::TinyIntVec
    colind::TinyIntVec
    nzind::TinyIntVec
    nzval::Vector{T}
    rowind_wblock::Vector{UInt32}
    colind_wblock::Vector{UInt32}
    nzind_wblock::Vector{UInt32}
    _issymmetric::Bool
    _ishermitian::Bool
    _isposdef::Bool
end

# Properties
Base.size(A::TinySparseMat) = A.dims
Base.nnz(A::TinySparseMat) = A._nnz
Base.issymmetric(A::TinySparseMat) = A._issymmetric
Base.ishermitian(A::TinySparseMat) = A._ishermitian
Base.isposdef(A::TinySparseMat) = A._isposdef

# Multiplication with vector
function Base.A_mul_B!(y::AbstractVector, A::TinySparseMat, x::AbstractVector)
    (length(x) == A.N && length(y) == A.M) || throw(DimensionMismatch())
    A.f(y,A,x)
    return y
end

function *(A::TinySparseMat, x::AbstractVector)
    length(x) == A.N || throw(DimensionMismatch())
    y = similar(x, promote_type(eltype(A), eltype(x)), A.M)
    A.f(y,A,x)
    return y
end

function Base.At_mul_B!(y::AbstractVector, A::TinySparseMat, x::AbstractVector)
    issymmetric(A) && return Base.A_mul_B!(y, A, x)
    (length(x) == A.M && length(y) == A.N) || throw(DimensionMismatch())
    if !isreal(A)
        x = conj(x)
    end
    A.fc(y,A,x)
    if !isreal(A)
        conj!(y)
    end
    return y
end
function Base.At_mul_B(A::TinySparseMat, x::AbstractVector)
    issymmetric(A) && return A*x
    length(x) == A.M || throw(DimensionMismatch())
    if !isreal(A)
        x = conj(x)
    end
    y = similar(x, promote_type(eltype(A), eltype(x)), A.N)
    A.fc(y,A,x)
    if !isreal(A)
        conj!(y)
    end
    return y
end

function Base.Ac_mul_B!(y::AbstractVector, A::TinySparseMat, x::AbstractVector)
    ishermitian(A) && return Base.A_mul_B!(y,A,x)
    (length(x) == A.M && length(y) == A.N) || throw(DimensionMismatch())
    return A.fc(y,A,x)
end

function Base.Ac_mul_B(A::TinySparseMat, x::AbstractVector)
    ishermitian(A) && return A*x
    length(x) == A.M || throw(DimensionMismatch())
    y = similar(x, promote_type(eltype(A), eltype(x)), A.N)
    A.fc(y,A,x)
    return y
end