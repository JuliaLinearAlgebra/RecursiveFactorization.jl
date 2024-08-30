using LoopVectorization
using Base: @propagate_inbounds
using TriangularSolve: ldiv!, schur_complement!
using LinearAlgebra: BlasInt, BlasFloat, LU, UnitLowerTriangular, checknonsingular, BLAS,
                     LinearAlgebra, Adjoint, Transpose, UpperTriangular, AbstractVecOrMat
using StrideArraysCore
using StrideArraysCore: square_view, unsafe_getindex, unsafe_setindex!
using Polyester: @batch

# 1.7 compat
normalize_pivot(t::Val{T}) where {T} = t
to_stdlib_pivot(t) = t
if VERSION >= v"1.7.0-DEV.1188"
    normalize_pivot(::LinearAlgebra.RowMaximum) = Val(true)
    normalize_pivot(::LinearAlgebra.NoPivot) = Val(false)
    to_stdlib_pivot(::Val{true}) = LinearAlgebra.RowMaximum()
    to_stdlib_pivot(::Val{false}) = LinearAlgebra.NoPivot()
end

function lu(A::AbstractMatrix, pivot = Val(true), thread = Val(false); kwargs...)
    return lu!(copy(A), normalize_pivot(pivot), thread; kwargs...)
end

const CUSTOMIZABLE_PIVOT = VERSION >= v"1.8.0-DEV.1507"

struct NotIPIV <: AbstractVector{BlasInt}
    len::Int
end
Base.size(A::NotIPIV) = (A.len,)
Base.getindex(::NotIPIV, i::Int) = i
Base.view(::NotIPIV, r::AbstractUnitRange) = NotIPIV(length(r))
Base.pointer(p::NotIPIV) = p
function init_pivot(::Val{false}, minmn)
    @static if CUSTOMIZABLE_PIVOT
        NotIPIV(minmn)
    else
        init_pivot(Val(true), minmn)
    end
end
init_pivot(::Val{true}, minmn) = Vector{BlasInt}(undef, minmn)

if CUSTOMIZABLE_PIVOT && isdefined(LinearAlgebra, :_ipiv_cols!)
    function LinearAlgebra._ipiv_cols!(::LU{<:Any, <:Any, NotIPIV}, ::OrdinalRange,
            B::StridedVecOrMat)
        return B
    end
end
if CUSTOMIZABLE_PIVOT && isdefined(LinearAlgebra, :_ipiv_rows!)
    function LinearAlgebra._ipiv_rows!(::(LU{T, <:AbstractMatrix{T}, NotIPIV} where {T}),
            ::OrdinalRange,
            B::StridedVecOrMat)
        return B
    end
end
if CUSTOMIZABLE_PIVOT
    function LinearAlgebra.ldiv!(A::LU{T, <:StridedMatrix, <:NotIPIV},
            B::StridedVecOrMat{T}) where {T <: BlasFloat}
        tri = @inbounds square_view(A.factors, size(A.factors, 1))
        ldiv!(UpperTriangular(A.factors), ldiv!(UnitLowerTriangular(A.factors), B))
    end
end

function lu!(A, pivot = Val(true), thread = Val(false); kwargs...)
    m, n = size(A)
    minmn = min(m, n)
    npivot = normalize_pivot(pivot)
    # we want the type on both branches to match. When pivot = Val(false), we construct
    # a `NotIPIV`, which `LinearAlgebra.generic_lufact!` does not.
    if pivot === Val(true) && minmn < 10 # avx introduces small performance degradation
        LinearAlgebra.generic_lufact!(A, to_stdlib_pivot(pivot); check = false)
    else
        lu!(A, init_pivot(npivot, minmn), npivot, thread; kwargs...)
    end
end

for (f, T) in [(:adjoint, :Adjoint), (:transpose, :Transpose)], lu in (:lu, :lu!)
    @eval $lu(A::$T, args...; kwargs...) = $f($lu(parent(A), args...; kwargs...))
end

# AVX512 needs a smaller recursion limit
pick_threshold() = LoopVectorization.register_size() == 64 ? static(48) : static(40)

recurse(::StridedArray) = true
recurse(_) = false
const LPtr{T} = Core.LLVMPtr{T, 0}
_lptr(x::Ptr{T}) where {T} = Base.bitcast(LPtr{T}, x)::LPtr{T}
_ptr(x::LPtr{T}) where {T} = Base.bitcast(Ptr{T}, x)::Ptr{T}
_lptr(x::NotIPIV) = x
_ptr(x::NotIPIV) = x

_ptrarray(ipiv) = PtrArray(ipiv)
_ptrarray(ipiv::NotIPIV) = ipiv
function _lu!(A::AbstractMatrix{T}, ipiv::AbstractVector{<:Integer},
        pivot = LinearAlgebra.RowMaximum(), thread = Val(false),
        # the performance is not sensitive wrt blocksize, and 8 is a good default
        blocksize = static(16),
        threshold = pick_threshold()) where {T}
    pivot = normalize_pivot(pivot)
    m, n = size(A)
    mnmin = min(m, n)
    if pivot === Val(false) && !CUSTOMIZABLE_PIVOT
        copyto!(ipiv, 0:(mnmin - 1))
    end
    if recurse(A) && mnmin > threshold
        if T <: Union{Float32, Float64}
            GC.@preserve ipiv A begin
                info = recurse!(view(PtrArray(A), axes(A)...), pivot,
                    m, n, mnmin,
                    _ptrarray(ipiv), blocksize,
                    thread)
            end
        else
            info = recurse!(A, pivot, m, n, mnmin, ipiv, blocksize, thread)
        end
    else # generic fallback
        info = _generic_lufact!(
            _lptr(pointer(A)), pivot, size(A, 1), size(A, 2), stride(A, 2), _lptr(pointer(ipiv)), 0)
    end
    LU(A, ipiv, info)
end
function lu!(A::AbstractMatrix{T}, ipiv::AbstractVector{<:Integer},
        pivot = Val(true), thread = Val(false);
        # the performance is not sensitive wrt blocksize, and 8 is a good default
        blocksize = static(16),
        threshold = pick_threshold(), check = nothing) where {T}
    F = _lu!(A, ipiv, pivot, thread, blocksize, threshold)
    (F.ipiv isa NotIPIV) || (F.ipiv .+= 1)
    F
end

@inline function recurse!(A, ::Val{Pivot}, m, n, mnmin, ipiv, blocksize,
        ::Val{true}) where {Pivot}
    if length(A) * _sizeof(eltype(A)) >
       0.92 * LoopVectorization.VectorizationBase.cache_size(Val(2))
        _recurse!(A, Val{Pivot}(), m, n, mnmin, ipiv, blocksize, Val(true))
    else
        _recurse!(A, Val{Pivot}(), m, n, mnmin, ipiv, blocksize, Val(false))
    end
end
@inline function recurse!(A, ::Val{Pivot}, m, n, mnmin, ipiv, blocksize,
        ::Val{false}) where {Pivot}
    _recurse!(A, Val{Pivot}(), m, n, mnmin, ipiv, blocksize, Val(false))
end

# to ensure we have the call, so we can substitute it for JuliaSimCompilerRuntime
@noinline function _ldiv!(
        pA::Core.LLVMPtr{Float64, 0}, pB::Core.LLVMPtr{Float64, 0}, M::Int, N::Int)
    A = PtrArray(Base.bitcast(Ptr{Float64}, pA), (M, M))
    B = PtrArray(Base.bitcast(Ptr{Float64}, pB), (M, N))
    # We're just generating code and substituting this function for `TriangularSolve.ldiv!`
    # We have the naive implementaiton here instead to save time, i.e., save us from having to delete an entire callgraph when we replace this definition wth a declaration.
    # TriangularSolve.ldiv!(UnitLowerTriangular(A), B, Val(false))
    @inbounds for k in 1:N
        C1 = B[1, k] = 1.0 \ B[1, k]
        # fill C-column
        for i in 2:M
            B[i, k] = 1.0 \ B[i, k] - A[i, 1] * C1
        end
        for j in 2:M
            Cj = B[j, k]
            for i in (j + 1):M
                B[i, k] -= A[i, j] * Cj
            end
        end
    end
    return nothing
end
function _ldiv!(A::UnitLowerTriangular, B::AbstractMatrix)
    M, N = size(B)
    Ad = A.data
    T = Core.LLVMPtr{Float64, 0}
    GC.@preserve Ad B begin
        _ldiv!(Base.bitcast(T, pointer(Ad)), Base.bitcast(T, pointer(B)), Int(M), Int(N))
    end
end

@inline function _recurse!(A, ::Val{Pivot}, m, n, mnmin, ipiv, blocksize,
        ::Val{Thread})::Int where {Pivot, Thread}
    GC.@preserve A ipiv begin
        info = reckernel!(_lptr(pointer(A)), Val(Pivot), m, mnmin, stride(A, 2),
            _lptr(pointer(ipiv)), blocksize, Val(Thread), 0)
        info == 0 || return info
    end
    @inbounds if m < n # fat matrix
        # [AL AR]
        AL = square_view(A, m)
        AR = @view A[:, (m + 1):n]
        Pivot && apply_permutation!(ipiv, AR, Val{Thread}(), 0)
        _ldiv!(UnitLowerTriangular(AL), AR)
    end
    0
end

@inline function nsplit(::Type{T}, n) where {T}
    k = max(2, 128 ÷ (isbitstype(T) ? sizeof(T) : 8))
    k_2 = k ÷ 2
    return n >= k ? ((n + k_2) ÷ k) * k_2 : n ÷ 2
end

function apply_permutation!(P, A, ::Val{true}, offset)
    batchsize = cld(2000, length(P))
    @batch minbatch=batchsize for j in axes(A, 2)
        for i in axes(P, 1)
            i′ = (unsafe_getindex(P, i) - offset) + 1
            tmp = unsafe_getindex(A, i, j)
            unsafe_setindex!(A, unsafe_getindex(A, i′, j), i, j)
            unsafe_setindex!(A, tmp, i′, j)
        end
    end
    nothing
end
_sizeof(::Type{T}) where {T} = Base.isbitstype(T) ? sizeof(T) : sizeof(Int)
Base.@propagate_inbounds function apply_permutation!(P, A, ::Val{false}, offset)
    for i in 0:(length(P) - 1)
        i′ = unsafe_getindex(P, i + 1) - offset
        i′ == i && continue
        for j in 0:(size(A, 2) - 1)
            tmp = unsafe_getindex(A, i + 1, j + 1)
            unsafe_setindex!(A, unsafe_getindex(A, i′ + 1, j + 1), i + 1, j + 1)
            unsafe_setindex!(A, tmp, i′ + 1, j + 1)
        end
    end
    nothing
end
function reckernel!(
        lAp::LPtr{T}, _::Val{Pivot}, m, n, ax, lip::Union{LPtr{Int}, NotIPIV}, blocksize,
        thread, offset::Int)::Int where {T, Pivot}
    @inbounds begin
        if n <= max(blocksize, 1)
            return _generic_lufact!(lAp, Val(Pivot), m, n, ax, lip, offset)
        end
        A = PtrArray(_ptr(lAp), (m, n), (nothing, StrideArraysCore.StrideReset(ax)))
        ipiv = lip isa NotIPIV ? lip :
               PtrArray(_ptr(lip), (n,), (nothing,), (static(1),), Val((1,)))
        n1 = nsplit(T, n)
        n2 = n - n1
        m2 = m - n1

        # ======================================== #
        # Now, our LU process looks like this
        # [ P1 ] [ A11 A12 ]   [ L11 0 ] [ U11 U12  ]
        # [    ] [         ] = [       ] [          ]
        # [ P2 ] [ A21 A22 ]   [ L21 I ] [ 0   A′22 ]
        # ======================================== #

        # ======================================== #
        # Partition the matrix A
        # [AL AR]
        AL = @view A[:, 1:n1]
        AR = @view A[:, (n1 + 1):n]
        #  AL  AR
        # [A11 A12]
        # [A21 A22]
        A11 = square_view(A, n1)
        A12 = @view A[1:n1, (n1 + 1):n]
        A21 = @view A[(n1 + 1):m, 1:n1]
        A22 = @view A[(n1 + 1):m, (n1 + 1):n]
        # [P1]
        # [P2]
        P1 = @view ipiv[1:n1]
        P2 = @view ipiv[(n1 + 1):n]

        # ========================================

        #   [ A11 ]   [ L11 ]
        # P [     ] = [     ] U11
        #   [ A21 ]   [ L21 ]
        info = reckernel!(
            _lptr(pointer(AL)), Val(Pivot), m, n1, ax, _lptr(pointer(P1)), blocksize, thread, offset)
        info == 0 || return info
        # [ A12 ]    [ P1 ] [ A12 ]
        # [     ] <- [    ] [     ]
        # [ A22 ]    [ 0  ] [ A22 ]
        Pivot && apply_permutation!(P1, AR, thread, offset)
        # A12 = L11 U12  =>  U12 = L11 \ A12
        _ldiv!(UnitLowerTriangular(A11), A12)
        # Schur complement:
        # We have A22 = L21 U12 + A′22, hence
        # A′22 = A22 - L21 U12
        #mul!(A22, A21, A12, -one(T), one(T))
        schur_complement!(A22, A21, A12, thread)
        # record info
        # P2 A22 = L22 U22
        info = reckernel!(_lptr(pointer(A22)), Val(Pivot), m2, n2, ax,
            _lptr(pointer(P2)), blocksize, thread, offset + n1)
        info == 0 || return info
        # A21 <- P2 A21
        Pivot && apply_permutation!(P2, A21, thread, offset + n1)

        # info != previnfo && (info += n1)
        # return info
    end # inbounds
    return 0
end

#=
    Modified from https://github.com/JuliaLang/julia/blob/b56a9f07948255dfbe804eef25bdbada06ec2a57/stdlib/LinearAlgebra/src/lu.jl
    License is MIT: https://julialang.org/license
=#
function _generic_lufact!(
        lAp::LPtr, ::Val{Pivot}, m, n, ax, lip, offset::Int)::Int where {Pivot}
    A = PtrArray(
        _ptr(lAp), (m, n), (nothing, StrideArraysCore.StrideReset(ax)),
        (static(0), static(0)), Val((1, 2)))
    minmn = min(m, n)
    ipiv = lip isa NotIPIV ? lip :
           PtrArray(_ptr(lip), (minmn,), (nothing,), (static(0),), Val((1,)))
    for _k in 0:(minmn - 1)
        # find index max
        kp = _k
        if Pivot
            amax = abs(zero(eltype(A)))
            @turbo warn_check_args=false for i in _k:(m - 1)
                absi = abs(A[i, _k])
                isnewmax = absi > amax
                kp = isnewmax ? i : kp
                amax = isnewmax ? absi : amax
            end
            unsafe_setindex!(ipiv, kp + offset, _k)
        end
        if !iszero(unsafe_getindex(A, kp, _k))
            if _k != kp
                # Interchange
                for i in 0:(n - 1)
                    tmp = unsafe_getindex(A, _k, i)
                    unsafe_setindex!(A, unsafe_getindex(A, kp, i), _k, i)
                    unsafe_setindex!(A, tmp, kp, i)
                end
            end
            # Scale first column
            Akkinv = inv(unsafe_getindex(A, _k, _k))
            @turbo check_empty=true warn_check_args=false for i in (_k + 1):(m - 1)
                A[i, _k] *= Akkinv
            end
        else
            return _k + offset + 1
        end
        _k + 1 == minmn && break
        # Update the rest
        @turbo warn_check_args=false for j in (_k + 1):(n - 1), i in (_k + 1):(m - 1)
            A[i, j] -= A[i, _k] * A[_k, j]
        end
    end
    return 0
end
