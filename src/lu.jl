using LoopVectorization
using TriangularSolve: ldiv!
using LinearAlgebra: BlasInt, BlasFloat, LU, UnitLowerTriangular, checknonsingular, BLAS,
                     LinearAlgebra, Adjoint, Transpose, UpperTriangular, AbstractVecOrMat
using StrideArraysCore
using StrideArraysCore: square_view
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
# Julia 1.11+ uses negative info for NoPivot() failures
const NOPIVOT_NEGATIVE_INFO = VERSION >= v"1.11.0-DEV"

struct NotIPIV <: AbstractVector{BlasInt}
    len::Int
end
Base.size(A::NotIPIV) = (A.len,)
Base.getindex(::NotIPIV, i::Int) = i
Base.view(::NotIPIV, r::AbstractUnitRange) = NotIPIV(length(r))
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

function lu!(A, pivot = Val(true), thread = Val(false);
        check::Union{Bool, Val{true}, Val{false}} = Val(true), kwargs...)
    m, n = size(A)
    minmn = min(m, n)
    npivot = normalize_pivot(pivot)
    # we want the type on both branches to match. When pivot = Val(false), we construct
    # a `NotIPIV`, which `LinearAlgebra.generic_lufact!` does not.
    F = if pivot === Val(true) && minmn < 10 # avx introduces small performance degradation
        LinearAlgebra.generic_lufact!(A, to_stdlib_pivot(pivot);
            check = ((check isa Bool && check) || (check === Val(true)))
        )
    else
        lu!(A, init_pivot(npivot, minmn), npivot, thread; check = check,
            kwargs...)
    end
    return F
end

for (f, T) in [(:adjoint, :Adjoint), (:transpose, :Transpose)], lu in (:lu, :lu!)
    @eval $lu(A::$T, args...; kwargs...) = $f($lu(parent(A), args...; kwargs...))
end

# AVX512 needs a smaller recursion limit
pick_threshold() = LoopVectorization.register_size() == 64 ? 48 : 40

recurse(::StridedArray) = true
recurse(_) = false

_ptrarray(ipiv) = PtrArray(ipiv)
_ptrarray(ipiv::NotIPIV) = ipiv
function lu!(A::AbstractMatrix{T}, ipiv::AbstractVector{<:Integer},
        pivot = Val(true), thread = Val(false);
        check::Union{Bool, Val{true}, Val{false}} = Val(true),
        # the performance is not sensitive wrt blocksize, and 8 is a good default
        blocksize::Integer = length(A) ≥ 40_000 ? 8 : 16,
        threshold::Integer = pick_threshold()) where {T}
    pivot = normalize_pivot(pivot)
    info = zero(BlasInt)
    m, n = size(A)
    mnmin = min(m, n)
    if pivot === Val(false) && !CUSTOMIZABLE_PIVOT
        copyto!(ipiv, 1:mnmin)
    end
    if recurse(A) && mnmin > threshold
        if T <: Union{Float32, Float64}
            GC.@preserve ipiv A begin
                info = recurse!(view(PtrArray(A), axes(A)...), pivot,
                    m, n, mnmin,
                    _ptrarray(ipiv), info, blocksize,
                    thread)
            end
        else
            info = recurse!(A, pivot, m, n, mnmin, ipiv, info, blocksize, thread)
        end
    else # generic fallback
        info = _generic_lufact!(A, pivot, ipiv, info)
    end
    ((check isa Bool && check) || (check === Val(true))) && checknonsingular(info)
    LU(A, ipiv, info)
end

@inline function recurse!(A, ::Val{Pivot}, m, n, mnmin, ipiv, info, blocksize,
        ::Val{true}) where {Pivot}
    if length(A) * _sizeof(eltype(A)) >
       0.92 * LoopVectorization.VectorizationBase.cache_size(Val(2))
        _recurse!(A, Val{Pivot}(), m, n, mnmin, ipiv, info, blocksize, Val(true))
    else
        _recurse!(A, Val{Pivot}(), m, n, mnmin, ipiv, info, blocksize, Val(false))
    end
end
@inline function recurse!(A, ::Val{Pivot}, m, n, mnmin, ipiv, info, blocksize,
        ::Val{false}) where {Pivot}
    _recurse!(A, Val{Pivot}(), m, n, mnmin, ipiv, info, blocksize, Val(false))
end
@inline function _recurse!(A, ::Val{Pivot}, m, n, mnmin, ipiv, info, blocksize,
        ::Val{Thread}) where {Pivot, Thread}
    info = reckernel!(A, Val(Pivot), m, mnmin, ipiv, info, blocksize, Val(Thread))::Int
    @inbounds if m < n # fat matrix
        # [AL AR]
        AL = square_view(A, m)
        AR = @view A[:, (m + 1):n]
        Pivot && apply_permutation!(ipiv, AR, Val{Thread}())
        ldiv!(UnitLowerTriangular(AL), AR, Val{Thread}())
    end
    info
end

@inline function nsplit(::Type{T}, n) where {T}
    k = max(2, 128 ÷ (isbitstype(T) ? sizeof(T) : 8))
    k_2 = k ÷ 2
    return n >= k ? ((n + k_2) ÷ k) * k_2 : n ÷ 2
end

function apply_permutation!(P, A, ::Val{true})
    batchsize = cld(2000, length(P))
    @batch minbatch=batchsize for j in axes(A, 2)
        @inbounds for i in axes(P, 1)
            i′ = P[i]
            tmp = A[i, j]
            A[i, j] = A[i′, j]
            A[i′, j] = tmp
        end
    end
    nothing
end
_sizeof(::Type{T}) where {T} = Base.isbitstype(T) ? sizeof(T) : sizeof(Int)
Base.@propagate_inbounds function apply_permutation!(P, A, ::Val{false})
    for i in axes(P, 1)
        i′ = P[i]
        i′ == i && continue
        @simd for j in axes(A, 2)
            tmp = A[i, j]
            A[i, j] = A[i′, j]
            A[i′, j] = tmp
        end
    end
    nothing
end
function reckernel!(A::AbstractMatrix{T}, pivot::Val{Pivot}, m, n, ipiv, info, blocksize,
        thread)::BlasInt where {T, Pivot}
    @inbounds begin
        if n <= max(blocksize, 1)
            info = _generic_lufact!(A, Val(Pivot), ipiv, info)
            return info
        end
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
        info = reckernel!(AL, Val(Pivot), m, n1, P1, info, blocksize, thread)
        # [ A12 ]    [ P1 ] [ A12 ]
        # [     ] <- [    ] [     ]
        # [ A22 ]    [ 0  ] [ A22 ]
        Pivot && apply_permutation!(P1, AR, thread)
        # A12 = L11 U12  =>  U12 = L11 \ A12
        ldiv!(UnitLowerTriangular(A11), A12, thread)
        # Schur complement:
        # We have A22 = L21 U12 + A′22, hence
        # A′22 = A22 - L21 U12
        #mul!(A22, A21, A12, -one(T), one(T))
        schur_complement!(A22, A21, A12, thread)
        # record info
        previnfo = info
        # P2 A22 = L22 U22
        info = reckernel!(A22, Val(Pivot), m2, n2, P2, info, blocksize, thread)
        # A21 <- P2 A21
        Pivot && apply_permutation!(P2, A21, thread)

        if info != previnfo
            # Handle negative info for NoPivot (Julia 1.11+ convention)
            if NOPIVOT_NEGATIVE_INFO && info < 0
                info -= n1
            else
                info += n1
            end
        end
        if Pivot
            @turbo warn_check_args=false for i in 1:n2
                P2[i] += n1
            end
        end
        return info
    end # inbounds
end

function schur_complement!(𝐂, 𝐀, 𝐁, ::Val{THREAD} = Val(true)) where {THREAD}
    # mul!(𝐂,𝐀,𝐁,-1,1)
    if THREAD
        @tturbo warn_check_args=false for m in 1:size(𝐀, 1), n in 1:size(𝐁, 2)
            𝐂ₘₙ = zero(eltype(𝐂))
            for k in 1:size(𝐀, 2)
                𝐂ₘₙ -= 𝐀[m, k] * 𝐁[k, n]
            end
            𝐂[m, n] = 𝐂ₘₙ + 𝐂[m, n]
        end
    else
        @turbo warn_check_args=false for m in 1:size(𝐀, 1), n in 1:size(𝐁, 2)
            𝐂ₘₙ = zero(eltype(𝐂))
            for k in 1:size(𝐀, 2)
                𝐂ₘₙ -= 𝐀[m, k] * 𝐁[k, n]
            end
            𝐂[m, n] = 𝐂ₘₙ + 𝐂[m, n]
        end
    end
end

#=
    Modified from https://github.com/JuliaLang/julia/blob/b56a9f07948255dfbe804eef25bdbada06ec2a57/stdlib/LinearAlgebra/src/lu.jl
    License is MIT: https://julialang.org/license
=#
function _generic_lufact!(A, ::Val{Pivot}, ipiv, info) where {Pivot}
    m, n = size(A)
    minmn = length(ipiv)
    @inbounds begin
        for k in 1:minmn
            # find index max
            kp = k
            if Pivot
                amax = abs(zero(eltype(A)))
                @turbo warn_check_args=false for i in k:m
                    absi = abs(A[i, k])
                    isnewmax = absi > amax
                    kp = isnewmax ? i : kp
                    amax = isnewmax ? absi : amax
                end
                ipiv[k] = kp
            end
            if !iszero(A[kp, k])
                if k != kp
                    # Interchange
                    @simd for i in 1:n
                        tmp = A[k, i]
                        A[k, i] = A[kp, i]
                        A[kp, i] = tmp
                    end
                end
                # Scale first column
                Akkinv = inv(A[k, k])
                @turbo check_empty=true warn_check_args=false for i in (k + 1):m
                    A[i, k] *= Akkinv
                end
            elseif info == 0
                info = k
                # Julia 1.11+ convention: negative info for NoPivot
                if !Pivot && NOPIVOT_NEGATIVE_INFO
                    info = -info
                end
            end
            k == minmn && break
            # Update the rest
            @turbo warn_check_args=false for j in (k + 1):n
                for i in (k + 1):m
                    A[i, j] -= A[i, k] * A[k, j]
                end
            end
        end
    end
    return info
end
