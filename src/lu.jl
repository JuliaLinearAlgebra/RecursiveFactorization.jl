using LoopVectorization
using TriangularSolve: ldiv!
using LinearAlgebra: BlasInt, BlasFloat, LU, UnitLowerTriangular, checknonsingular, BLAS, LinearAlgebra

# 1.7 compat
normalize_pivot(t::Val{T}) where T = t
to_stdlib_pivot(t) = t
if VERSION >= v"1.7.0-DEV.1188"
    normalize_pivot(::LinearAlgebra.RowMaximum) = Val(true)
    normalize_pivot(::LinearAlgebra.NoPivot) = Val(false)
    to_stdlib_pivot(::Val{true}) = LinearAlgebra.RowMaximum()
    to_stdlib_pivot(::Val{false}) = LinearAlgebra.NoPivot()
end

function lu(A::AbstractMatrix, pivot = Val(true); kwargs...)
    return lu!(copy(A), normalize_pivot(pivot); kwargs...)
end

function lu!(A, pivot = Val(true); check=true, kwargs...)
    m, n  = size(A)
    minmn = min(m, n)
    F = if minmn < 10 # avx introduces small performance degradation
        LinearAlgebra.generic_lufact!(A, to_stdlib_pivot(pivot); check=check)
    else
        lu!(A, Vector{BlasInt}(undef, minmn), normalize_pivot(pivot); check=check, kwargs...)
    end
    return F
end

const RECURSION_THRESHOLD = Ref(-1)

# AVX512 needs a smaller recursion limit
function pick_threshold()
    RECURSION_THRESHOLD[] >= 0 && return RECURSION_THRESHOLD[]
    LoopVectorization.register_size() == 64 ? 48 : 40
end

recurse(::StridedArray) = true
recurse(::LinearAlgebra.Adjoint{<:Any,<:StridedArray}) = true
recurse(::LinearAlgebra.Transpose{<:Any,<:StridedArray}) = true
recurse(_) = false

function lu!(
    A::AbstractMatrix{T}, ipiv::AbstractVector{<:Integer},
    pivot = Val(true);
    check::Bool=true,
    # the performance is not sensitive wrt blocksize, and 16 is a good default
    blocksize::Integer=8,
    threshold::Integer=pick_threshold()
) where T
    pivot = normalize_pivot(pivot)
    info = zero(BlasInt)
    m, n = size(A)
    mnmin = min(m, n)
    if recurse(A) && mnmin > threshold
        info = reckernel!(A, pivot, m, mnmin, ipiv, info, blocksize)
        @inbounds if m < n # fat matrix
            # [AL AR]
            AL = @view A[:, 1:m]
            AR = @view A[:, m+1:n]
            apply_permutation!(ipiv, AR)
            ldiv!(UnitLowerTriangular(AL), AR)
        end
    else # generic fallback
        info = _generic_lufact!(A, pivot, ipiv, info)
    end
    check && checknonsingular(info)
    LU{T, typeof(A)}(A, ipiv, info)
end

function nsplit(::Type{T}, n) where T
    k = 512 ÷ (isbitstype(T) ? sizeof(T) : 8)
    k_2 = k ÷ 2
    return n >= k ? ((n + k_2) ÷ k) * k_2 : n ÷ 2
end

Base.@propagate_inbounds function apply_permutation!(P, A)
    @tturbo for j in axes(A, 2), i in axes(P, 1)
        i′ = P[i]
        tmp = A[i, j]
        A[i, j] = A[i′, j]
        A[i′, j] = tmp
    end
    nothing
end

function reckernel!(A::AbstractMatrix{T}, pivot::Val{Pivot}, m, n, ipiv, info, blocksize)::BlasInt where {T,Pivot}
    @inbounds begin
        if n <= max(blocksize, 1)
            info = _generic_lufact!(A, pivot, ipiv, info)
            return info
        end
        n1 = nsplit(T, n)
        n2 = n - n1
        m2 = m - n1

        # ======================================== #
        # Now, our LU process looks like this
        # [ P1 ] [ A11 A21 ]   [ L11 0 ] [ U11 U12  ]
        # [    ] [         ] = [       ] [          ]
        # [ P2 ] [ A21 A22 ]   [ L21 I ] [ 0   A′22 ]
        # ======================================== #

        # ======================================== #
        # Partition the matrix A
        # [AL AR]
        AL = @view A[:, 1:n1]
        AR = @view A[:, n1+1:n]
        #  AL  AR
        # [A11 A12]
        # [A21 A22]
        A11 = @view A[1:n1, 1:n1]
        A12 = @view A[1:n1, n1+1:n]
        A21 = @view A[n1+1:m, 1:n1]
        A22 = @view A[n1+1:m, n1+1:n]
        # [P1]
        # [P2]
        P1 = @view ipiv[1:n1]
        P2 = @view ipiv[n1+1:n]
        # ========================================

        #   [ A11 ]   [ L11 ]
        # P [     ] = [     ] U11
        #   [ A21 ]   [ L21 ]
        info = reckernel!(AL, pivot, m, n1, P1, info, blocksize)
        # [ A12 ]    [ P1 ] [ A12 ]
        # [     ] <- [    ] [     ]
        # [ A22 ]    [ 0  ] [ A22 ]
        Pivot && apply_permutation!(P1, AR)
        # A12 = L11 U12  =>  U12 = L11 \ A12
        ldiv!(UnitLowerTriangular(A11), A12)
        # Schur complement:
        # We have A22 = L21 U12 + A′22, hence
        # A′22 = A22 - L21 U12
        #mul!(A22, A21, A12, -one(T), one(T))
        schur_complement!(A22, A21, A12)
        # record info
        previnfo = info
        # P2 A22 = L22 U22
        info = reckernel!(A22, pivot, m2, n2, P2, info, blocksize)
        # A21 <- P2 A21
        Pivot && apply_permutation!(P2, A21)

        info != previnfo && (info += n1)
        @avx for i in 1:n2
            P2[i] += n1
        end
        return info
    end # inbounds
end

function schur_complement!(𝐂, 𝐀, 𝐁)
    @tturbo for m ∈ 1:size(𝐀,1), n ∈ 1:size(𝐁,2)
        𝐂ₘₙ = zero(eltype(𝐂))
        for k ∈ 1:size(𝐀,2)
            𝐂ₘₙ -= 𝐀[m,k] * 𝐁[k,n]
        end
        𝐂[m,n] = 𝐂ₘₙ + 𝐂[m,n]
    end
end

#=
    Modified from https://github.com/JuliaLang/julia/blob/b56a9f07948255dfbe804eef25bdbada06ec2a57/stdlib/LinearAlgebra/src/lu.jl
    License is MIT: https://julialang.org/license
=#
function _generic_lufact!(A, ::Val{Pivot}, ipiv, info) where Pivot
    m, n = size(A)
    minmn = length(ipiv)
    @inbounds begin
        for k = 1:minmn
            # find index max
            kp = k
            if Pivot
              amax = abs(zero(eltype(A)))
              for i = k:m
                  absi = abs(A[i,k])
                  if absi > amax
                      kp = i
                      amax = absi
                  end
              end
            end
            ipiv[k] = kp
            if !iszero(A[kp,k])
                if k != kp
                    # Interchange
                    @simd for i = 1:n
                        tmp = A[k,i]
                        A[k,i] = A[kp,i]
                        A[kp,i] = tmp
                    end
                end
                # Scale first column
                Akkinv = inv(A[k,k])
                @avx check_empty=true for i = k+1:m
                    A[i,k] *= Akkinv
                end
            elseif info == 0
                info = k
            end
            k == minmn && break
            # Update the rest
            @avx for j = k+1:n
                for i = k+1:m
                    A[i,j] -= A[i,k]*A[k,j]
                end
            end
        end
    end
    return info
end
