using LinearAlgebra: BlasInt, BlasFloat, LU, UnitLowerTriangular, ldiv!, BLAS, checknonsingular

function lu(A::AbstractMatrix, pivot::Union{Val{false}, Val{true}} = Val(true);
            check::Bool = true, blocksize::Integer = 16)
    lu!(copy(A), pivot; check = check, blocksize = blocksize)
end

function lu!(A, pivot::Union{Val{false}, Val{true}} = Val(true);
             check::Bool = true, blocksize::Integer = 16)
    lu!(A, Vector{BlasInt}(undef, min(size(A)...)), pivot;
        check = check, blocksize = blocksize)
end

function lu!(A::AbstractMatrix{T}, ipiv::AbstractVector{<:Integer},
             pivot::Union{Val{false}, Val{true}} = Val(true);
             check::Bool=true, blocksize::Integer=16) where T
    info = Ref(zero(BlasInt))
    m, n = size(A)
    mnmin = min(m, n)
    if T <: BlasFloat && A isa StridedArray
        reckernel!(A, pivot, m, mnmin, ipiv, info, blocksize)
        if m < n # fat matrix
            # [AL AR]
            AL = @view A[:, 1:m]
            AR = @view A[:, m+1:n]
            apply_permutation!(ipiv, AR)
            ldiv!(UnitLowerTriangular(AL), AR)
        end
      else # generic fallback
        _generic_lufact!(A, pivot, ipiv, info)
    end
    check && checknonsingular(info[])
    LU{T, typeof(A)}(A, ipiv, info[])
end

function nsplit(::Type{T}, n) where T
    k = 128 ÷ sizeof(T)
    k_2 = k ÷ 2
    return n >= k ? ((n + k_2) ÷ k) * k_2 : n ÷ 2
end

Base.@propagate_inbounds function apply_permutation!(P, A)
    for i in axes(P, 1)
        i′ = P[i]
        i′ == i && continue
        @simd for j in axes(A, 2)
            A[i, j], A[i′, j] = A[i′, j], A[i, j]
        end
    end
    nothing
end

function reckernel!(A::AbstractMatrix{T}, pivot::Val{Pivot}, m, n, ipiv, info, blocksize)::Nothing where {T,Pivot}
    @inbounds begin
        if n <= max(blocksize, 1)
            _generic_lufact!(A, pivot, ipiv, info)
            return nothing
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
        reckernel!(AL, pivot, m, n1, P1, info, blocksize)
        # [ A12 ]    [ P1 ] [ A12 ]
        # [     ] <- [    ] [     ]
        # [ A22 ]    [ 0  ] [ A22 ]
        Pivot && apply_permutation!(P1, AR)
        # A12 = L11 U12  =>  U12 = L11 \ A12
        ldiv!(UnitLowerTriangular(A11), A12)
        # Schur complement:
        # We have A22 = L21 U12 + A′22, hence
        # A′22 = A22 - L21 U12
        BLAS.gemm!('N', 'N', -one(T), A21, A12, one(T), A22)
        # record info
        previnfo = info[]
        # P2 A22 = L22 U22
        reckernel!(A22, pivot, m2, n2, P2, info, blocksize)
        # A21 <- P2 A21
        Pivot && apply_permutation!(P2, A21)

        info[] != previnfo && (info[] += n1)
        @simd for i in 1:n2
            P2[i] += n1
        end
        return nothing
    end # inbounds
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
                @simd for i = k+1:m
                    A[i,k] *= Akkinv
                end
            elseif info[] == 0
                info[] = k
            end
            # Update the rest
            for j = k+1:n
                @simd for i = k+1:m
                    A[i,j] -= A[i,k]*A[k,j]
                end
            end
        end
    end
    return nothing
end
