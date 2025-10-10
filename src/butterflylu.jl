using VectorizedRNG
using LinearAlgebra: Diagonal, I
using LoopVectorization
using RecursiveFactorization
using SparseBandedMatrices

@inline exphalf(x) = exp(x) * oftype(x, 0.5)
function generate_rand_butterfly_vals!(wv, ::Val{SEED} = Val(888)) where {SEED}
    T = eltype(wv)
    mrng = VectorizedRNG.MutableXoshift(SEED)
    GC.@preserve mrng begin rand!(exphalf, VectorizedRNG.Xoshift(mrng), wv, static(0),
                                  T(-0.05), T(0.1)) end
end

function 🦋generate_random!(A, ::Val{SEED} = Val(888)) where {SEED}
    uv = similar(A, 4 * size(A, 1))
    generate_rand_butterfly_vals!(uv, Val(SEED))
    uv
end
struct 🦋workspace{T}
    A::Matrix{T}
    b::Vector{T}
    ws::Vector{T}
    U::Matrix{T}
    V::Matrix{T}
    out::Vector{T}
    n::Int
    function 🦋workspace(A, b, ::Val{SEED} = Val(888)) where {SEED}
        len = size(A, 1)
        out = similar(b, N)
        if (M % 4 != 0)
            A = pad!(A)
            xn = 4 - M % 4
            b = [b; rand(xn)]
        end
        U, V = (similar(A), similar(A))
        ws = 🦋generate_random!(A)
        materializeUV(U, V, ws)
        new{eltype(A)}(A, b, ws, U, V, out, n)
    end
end

function 🦋solve!(workspace::🦋workspace, thread)
    (;A, b, ws, U, V, out, n) = workspace
    🦋mul!(A, ws)
    F = RecursiveFactorization.lu!(A, Val(false), thread)
    
    mul!(b, U', b)
    ldiv!(b, UnitLowerTriangular(F.factors), b, thread)
    ldiv!(b, UpperTriangular(F.factors), b, thread)
    mul!(b, V, b)
    out .= @view b[1:n]
    out
end

const butterfly_workspace = 🦋workspace;

function 🦋mul_level!(A, u, v)
    M, N = size(A)
    @assert M == length(u) && N == length(v)
    M_half = M >>> 1
    N_half = N >>> 1
    @turbo for n in 1 : N_half
        for m in 1 : M_half
            A11 = A[m, n]
            A21 = A[m + M_half, n]
            A12 = A[m, n + N_half]
            A22 = A[m + M_half, n + N_half]

            T1 = A11 + A12
            T2 = A21 + A22
            T3 = A11 - A12
            T4 = A21 - A22
            C11 = T1 + T2
            C21 = T1 - T2
            C12 = T3 + T4
            C22 = T3 - T4

            u1 = u[m]
            u2 = u[m + M_half]
            v1 = v[n]
            v2 = v[n + N_half]

            A[m, n] = u1 * C11 * v1
            A[m + M_half, n] = u2 * C21 * v1
            A[m, n + N_half] = u1 * C12 * v2
            A[m + M_half, n + N_half] = u2 * C22 * v2
        end
    end 
end

function 🦋mul!(A, uv)
    M, N = size(A)
    @assert M == N
    M_half = M >>> 1

    U₁ = @view(uv[1:M_half]) 
    V₁ = @view(uv[(M_half + 1):(M)]) 
    U₂ = @view(uv[(1 + M):(M + M_half)]) 
    V₂ = @view(uv[(1 + M + M_half):(2 * M)]) 

    🦋mul_level!(@view(A[1:M_half, 1:M_half]), U₁, V₁) 
    🦋mul_level!(@view(A[M_half + 1:M, 1:M_half]), U₂, V₁) 
    🦋mul_level!(@view(A[1:M_half, M_half + 1:M]), U₁, V₂) 
    🦋mul_level!(@view(A[M_half + 1:M, M_half + 1:M]), U₂, V₂) 

    U = @view(uv[(1 + 2 * M):(3 * M)]) 
    V = @view(uv[(1 + 3 * M):(4 * M)]) 

    🦋mul_level!(@view(A[1:M, 1:N]), U, V)
    A
end

function diagnegbottom(x)
    N = length(x)
    y = similar(x, N >>> 1)
    z = similar(x, N >>> 1)
    for n in 1:(N >>> 1)
        y[n] = x[n]
    end
    for n in 1:(N >>> 1)
        z[n] = x[n + (N >>> 1)]
    end
    Diagonal(y), Diagonal(z)
end

function 🦋!(C::SparseBandedMatrix, A::Diagonal, B::Diagonal)
    setdiagonal!(C, [A.diag; -B.diag], true)
    setdiagonal!(C, A.diag, true)
    setdiagonal!(C, B.diag, false)
    C
end

function 🦋!(C, A::Diagonal, B::Diagonal)
    @assert size(A) == size(B)
    A1 = size(A, 1)

    for i in 1:A1
        C[i, i] = A[i, i]
        C[i + A1, i] = A[i, i]
        C[i, i + A1] = B[i, i]
        C[i + A1, i + A1] = -B[i, i]
    end

    C
end

function materializeUV(U, V, uv)
    M = size(U, 1)
    M_half = M >>> 1   

    U₁u, U₁l = diagnegbottom(@view(uv[1:M_half])) #M_half
    U₂u, U₂l = diagnegbottom(@view(uv[(1 + 2 * M_half):(M + M_half)])) #M_half
    V₁u, V₁l = diagnegbottom(@view(uv[(M_half + 1):(2 * M_half)])) #M_half
    V₂u, V₂l = diagnegbottom(@view(uv[(1 + 3 * M_half):(2 * M_half + M)])) #M_half
    Uu, Ul = diagnegbottom(@view(uv[(1 + 2 * M):(3 * M)])) #M
    Vu, Vl = diagnegbottom(@view(uv[(1 + 3 * M):(4 * M)])) #M

    Bu2 = SparseBandedMatrix{typeof(uv[1])}(undef, M, M)
    
    🦋!(view(Bu2, 1 : M_half, 1 : M_half), U₁u, U₁l)
    🦋!(view(Bu2, M_half + 1: M, M_half + 1: M), U₂u, U₂l)

    Bu1 = SparseBandedMatrix{typeof(uv[1])}(undef, M, M)
    🦋!(Bu1, Uu, Ul)

    Bv2 = SparseBandedMatrix{typeof(uv[1])}(undef, M, M)

    🦋!(view(Bv2, 1 : M_half, 1 : M_half), V₁u, V₁l)
    🦋!(view(Bv2, M_half + 1: M, M_half + 1: M), V₂u, V₂l)

    Bv1 = SparseBandedMatrix{typeof(uv[1])}(undef, M, M)
    🦋!(Bv1, Vu, Vl)

    mul!(U, Bu2, Bu1)
    mul!(V, Bv2, Bv1)
end

function pad!(A)
    M, N = size(A)
    xn = 4 - M % 4
    A_new = similar(A, M + xn, N + xn)
    for j in 1 : N, i in 1 : M
        @inbounds A_new[i, j] = A[i, j]
    end

    for j in M + 1 : M + xn, i in 1:M
        @inbounds A_new[i, j] = 0
        @inbounds A_new[j, i] = 0
    end

    for j in N + 1 : N + xn, i in M + 1 : M + xn
        @inbounds A_new[i,j] = i == j
    end
    A_new
end
