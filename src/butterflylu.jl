using VectorizedRNG
using LinearAlgebra: Diagonal, I
using LoopVectorization
using RecursiveFactorization
using SparseArrays, SparseBandedMatrices

@inline exphalf(x) = exp(x) * oftype(x, 0.5)
function 🦋!(wv, ::Val{SEED} = Val(888)) where {SEED}
    T = eltype(wv)
    mrng = VectorizedRNG.MutableXoshift(SEED)
    GC.@preserve mrng begin rand!(exphalf, VectorizedRNG.Xoshift(mrng), wv, static(0),
                                  T(-0.05), T(0.1)) end
end

function 🦋generate_random!(A, ::Val{SEED} = Val(888)) where {SEED}
    Usz = 2 * size(A, 1)
    Vsz = 2 * size(A, 2)
    uv = similar(A, Usz + Vsz)
    🦋!(uv, Val(SEED))
    (uv,)
end

function 🦋workspace(A, b, B::Matrix{T}, U::Adjoint{T, Matrix{T}}, V::Matrix{T}, thread, ::Val{SEED} = Val(888)) where {T, SEED}
    M = size(A, 1)
    if (M % 4 != 0)
        A = pad!(A)
    end
    B = similar(A)
    ws = 🦋generate_random!(copyto!(B, A))
    🦋mul!(copyto!(B, A), ws)
    U, V = materializeUV(B, ws)
    F = RecursiveFactorization.lu!(B, thread)
    out = similar(b, M)

    U, V, F, out
end

const butterfly_workspace = 🦋workspace;

function 🦋mul_level!(A, u, v)
    M, N = size(A)
    @assert M == length(u) && N == length(v)
    Mh = M >>> 1
    Nh = N >>> 1
    @turbo for n in 1 : Nh
        for m in 1 : Mh
            A11 = A[m, n]
            A21 = A[m + Mh, n]
            A12 = A[m, n + Nh]
            A22 = A[m + Mh, n + Nh]

            T1 = A11 + A12
            T2 = A21 + A22
            T3 = A11 - A12
            T4 = A21 - A22
            C11 = T1 + T2
            C21 = T1 - T2
            C12 = T3 + T4
            C22 = T3 - T4

            u1 = u[m]
            u2 = u[m + Mh]
            v1 = v[n]
            v2 = v[n + Nh]

            A[m, n] = u1 * C11 * v1
            A[m + Mh, n] = u2 * C21 * v1
            A[m, n + Nh] = u1 * C12 * v2
            A[m + Mh, n + Nh] = u2 * C22 * v2
        end
    end 
end

function 🦋mul!(A, (uv,))
    M, N = size(A)
    @assert M == N
    Mh = M >>> 1

    U₁ = @view(uv[1:Mh]) 
    V₁ = @view(uv[(Mh + 1):(M)]) 
    U₂ = @view(uv[(1 + M):(M + Mh)]) 
    V₂ = @view(uv[(1 + M + Mh):(2 * M)]) 

    🦋mul_level!(@view(A[1:Mh, 1:Mh]), U₁, V₁) 
    🦋mul_level!(@view(A[Mh + 1:M, 1:Mh]), U₂, V₁) 
    🦋mul_level!(@view(A[1:Mh, Mh + 1:M]), U₁, V₂) 
    🦋mul_level!(@view(A[Mh + 1:M, Mh + 1:M]), U₂, V₂) 

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

function 🦋2!(C, A::Diagonal, B::Diagonal)
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

function 🦋!(A::Matrix, C::SparseBandedMatrix, X::Diagonal, Y::Diagonal)
    @assert size(X) == size(Y)
    if (size(X, 1) + size(Y, 1) != size(A, 1))
        x = size(A, 1) - size(X, 1) - size(Y, 1)
        setdiagonal!(C, [X.diag; rand(x); -Y.diag], true)
        setdiagonal!(C, X.diag, true)
        setdiagonal!(C, Y.diag, false)
    else
        setdiagonal!(C, [X.diag; -Y.diag], true)
        setdiagonal!(C, X.diag, true)
        setdiagonal!(C, Y.diag, false)
    end

    C
end

function 🦋2!(C::SparseBandedMatrix, A::Diagonal, B::Diagonal)
    setdiagonal!(C, [A.diag; -B.diag], true)
    setdiagonal!(C, A.diag, true)
    setdiagonal!(C, B.diag, false)
    C
end

function materializeUV(A, (uv,))
    M, N = size(A)
    Mh = M >>> 1    
    Nh = N >>> 1

    U₁u, U₁l = diagnegbottom(@view(uv[1:Mh])) #Mh
    U₂u, U₂l = diagnegbottom(@view(uv[(1 + Mh + Nh):(M + Nh)])) #M2
    V₁u, V₁l = diagnegbottom(@view(uv[(Mh + 1):(Mh + Nh)])) #Nh
    V₂u, V₂l = diagnegbottom(@view(uv[(1 + 2 * Mh + Nh):(2 * Mh + N)])) #N2
    Uu, Ul = diagnegbottom(@view(uv[(1 + M + N):(2 * M + N)])) #M
    Vu, Vl = diagnegbottom(@view(uv[(1 + 2 * M + N):(2 * M + 2 * N)])) #N

    Bu2 = SparseBandedMatrix{typeof(uv[1])}(undef, M, N)
    
    🦋2!(view(Bu2, 1 : Mh, 1 : Nh), U₁u, U₁l)
    🦋2!(view(Bu2, Mh + 1: M, Nh + 1: N), U₂u, U₂l)

    Bu1 = SparseBandedMatrix{typeof(uv[1])}(undef, M, N)
    🦋!(A, Bu1, Uu, Ul)

    Bv2 = SparseBandedMatrix{typeof(uv[1])}(undef, M, N)

    🦋2!(view(Bv2, 1 : Mh, 1 : Nh), V₁u, V₁l)
    🦋2!(view(Bv2, Mh + 1: M, Nh + 1: N), V₂u, V₂l)

    Bv1 = SparseBandedMatrix{typeof(uv[1])}(undef, M, N)
    🦋!(A, Bv1, Vu, Vl)

    U = (Bu2 * Bu1)'
    V = Bv2 * Bv1
 
    U, V
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
