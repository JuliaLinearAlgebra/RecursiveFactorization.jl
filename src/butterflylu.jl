using VectorizedRNG
using LinearAlgebra: Diagonal, I
using LoopVectorization
using RecursiveFactorization

@inline exphalf(x) = exp(x) * oftype(x, 0.5)
function 🦋!(wv, ::Val{SEED} = Val(888)) where {SEED}
    T = eltype(wv)
    mrng = VectorizedRNG.MutableXoshift(SEED)
    GC.@preserve mrng begin rand!(exphalf, VectorizedRNG.Xoshift(mrng), wv, static(0),
                                  T(-0.05), T(0.1)) end
end

function 🦋generate!(A, ::Val{SEED} = Val(888)) where {SEED}
    Usz = 2 * size(A, 1)
    Vsz = 2 * size(A, 2)
    uv = similar(A, Usz + Vsz)
    🦋!(uv, Val(SEED))
    (uv,)
end

function 🦋workspace(A, ::Val{SEED} = Val(888)) where {SEED}
    A = pad!(A)
    B = similar(A);
    ws = 🦋generate!(B)
    🦋mul!(copyto!(B, A), ws)
    U, V, B = materializeUV(B, ws)
    F = RecursiveFactorization.lu!(B, Val(false))
    A, B, U, V, F
end

const butterfly_workspace = 🦋workspace;

function 🦋mul_level!(A, u, v)
    # for now, assume... 
    M, N = size(A)
    Ml = M >>> 1
    Nl = N >>> 1
    Mh = M - Ml
    Nh = N - Nl
    @turbo for n in 1:Nl
        for m in 1:Ml
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
    Mh = M >>> 1
    Nh = N >>> 1
    U₁ = @view(uv[1:Mh])
    U₂ = @view(uv[(1 + Mh + Nh):(2 * Mh + Nh)])
    V₁ = @view(uv[(Mh + 1):(Mh + Nh)])
    V₂ = @view(uv[(1 + 2 * Mh + Nh):(2 * Mh + 2 * Nh)])
    🦋mul_level!(@view(A[1:Mh, 1:Nh]), U₁, V₁)
    🦋mul_level!(@view(A[(1 + Mh):M, 1:Nh]), U₂, V₁)
    🦋mul_level!(@view(A[1:Mh, (1 + Nh):N]), U₁, V₂)
    🦋mul_level!(@view(A[(1 + Mh):M, (1 + Nh):N]), U₂, V₂)
    U = @view(uv[(1 + 2 * Mh + 2 * Nh):(2 * Mh + 2 * Nh + M)])
    V = @view(uv[(1 + 2 * Mh + 2 * Nh + M):(2 * Mh + 2 * Nh + M + N)])
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
🦋(A, B) = [A B
           A -B]

function materializeUV(A, (uv,))
    M, N = size(A)
    Mh = M >>> 1
    Nh = N >>> 1

    U₁u, U₁l = diagnegbottom(@view(uv[1:Mh]))
    U₂u, U₂l = diagnegbottom(@view(uv[(1 + Mh + Nh):(2 * Mh + Nh)]))
    V₁u, V₁l = diagnegbottom(@view(uv[(Mh + 1):(Mh + Nh)]))
    V₂u, V₂l = diagnegbottom(@view(uv[(1 + 2 * Mh + Nh):(2 * Mh + 2 * Nh)]))
    Uu, Ul = diagnegbottom(@view(uv[(1 + 2 * Mh + 2 * Nh):(2 * Mh + 2 * Nh + M)]))
    Vu, Vl = diagnegbottom(@view(uv[(1 + 2 * Mh + 2 * Nh + M):(2 * Mh + 2 * Nh + M + N)]))

    Bu2 = [🦋(U₁u, U₁l) 0*I
           0*I 🦋(U₂u, U₂l)]
    Bu1 = 🦋(Uu, Ul)

    Bv2 = [🦋(V₁u, V₁l) 0*I
           0*I 🦋(V₂u, V₂l)]
    Bv1 = 🦋(Vu, Vl)

    (Bu2 * Bu1)', Bv2 * Bv1, A
end

function pad!(A)
    M, N = size(A)
    xn = 4 - M % 4
    A = [A   zeros(N, xn)
    zeros(xn, N) I(xn)
    ]
    A
end