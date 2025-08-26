using Test
import RecursiveFactorization
import LinearAlgebra
using LinearAlgebra: norm, Adjoint, Transpose, ldiv!
using Random

Random.seed!(12)

const baselu = LinearAlgebra.lu
const mylu = RecursiveFactorization.lu

function testlu(A, MF, BF, p)
    @test MF.info == BF.info
    if !iszero(MF.info)
        return nothing
    end
    E = 20size(A, 1) * eps(real(one(float(first(A)))))
    @test norm(MF.L * MF.U - A[MF.p, :], Inf) < (p ? E : 10sqrt(E))
    if ==(size(A)...)
        b = ldiv!(MF, A[:, end])
        if all(isfinite, b)
            n = size(A, 2)
            rhs = [i == n for i in 1:n]
            @test b≈rhs atol=p ? 100E : 100sqrt(E)
        end
    end
    nothing
end
testlu(A::Union{Transpose, Adjoint}, MF, BF, p) = testlu(parent(A), parent(MF), BF, p)

@testset "Test LU factorization" begin
    for _p in (true, false),
        T in (Float64, Float32, ComplexF64, ComplexF32,
            Real)

        p = Val(_p)
        for (i, s) in enumerate([1:10; 50:80:200; 300])
            iseven(i) && (p = RecursiveFactorization.to_stdlib_pivot(p))
            for m in (s, s + 2)
                siz = (s, m)
                @info("size: $(siz[1]) × $(siz[2]), T = $T, p = $_p")
                if isconcretetype(T)
                    A = rand(T, siz...)
                else
                    _A = rand(siz...)
                    A = Matrix{T}(undef, siz...)
                    copyto!(A, _A)
                end
                MF = mylu(A, p)
                BF = baselu(A, p)
                testlu(A, MF, BF, _p)
                testlu(A, mylu(A, p, Val(true)), BF, false)
                A′ = permutedims(A)
                MF′ = mylu(A′', p)
                testlu(A′', MF′, BF, _p)
                testlu(A′', mylu(A′', p, Val(true)), BF, false)
                i = rand(1:s) # test `MF.info`
                A[:, i] .= 0
                MF = mylu(A, p, check = false)
                BF = baselu(A, p, check = false)
                testlu(A, MF, BF, _p)
                testlu(A, mylu(A, p, Val(true), check = false), BF, false)
            end
        end
    end
end

function wilkinson(N)
    A = zeros(N, N)
    A[1:(N+1):N*N] .= 1
    A[:, end] .= 1
    for n in 1:(N - 1)
        for r in (n + 1):N
            @inbounds A[r, n] = -1
        end
    end
    A
end

@testset "🦋" begin
    for i in 790 : 810
        A = wilkinson(i)
        b = rand(i)
        A_ext, U, V, F = RecursiveFactorization.🦋workspace(A)
        M, N = size(A)
        xn = 4 - M % 4
        b_ext = [b; rand(xn)]
        x = V * (F \ (U * b_ext))    
        @test norm(A * x[1:M] .- b) <= 1e-10
    end
end

i = 800
A = wilkinson(i)
b = rand(i)
U, V, F = RecursiveFactorization.🦋workspace(A)
x = V * (F \ (U * b))    
@test norm(A * x .- b) <= 1e-10

#=

i = 8
A = rand(i, i)
b = rand(i)
A_ext, U, V, F = RecursiveFactorization.🦋workspace(A)
tmp = U * b
F \ tmp
(F.L * F.U) \ tmp
#tmp * F.U ^ -1 * F.L^-1
F.U \ (tmp / F.L)
isapprox(F \ tmp, F.U \ (F.L \ tmp))

x = V * (F \ (U * b))    

x = V * (TriangularSolve.ldiv!(UpperTriangular(F.U), TriangularSolve.ldiv!(LowerTriangular(F.L), U * b)))

norm(V * (TriangularSolve.ldiv!(UpperTriangular(F.U), TriangularSolve.ldiv!(LowerTriangular(F.L), U * b)))- V*(F\(U*b)))
# goal: solve F.L * F.U * x = b
=#