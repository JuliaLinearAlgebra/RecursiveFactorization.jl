using Test
import RecursiveFactorization
import LinearAlgebra
using LinearAlgebra: norm, Adjoint
using Random

Random.seed!(12)

const baselu = LinearAlgebra.lu
const mylu = RecursiveFactorization.lu

function testlu(A, MF, BF)
    @test MF.info == BF.info
    @test norm(MF.L * MF.U - A[MF.p, :], Inf) < 200sqrt(eps(real(one(float(first(A))))))
    nothing
end
testlu(A::Adjoint, MF::Adjoint, BF) = testlu(parent(A), parent(MF), BF)

@testset "Test LU factorization" begin for _p in (true, false),
                                           T in (Float64, Float32, ComplexF64, ComplexF32,
                                                 Real)

    p = Val(_p)
    for (i, s) in enumerate([1:10; 50:80:200; 300])
        iseven(i) && (p = RecursiveFactorization.to_stdlib_pivot(p))
        siz = (s, s + 2)
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
        testlu(A, MF, BF)
        testlu(A, mylu(A, p, Val(false)), BF)
        A′ = permutedims(A)
        MF′ = mylu(A′', p)
        testlu(A′', MF′, BF)
        testlu(A′', mylu(A′', p, Val(false)), BF)
        i = rand(1:s) # test `MF.info`
        A[:, i] .= 0
        MF = mylu(A, p, check = false)
        BF = baselu(A, p, check = false)
        testlu(A, MF, BF)
        testlu(A, mylu(A, p, Val(false), check = false), BF)
    end
end end

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
    A800 = wilkinson(800);
    B800 = similar(A800);
    ws800 = RecursiveFactorization.🦋workspace(B800)
    RecursiveFactorization.🦋mul!(copyto!(B800, A800), ws800)
    U800, V800 = RecursiveFactorization.materializeUV(B800, ws800)
    F800 = RecursiveFactorization.lu!(B800, Val(false))

    b = rand(800)
    x = V800 * (F800 \ (U800 * b))
    @test norm(A800 * x .- b) <= 1e-12
end
