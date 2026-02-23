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
        ws = RecursiveFactorization.🦋workspace(copy(A), copy(b))
        out = RecursiveFactorization.🦋solve!(ws, Val(true))
        @test norm(A * out .- b) <= 1e-10
    end
end

@testset "Juliac server" begin
    if RecursiveFactorization._server_available()
        @info "Juliac server is available, running binary path tests"

        # Test Float64 square
        for s in (10, 50, 100, 300)
            A = rand(s, s)
            E = 20s * eps(Float64)
            MF = RecursiveFactorization.juliac_lu(A)
            BF = baselu(A)
            @test MF.info == BF.info
            @test norm(MF.L * MF.U - A[MF.p, :], Inf) < E
        end

        # Test Float32 square
        for s in (10, 50, 100)
            A = rand(Float32, s, s)
            E = 20s * eps(Float32)
            MF = RecursiveFactorization.juliac_lu(A)
            BF = baselu(A)
            @test MF.info == BF.info
            @test norm(Float64.(MF.L * MF.U) - Float64.(A[MF.p, :]), Inf) < E
        end

        # Test rectangular (tall and wide)
        for (m, n) in ((200, 100), (100, 200))
            A = rand(m, n)
            E = 20m * eps(Float64)
            MF = RecursiveFactorization.juliac_lu(A)
            BF = baselu(A)
            @test MF.info == BF.info
            @test norm(MF.L * MF.U - A[MF.p, :], Inf) < E
        end

        # Test juliac_lu! (mutating)
        A = rand(100, 100)
        A_orig = copy(A)
        MF = RecursiveFactorization.juliac_lu!(A)
        @test MF.info == 0
        @test norm(MF.L * MF.U - A_orig[MF.p, :], Inf) < 20 * 100 * eps(Float64)

        # Test singular matrix
        A = rand(50, 50)
        A[:, 25] .= 0
        MF = RecursiveFactorization.juliac_lu(A, check = false)
        BF = baselu(A, check = false)
        @test MF.info == BF.info

        # Test consistency with pure-Julia path
        Random.seed!(42)
        A1 = rand(100, 100)
        A2 = copy(A1)
        F_juliac = RecursiveFactorization.juliac_lu!(A1)
        F_julia = RecursiveFactorization.lu!(A2)
        @test F_juliac.info == F_julia.info
        @test F_juliac.factors ≈ F_julia.factors
        @test F_juliac.ipiv == F_julia.ipiv
    else
        @info "Juliac server not available, skipping binary path tests"
    end
end

