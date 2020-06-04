using  Test
import RecursiveFactorization
import LinearAlgebra
using  LinearAlgebra: norm
using  Random

Random.seed!(12)

const baselu = LinearAlgebra.lu
const mylu = RecursiveFactorization.lu

function testlu(A, MF, BF)
    @test MF.info == BF.info
    @test norm(MF.L*MF.U - A[MF.p, :], Inf) < 100sqrt(eps(real(one(float(first(A))))))
    nothing
end

@testset "Test LU factorization" begin
    for _p in (true, false), T in (Float64, Float32, ComplexF64, ComplexF32, Real)
        p = Val(_p)
        for s in [1:10; 50:80:200; 300]
            siz = (s, s+2)
            @info("size: $(siz[1]) × $(siz[2]), T = $T, p = $_p")
            if isconcretetype(T)
                A = rand(T, siz...)
            else
                _A = rand(siz...)
                A  = Matrix{T}(undef, siz...)
                copyto!(A, _A)
            end
            MF = mylu(A, p)
            BF = baselu(A, p)
            testlu(A, MF, BF)
            i = rand(1:s) # test `MF.info`
            A[:, i] .= 0
            MF = mylu(A, p, check=false)
            BF = baselu(A, p, check=false)
            testlu(A, MF, BF)
        end
    end
end

@testset "Test ldiv!" begin
    for T in (Float64, Float32, ComplexF64, ComplexF32, Real), s in 1:50
        @info("size: $s × $s, T = $T")
        if isconcretetype(T)
            A = randn(T, s, s)
            b = randn(T, s)
        else
            _A = randn(s, s)
            _b = randn(s)
            A  = similar(_A, T, s, s)
            copyto!(A, _A)
            b = similar(_b, T, s)
            copyto!(b, _b)
        end
        # just use LinearAlgebra.lu is fine, because we are testing `ldiv!`
        F = lu(A)
        _b = copy(b)
        ref = ldiv!(F, copy(b))
        # test alias
        @test _b === RecursiveFactorization.ldiv!(F, _b)
        # test precision
        @test _b ≈ ref
        _x = similar(b)
        copyto!(_b, b)
        # test alias
        @test _x === RecursiveFactorization.ldiv!(_x, F, _b)
        @test _b == b
        # test precision
        @test _x ≈ ref
    end
end
