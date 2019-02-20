using Test
import RecursiveFactorization
import LinearAlgebra

baselu = LinearAlgebra.lu
mylu = RecursiveFactorization.lu

function testlu(A, MF, BF)
    @test MF.info == BF.info
    @test MF.L*MF.U ≈ A[MF.p, :]
    nothing
end

@testset "Test LU factorization" begin
    for p in (Val(true), Val(false)), T in (Float64, Float32, ComplexF64, ComplexF32)
        A = rand(T, 50, 100)
        MF = mylu(A, p)
        BF = baselu(A, p)
        testlu(A, MF, BF)
        for i in 1:100
            A[:, i] .= 0
            MF = mylu(A, p, check=false)
            BF = baselu(A, p, check=false)
            testlu(A, MF, BF)
        end
    end
end
