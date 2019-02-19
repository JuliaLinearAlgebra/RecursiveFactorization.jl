using Test
import RecursiveFactorization
import LinearAlgebra

baselu = LinearAlgebra.lu
mylu = RecursiveFactorization.lu

function testlu(MF, BF)
    @test MF.info == BF.info
    @test MF.L ≈ BF.L
    @test MF.U ≈ BF.U
    @test MF.P ≈ BF.P
    nothing
end

@testset "Test LU factorization" begin
    for p in (Val(true), Val(false))
        A = rand(100, 100)
        MF = mylu(A, p)
        BF = baselu(A, p)
        testlu(MF, BF)
        for i in 1:100
            A[:, i] .= 0
            MF = mylu(A, p, check=false)
            BF = baselu(A, p, check=false)
            testlu(MF, BF)
        end
    end
end
