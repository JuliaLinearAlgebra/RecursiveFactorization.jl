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
    @test norm(MF.L*MF.U - A[MF.p, :], Inf) < sqrt(eps(real(first(A))))
    nothing
end

@testset "Test LU factorization" begin
    for p in (Val(true), Val(false)), T in (Float64, Float32, ComplexF64, ComplexF32, Real)
        siz = (50, 100)
        if isconcretetype(T)
            A = rand(T, siz...)
        else
            _A = rand(50, 100)
            A  = Matrix{T}(undef, siz...)
            copyto!(A, _A)
        end
        MF = mylu(A, p)
        BF = baselu(A, p)
        testlu(A, MF, BF)
        for i in 50:7:100 # test `MF.info`
            A[:, i] .= 0
            MF = mylu(A, p, check=false)
            BF = baselu(A, p, check=false)
            testlu(A, MF, BF)
        end
    end
end
