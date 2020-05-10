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
