using Test
import RecursiveFactorization
import LinearAlgebra
import TriangularSolve
using LinearAlgebra: norm, Adjoint, Transpose, ldiv!, UnitLowerTriangular,
                     UpperTriangular
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

@testset "NoPivot lu! with a user-supplied ipiv leaves valid pivots" begin
    # Stdlib consumers (`LAPACK.getrs!`, `_ipiv_rows!`) read `F.ipiv`; it must
    # hold the identity, not undefined memory (which crashed in `dlaswp`).
    for T in (Float64, Float32, ComplexF64)
        n = 30
        A = rand(T, n, n) + T(10) * LinearAlgebra.I
        b = rand(T, n)
        ipiv = Vector{LinearAlgebra.BlasInt}(undef, n)
        fill!(ipiv, typemax(LinearAlgebra.BlasInt) - 7) # poison undefined memory
        F = RecursiveFactorization.lu!(copy(A), ipiv, Val(false), Val(false))
        @test F.ipiv == 1:n
        x = LinearAlgebra.ldiv!(F, copy(b)) # stdlib LU path consumes F.ipiv
        @test norm(A * x - b) < 1000 * n * eps(real(T))
    end
end

@testset "NotIPIV backsolves stay on TriangularSolve's native kernels" begin
    # The signatures `ldiv!(::LU{..., <:NotIPIV}, ...)` hands to
    # TriangularSolve.ldiv! must keep resolving to native kernel methods, never
    # to a catch-all that forwards to LinearAlgebra (= BLAS for BLAS types).
    catchall2 = which(TriangularSolve.ldiv!, Tuple{Any, Any})
    catchall3 = which(TriangularSolve.ldiv!, Tuple{Any, Any, Val{true}})
    for T in (Float64, Float32)
        MT = Matrix{T}
        F = RecursiveFactorization.lu(rand(T, 40, 40) + T(10) * LinearAlgebra.I,
            Val(false))
        @test F.ipiv isa RecursiveFactorization.NotIPIV
        for BT in (Vector{T}, MT)
            m = which(LinearAlgebra.ldiv!, Tuple{typeof(F), BT})
            @test m.module === RecursiveFactorization
        end
        # vector and matrix right-hand sides both resolve to native kernels
        # (the vector methods require TriangularSolve >= 0.2.5)
        for W in (UnitLowerTriangular{T, MT}, UpperTriangular{T, MT}),
            BT in (Vector{T}, MT)

            m2 = which(TriangularSolve.ldiv!, Tuple{W, BT})
            m3 = which(TriangularSolve.ldiv!, Tuple{W, BT, Val{true}})
            @test m2 !== catchall2
            @test m3 !== catchall3
            @test m2.module === TriangularSolve
            @test m3.module === TriangularSolve
        end
    end
end

@testset "NotIPIV ldiv! correctness across small-to-large sizes" begin
    for T in (Float64, Float32), n in (8, 64, 200, 300)
        A = rand(T, n, n) + T(10) * LinearAlgebra.I
        b = rand(T, n)
        B = rand(T, n, 3)
        F = RecursiveFactorization.lu(A, Val(false))
        x = ldiv!(F, copy(b))
        @test x isa Vector{T}
        @test norm(A * x - b) < 1000 * n * eps(T)
        X = ldiv!(F, copy(B))
        @test norm(A * X - B) < 1000 * n * eps(T)
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
    # Seeded so earlier testsets can't shift which b's this draws.  The bound is
    # a relative residual: the old absolute 1e-10 sat inside the backward-stable
    # rounding band for n≈800 (c·n·eps·‖A‖·‖x‖ ≈ 1e-9) and failed spuriously on
    # CI hardware — the butterfly transforms themselves are hardware-dependent
    # (VectorizedRNG streams vary with SIMD width), so residuals near 1e-9
    # absolute are legitimate rounding, not breakage (measured: ≤2.3e-11 over
    # 550 draws on Zen 2, 7.6e-10 once on a GitHub runner; genuine
    # routing/pivot breakage produces ≥1e-5).
    Random.seed!(1234)
    for i in 790 : 810
        A = wilkinson(i)
        b = rand(i)
        ws = RecursiveFactorization.🦋workspace(copy(A), copy(b))
        out = RecursiveFactorization.🦋solve!(ws, Val(true))
        @test norm(A * out .- b) <= 1e-8 * norm(b)
    end
end


@testset "Factorization panel solves stay on TriangularSolve" begin
    catchall2 = which(TriangularSolve.ldiv!, Tuple{Any, Any})
    catchall3 = which(TriangularSolve.ldiv!, Tuple{Any, Any, Val{true}})
    # (a) The exact panel types the recursive lu! constructs for Float32/Float64
    # (PtrArray-wrapped views) must resolve to native TriangularSolve kernel
    # methods, threaded and not.
    for T in (Float64, Float32)
        A = rand(T, 100, 100)
        B = view(RecursiveFactorization.PtrArray(A), axes(A)...)
        n1 = RecursiveFactorization.nsplit(T, 100)
        A11 = RecursiveFactorization.square_view(B, n1)
        A12 = @view B[1:n1, (n1 + 1):100]
        for V in (Val{false}, Val{true})
            m = which(TriangularSolve.ldiv!,
                Tuple{UnitLowerTriangular{T, typeof(A11)}, typeof(A12), V})
            @test m !== catchall3
            @test m.module === TriangularSolve
        end
    end
    # (b) Whole-suite sweep: every factorization and backsolve above has already
    # run.  A TriangularSolve catch-all specialization whose triangular argument
    # has Float32/Float64 eltype would mean some real-eltype solve silently fell
    # back to LinearAlgebra/BLAS; none may exist.  Complex specializations are
    # expected — TriangularSolve has no complex kernels, see (c).
    flagged = Union{UpperTriangular{Float32}, UpperTriangular{Float64},
        UnitLowerTriangular{Float32}, UnitLowerTriangular{Float64}}
    for ca in (catchall2, catchall3), mi in Base.specializations(ca)
        sig = Base.unwrap_unionall(mi.specTypes)
        TA = sig.parameters[2]
        @test !(TA isa Type && TA <: flagged)
    end
    # (c) Characterization of the known gap (a finding, not policy): complex
    # panel solves resolve to the catch-all, i.e. LinearAlgebra -> LAPACK
    # trtrs!/trsm.  If TriangularSolve gains native complex kernels these flip
    # and the routing table should be updated.
    for T in (ComplexF64, ComplexF32)
        A = rand(T, 100, 100)
        n1 = RecursiveFactorization.nsplit(T, 100)
        A11 = RecursiveFactorization.square_view(A, n1)
        A12 = @view A[1:n1, (n1 + 1):100]
        m = which(TriangularSolve.ldiv!,
            Tuple{UnitLowerTriangular{T, typeof(A11)}, typeof(A12), Val{false}})
        @test m === catchall3
    end
end
