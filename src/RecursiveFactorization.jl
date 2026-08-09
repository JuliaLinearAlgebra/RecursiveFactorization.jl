module RecursiveFactorization
if isdefined(Base, :Experimental) &&
   isdefined(Base.Experimental, Symbol("@max_methods"))
    @eval Base.Experimental.@max_methods 1
end
include("./lu.jl")
include("./butterflylu.jl")

import PrecompileTools

# Inference is whole-body, so n = 4 already reaches the recursive kernel and
# the TriangularSolve legs that only run above the blocksize threshold. Only
# type changes need their own case: eltype, the two `Val`s, and a matrix RHS.
PrecompileTools.@setup_workload begin
    n = 4
    for T in (Float64, Float32)
        A = Matrix{T}(undef, n, n)
        A .= rand.(T)
        @view(A[LinearAlgebra.diagind(A)]) .+= T(n)
        b = Vector{T}(undef, n)
        b .= rand.(T)
        B = Matrix{T}(undef, n, 2)
        B .= rand.(T)
        PrecompileTools.@compile_workload begin
            lu(A)
            lu!(copy(A))
            lu!(copy(A), Val(true), Val(true))
            lu!(copy(A), Val(false), Val(true))
            F = lu!(copy(A), Val(false))
            LinearAlgebra.ldiv!(F, copy(b))
            LinearAlgebra.ldiv!(F, copy(B))
            LinearAlgebra.ldiv!(lu(A), copy(b))
            🦋solve!(🦋workspace(copy(A), copy(b)), Val(false))
        end
    end
end

end # module
