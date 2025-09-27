module RecursiveFactorization
if isdefined(Base, :Experimental) &&
   isdefined(Base.Experimental, Symbol("@max_methods"))
    @eval Base.Experimental.@max_methods 1
end
include("./lu.jl")
include("./butterflylu.jl")

import PrecompileTools

PrecompileTools.@compile_workload begin
    lu!(rand(2, 2))
end

end # module
