module RecursiveFactorization
if isdefined(Base, :Experimental) &&
   isdefined(Base.Experimental, Symbol("@max_methods"))
    @eval Base.Experimental.@max_methods 1
end
include("./lu.jl")
include("./butterflylu.jl")
include("./juliac_server.jl")

import PrecompileTools

PrecompileTools.@compile_workload begin
    lu!(rand(2, 2))
end

function __init__()
    _init_juliac_server()
    atexit(_finalize_juliac_server)
end

end # module
