module RecursiveFactorization

include("./lu.jl")

import SnoopPrecompile

SnoopPrecompile.@precompile_all_calls begin lu!(rand(2, 2)) end

end # module
