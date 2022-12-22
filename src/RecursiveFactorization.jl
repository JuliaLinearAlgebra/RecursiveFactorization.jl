module RecursiveFactorization

include("lu.jl")
include("butterflies.jl")

import SnoopPrecompile

SnoopPrecompile.@precompile_all_calls begin lu!([1.0 0.0; 0.0 1.0]) end

end # module
