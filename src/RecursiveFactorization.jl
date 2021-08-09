module RecursiveFactorization

include("./lu.jl")

let
    while true
        lu!(rand(2,2))
        break
    end
end

end # module
