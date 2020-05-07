using BenchmarkTools
using LinearAlgebra, RecursiveFactorization

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.5

function luflop(m, n=m; innerflop=2)
    sum(1:min(m, n)) do k
        invflop = 1
        scaleflop = isempty(k+1:m) ? 0 : sum(k+1:m)
        updateflop = isempty(k+1:n) ? 0 : sum(k+1:n) do j
            isempty(k+1:m) ? 0 : sum(k+1:m) do i
                innerflop
            end
        end
        invflop + scaleflop + updateflop
    end
end

bas_mflops = Float64[]
rec72_mflops = Float64[]
rec80_mflops = Float64[]
rec192_mflops = Float64[]
nb_mflops = Float64[]
ref_mflops = Float64[]
ns = 4:64:600
for n in ns
    @info "$n × $n"
    A = rand(n, n)
    bt = @belapsed LinearAlgebra.lu!(B) setup=(B = copy(A))
    push!(bas_mflops, luflop(n)/bt/1e9)

    rt72 = @belapsed RecursiveFactorization.lu!(B; threshold=72) setup=(B = copy(A))
    push!(rec72_mflops, luflop(n)/rt72/1e9)

    rt80 = @belapsed RecursiveFactorization.lu!(B; threshold=80) setup=(B = copy(A))
    push!(rec80_mflops, luflop(n)/rt80/1e9)

    rt192 = @belapsed RecursiveFactorization.lu!(B; threshold=192) setup=(B = copy(A))
    push!(rec192_mflops, luflop(n)/rt192/1e9)

    nb = @belapsed RecursiveFactorization._generic_lufact!(B, Val(true), Vector{Int}(undef, $n), Ref(0)) setup=(B = copy(A))
    push!(nb_mflops, luflop(n)/nb/1e9)

    ref = @belapsed LinearAlgebra.generic_lufact!(B) setup=(B = copy(A))
    push!(ref_mflops, luflop(n)/ref/1e9)
end

using Plots
plt = plot(ns, bas_mflops, legend=:bottomright, lab="OpenBLAS", title="LU Factorization Benchmark", marker=:auto, dpi=150)
plot!(plt, ns, rec72_mflops, lab="RF72", marker=:auto)
plot!(plt, ns, rec80_mflops, lab="RF80", marker=:auto)
plot!(plt, ns, rec192_mflops, lab="RF192", marker=:auto)
plot!(plt, ns, nb_mflops, lab="No recursion", marker=:auto)
plot!(plt, ns, ref_mflops, lab="Reference", marker=:auto)
xaxis!(plt, "size (N x N)")
yaxis!(plt, "GFLOPS")
savefig("lubench.png")
savefig("lubench.pdf")
