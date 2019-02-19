using BenchmarkTools
import LinearAlgebra, RecursiveFactorization

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.5

luflop(m, n) = n^3÷3 - n÷3 + m*n^2
luflop(n) = luflop(n, n)

bas_mflops = Float64[]
rec_mflops = Float64[]
ns = 50:50:800
for n in ns
    A = rand(n, n)
    bt = @belapsed LinearAlgebra.lu!($(copy(A)))
    rt = @belapsed RecursiveFactorization.lu!($(copy(A)))
    push!(bas_mflops, luflop(n)/bt/1e9)
    push!(rec_mflops, luflop(n)/rt/1e9)
end

using Plots
plt = plot(ns, bas_mflops, legend=:bottomright, lab="OpenBLAS", title="LU Factorization Benchmark", marker=:auto, dpi=150)
plot!(plt, ns, rec_mflops, lab="RecursiveFactorization", marker=:auto)
xaxis!(plt, "size (N x N)")
yaxis!(plt, "GFLOPS")
savefig("lubench.png")
savefig("lubench.pdf")
