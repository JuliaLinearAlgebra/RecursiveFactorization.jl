using BenchmarkTools
import LinearAlgebra, RecursiveFactorization

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.08

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
rec8_mflops = Float64[]
rec16_mflops = Float64[]
rec32_mflops = Float64[]
ref_mflops = Float64[]
ns = 4:32:500
for n in ns
    @info "$n × $n"
    A = rand(n, n)
    bt = @belapsed LinearAlgebra.lu!($(copy(A)))
    push!(bas_mflops, luflop(n)/bt/1e9)

    rt8 = @belapsed RecursiveFactorization.lu!($(copy(A)); blocksize=8)
    push!(rec8_mflops, luflop(n)/rt8/1e9)

    rt16 = @belapsed RecursiveFactorization.lu!($(copy(A)); blocksize=16)
    push!(rec16_mflops, luflop(n)/rt16/1e9)

    rt32 = @belapsed RecursiveFactorization.lu!($(copy(A)); blocksize=32)
    push!(rec32_mflops, luflop(n)/rt32/1e9)

    ref = @belapsed LinearAlgebra.generic_lufact!($(copy(A)))
    push!(ref_mflops, luflop(n)/ref/1e9)
end

using Plots
plt = plot(ns, bas_mflops, legend=:bottomright, lab="OpenBLAS", title="LU Factorization Benchmark", marker=:auto, dpi=150)
plot!(plt, ns, rec8_mflops, lab="RF8", marker=:auto)
plot!(plt, ns, rec16_mflops, lab="RF16", marker=:auto)
plot!(plt, ns, rec32_mflops, lab="RF32", marker=:auto)
plot!(plt, ns, ref_mflops, lab="Reference", marker=:auto)
xaxis!(plt, "size (N x N)")
yaxis!(plt, "GFLOPS")
savefig("lubench.png")
savefig("lubench.pdf")
