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
rec_mflops = Float64[]
ref_mflops = Float64[]
ns = 4:8:500
for n in ns
    @info "$n × $n"
    global A = rand(n, n)
    bt = @belapsed LinearAlgebra.lu!(B) setup=(B = copy(A))
    push!(bas_mflops, luflop(n)/bt/1e9)

    rt = @belapsed RecursiveFactorization.lu!(B) setup=(B = copy(A))
    push!(rec_mflops, luflop(n)/rt/1e9)

    ref = @belapsed LinearAlgebra.generic_lufact!(B) setup=(B = copy(A))
    push!(ref_mflops, luflop(n)/ref/1e9)
end

using DataFrames, VegaLite
df = DataFrame(Size = ns, RecursiveFactorization = rec_mflops, OpenBLAS = bas_mflops, Reference = ref_mflops)
df = stack(df, [:RecursiveFactorization, :OpenBLAS, :Reference], variable_name = :Library, value_name = :GFLOPS)
plt = df |> @vlplot(
        :line, color = :Library,
        x = {:Size}, y = {:GFLOPS},
        width = 2400, height = 600
    )
save(joinpath(homedir(), "Pictures", "lu_float64.png"), plt)

#=
using Plot
plt = plot(ns, bas_mflops, legend=:bottomright, lab="OpenBLAS", title="LU Factorization Benchmark", marker=:auto, dpi=300)
plot!(plt, ns, rec_mflops, lab="RecursiveFactorization", marker=:auto)
plot!(plt, ns, ref_mflops, lab="Reference", marker=:auto)
xaxis!(plt, "size (N x N)")
yaxis!(plt, "GFLOPS")
savefig("lubench.png")
savefig("lubench.pdf")
=#
