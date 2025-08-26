using LinearAlgebra, .Threads

struct SparseBandedMatrix{T} <: AbstractMatrix{T}
    size :: Tuple{Int, Int}
    indices :: Vector{Int}
    diags :: Vector{Vector{T}}
    function SparseBandedMatrix{T}(::UndefInitializer, N, M) where T
        size = (N, M)
        indices = Int[]
        diags = Vector{T}[]
        new(size, indices, diags)
    end
    function SparseBandedMatrix{T}(ind_vals, diag_vals, N, M) where T
        size = (N, M)
        perm = sortperm(ind_vals)
        indices = ind_vals[perm]
        for i in 1 : length(indices) - 1
            @assert indices[i] != indices[i + 1]
        end
        diags = diag_vals[perm]
        new(size, indices, diags)
    end
end

function Base.size(M :: SparseBandedMatrix) 
    M.size
end

function Base.getindex(M :: SparseBandedMatrix{T}, i :: Int, j :: Int, I :: Int...) where T
    @boundscheck checkbounds(M, i, j, I...)
    rows, cols = size(M)
    wanted_ind = rows - i + j
    ind = searchsortedfirst(M.indices, wanted_ind)
    if (ind <= length(M.indices) && M.indices[ind] == wanted_ind)
        if (i > j)
            return M.diags[ind][j]
        else
            return M.diags[ind][i]
        end
    end
    zero(T)
end

function Base.setindex!(M :: SparseBandedMatrix{T}, val, i :: Int, j :: Int, I :: Int...) where T 
    @boundscheck checkbounds(M, i, j, I...) 
    rows = size(M, 1)
    wanted_ind = rows - i + j
    ind = searchsortedfirst(M.indices, wanted_ind)
    if (ind > length(M.indices) || M.indices[ind] != wanted_ind)
        insert!(M.indices, ind, wanted_ind)
        insert!(M.diags, ind, zeros(T, rows - abs(wanted_ind - rows)))
    end
    if (i > j)
        M.diags[ind][j] = val isa T ? val : convert(T, val)::T
    else
        M.diags[ind][i] = val isa T ? val : convert(T, val)::T
    end
    val
 end

 function setdiagonal!(M :: SparseBandedMatrix{T}, diagvals, lower :: Bool) where T
    rows, cols = size(M) 
    if length(diagvals) > rows
        error("size of diagonal is too big for the matrix")
    end
    if lower
        wanted_ind = length(diagvals)
    else
        wanted_ind = 2 * rows - length(diagvals)
    end

    ind = searchsortedfirst(M.indices, wanted_ind)
    if (ind > length(M.indices) || M.indices[ind] != wanted_ind)
        insert!(M.indices, ind, wanted_ind)
        insert!(M.diags, ind, diagvals isa Vector{T} ? diagvals : convert(Vector{T}, diagvals)::Vector{T}) 
    else
        for i in 1 : eachindex(diagvals)
            M.diags[ind][i] = diagvals[i] isa T ? diagvals[i] : convert(T, diagvals[i])::T
        end
    end
    diagvals
end

# C = Cb + aAB
function LinearAlgebra.mul!(C :: Matrix{T}, A:: SparseBandedMatrix{T}, B :: Matrix{T}, a :: Number, b :: Number) where T
    @assert size(A, 2) == size(B, 1)
    @assert size(A, 1) == size(C, 1)
    @assert size(B, 2) == size(C, 2)
    C.*=b

    rows, cols = size(A)
    @inbounds for (ind, location) in enumerate(A.indices)
        @threads for i in 1:length(A.diags[ind])
            # value: diag[i]
            # index in array: 
            #       if ind < rows(A), then index = (rows - loc + i, i)
            #       else index = (i, loc - cols + i)
            val = A.diags[ind][i] * a
            if location < rows 
                index_i = rows - location + i 
                index_j = i 
            else
                index_i = i 
                index_j = location - cols + i 
            end
            #A[index_i, index_j] * B[index_j, j] = C[index_i, j]
            for j in 1 : size(B, 2)
                C[index_i, j] = fma(val, B[index_j, j], C[index_i, j])
            end
        end
    end
    C
end

# C = Cb + aBA
function LinearAlgebra.mul!(C :: Matrix{T}, A:: Matrix{T}, B :: SparseBandedMatrix{T}, a :: Number, b :: Number) where T
    @assert size(A, 2) == size(B, 1)
    @assert size(A, 1) == size(C, 1)
    @assert size(B, 2) == size(C, 2)

    C.*=b

    rows, cols = size(B)
    @inbounds for (ind, location) in enumerate(B.indices)
        @threads for i in eachindex(B.diags[ind])
                val = B.diags[ind][i] * a
                if location < rows 
                    index_i = rows - location + i 
                    index_j = i 
                else
                    index_i = i 
                    index_j = location - cols + i 
                end
            @simd for j in 1 : size(A, 1)
                C[j, index_j] = fma(val, A[j, index_i], C[j, index_j])
            end
        end
    end
    C
end

function LinearAlgebra.mul!(C :: SparseBandedMatrix{T}, A:: SparseBandedMatrix{T}, B :: SparseBandedMatrix{T}, a :: Number, b :: Number) where T
    @assert size(A, 2) == size(B, 1)
    @assert size(A, 1) == size(C, 1)
    @assert size(B, 2) == size(C, 2)

    C.*=b

    rows_a, cols_a = size(A)
    rows_b, cols_b = size(B)
    @inbounds for (ind_a, location_a) in enumerate(A.indices)
        @threads for i in eachindex(A.diags[ind_a])
            val_a = A.diags[ind_a][i] * a
            if location_a < rows_a 
                index_ia = rows_a - location_a + i 
                index_ja = i 
            else
                index_ia = i 
                index_ja = location_a - cols_a + i 
            end
            min_loc = rows_b - index_ja + 1
            max_loc = 2 * rows_b - index_ja
            for (ind_b, location_b) in enumerate(B.indices)
                #index_ib = index_ja
                #       if ind < rows(A), then index = (rows - loc + i, i)
                #rows - loc + j = index_ja, j = index_ja - rows + loc
                #       else index = (i, loc - cols + i)
                # if location < rows(B), then 
                if location_b <= rows_b && location_b >= min_loc
                    j = index_ja - rows_b + location_b
                    index_jb = j
                    val_b = B.diags[ind_b][j]
                    C[index_ia, index_jb] = muladd(val_a, val_b, C[index_ia, index_jb])         
                elseif location_b > rows_b && location_b <= max_loc
                    j = index_ja
                    index_jb = location_b - cols_b + j 
                    val_b = B.diags[ind_b][j]
                    C[index_ia, index_jb] = muladd(val_a, val_b, C[index_ia, index_jb])         
                end           
            end
        end
    end
    C
end

function LinearAlgebra.mul!(C :: Matrix{T}, A:: SparseBandedMatrix{T}, B :: SparseBandedMatrix{T}, a :: Number, b :: Number) where T
    @assert size(A, 2) == size(B, 1)
    @assert size(A, 1) == size(C, 1)
    @assert size(B, 2) == size(C, 2)

    C.*=b

    rows_a, cols_a = size(A)
    rows_b, cols_b = size(B)
    @inbounds for (ind_a, location_a) in enumerate(A.indices)
        @threads for i in eachindex(A.diags[ind_a])
            val_a = A.diags[ind_a][i] * a
            if location_a < rows_a 
                index_ia = rows_a - location_a + i 
                index_ja = i 
            else
                index_ia = i 
                index_ja = location_a - cols_a + i 
            end
            min_loc = rows_b - index_ja + 1
            max_loc = 2 * rows_b - index_ja
            for (ind_b, location_b) in enumerate(B.indices)
                #index_ib = index_ja
                #       if ind < rows(A), then index = (rows - loc + i, i)
                #rows - loc + j = index_ja, j = index_ja - rows + loc
                #       else index = (i, loc - cols + i)
                # if location < rows(B), then 
                if location_b <= rows_b && location_b >= min_loc
                    j = index_ja - rows_b + location_b
                    index_jb = j
                    val_b = B.diags[ind_b][j]
                    C[index_ia, index_jb] = muladd(val_a, val_b, C[index_ia, index_jb])         
                elseif location_b > rows_b && location_b <= max_loc
                    j = index_ja
                    index_jb = location_b - cols_b + j 
                    val_b = B.diags[ind_b][j]
                    C[index_ia, index_jb] = muladd(val_a, val_b, C[index_ia, index_jb])         
                end           
            end
        end
    end
    C
end

using VectorizedRNG
using LinearAlgebra: Diagonal, I
using LoopVectorization
using RecursiveFactorization
using SparseArrays

@inline exphalf(x) = exp(x) * oftype(x, 0.5)
function 🦋!(wv, ::Val{SEED} = Val(888)) where {SEED}
    T = eltype(wv)
    mrng = VectorizedRNG.MutableXoshift(SEED)
    GC.@preserve mrng begin rand!(exphalf, VectorizedRNG.Xoshift(mrng), wv, static(0),
                                  T(-0.05), T(0.1)) end
end

function 🦋generate_random!(A, ::Val{SEED} = Val(888)) where {SEED}
    Usz = 2 * size(A, 1)
    Vsz = 2 * size(A, 2)
    uv = similar(A, Usz + Vsz)
    🦋!(uv, Val(SEED))
    (uv,)
end

function 🦋workspace(A, ::Val{SEED} = Val(888)) where {SEED}
    B = similar(A);
    ws = 🦋generate_random!(B)
    🦋mul!(copyto!(B, A), ws)
    U, V = materializeUV(B, ws)
    F = RecursiveFactorization.lu!(B, Val(false))

    U, V, F
end

const butterfly_workspace = 🦋workspace;

function 🦋mul_level!(A, u, v)
    M, N = size(A)
    Ml = M >>> 1
    Nl = N >>> 1
    Mh = M - Ml
    Nh = N - Nl
    @turbo for n in 1:Nl
        for m in 1:Ml
            A11 = A[m, n]
            A21 = A[m + Mh, n]
            A12 = A[m, n + Nh]
            A22 = A[m + Mh, n + Nh]

            T1 = A11 + A12
            T2 = A21 + A22
            T3 = A11 - A12
            T4 = A21 - A22
            C11 = T1 + T2
            C21 = T1 - T2
            C12 = T3 + T4
            C22 = T3 - T4

            u1 = u[m]
            u2 = u[m + Mh]
            v1 = v[n]
            v2 = v[n + Nh]

            A[m, n] = u1 * C11 * v1
            A[m + Mh, n] = u2 * C21 * v1
            A[m, n + Nh] = u1 * C12 * v2
            A[m + Mh, n + Nh] = u2 * C22 * v2
        end
    end
end

function 🦋mul!(A, (uv,))
    M, N = size(A)
    Mh = M >>> 1
    Nh = N >>> 1
    U₁ = @view(uv[1:Mh])
    U₂ = @view(uv[(1 + Mh + Nh):(2 * Mh + Nh)])
    V₁ = @view(uv[(Mh + 1):(Mh + Nh)])
    V₂ = @view(uv[(1 + 2 * Mh + Nh):(2 * Mh + 2 * Nh)])
    🦋mul_level!(@view(A[1:Mh, 1:Nh]), U₁, V₁)
    🦋mul_level!(@view(A[(1 + Mh):M, 1:Nh]), U₂, V₁)
    🦋mul_level!(@view(A[1:Mh, (1 + Nh):N]), U₁, V₂)
    🦋mul_level!(@view(A[(1 + Mh):M, (1 + Nh):N]), U₂, V₂)
    U = @view(uv[(1 + 2 * Mh + 2 * Nh):(2 * Mh + 2 * Nh + M)])
    V = @view(uv[(1 + 2 * Mh + 2 * Nh + M):(2 * Mh + 2 * Nh + M + N)])
    🦋mul_level!(@view(A[1:M, 1:N]), U, V)
    A
end

function diagnegbottom(x)
    N = length(x)
    y = similar(x, N >>> 1)
    z = similar(x, N >>> 1)
    for n in 1:(N >>> 1)
        y[n] = x[n]
    end
    for n in 1:(N >>> 1)
        z[n] = x[n + (N >>> 1)]
    end
    Diagonal(y), Diagonal(z)
end

function 🦋2!(C, A::Diagonal, B::Diagonal)
    @assert size(A) == size(B)
    A1 = size(A, 1)

    for i in 1:A1
        C[i, i] = A[i, i]
        C[i + A1, i] = A[i, i]
        C[i, i + A1] = B[i, i]
        C[i + A1, i + A1] = -B[i, i]
    end

    C
end

function 🦋!(A::Matrix, C::SparseBandedMatrix, X::Diagonal, Y::Diagonal)
    @assert size(X) == size(Y)
    if (size(X, 1) + size(Y, 1) != size(A, 1))
        x = size(A, 1) - size(X, 1) - size(Y, 1)
        setdiagonal!(C, [X.diag; rand(x); -Y.diag], true)
        setdiagonal!(C, X.diag, true)
        setdiagonal!(C, Y.diag, false)
    else
        setdiagonal!(C, [X.diag; -Y.diag], true)
        setdiagonal!(C, X.diag, true)
        setdiagonal!(C, Y.diag, false)
    end

    C
end

function 🦋2!(C::SparseBandedMatrix, A::Diagonal, B::Diagonal)
    setdiagonal!(C, [A.diag; -B.diag], true)
    setdiagonal!(C, A.diag, true)
    setdiagonal!(C, B.diag, false)
    C
end

function materializeUV(A, (uv,))
    M, N = size(A)
    Mh = M >>> 1    
    Nh = N >>> 1

    U₁u, U₁l = diagnegbottom(@view(uv[1:Mh]))
    U₂u, U₂l = diagnegbottom(@view(uv[(1 + Mh + Nh):(2 * Mh + Nh)]))
    V₁u, V₁l = diagnegbottom(@view(uv[(Mh + 1):(Mh + Nh)]))
    V₂u, V₂l = diagnegbottom(@view(uv[(1 + 2 * Mh + Nh):(2 * Mh + 2 * Nh)]))
    Uu, Ul = diagnegbottom(@view(uv[(1 + 2 * Mh + 2 * Nh):(2 * Mh + 2 * Nh + M)]))
    Vu, Vl = diagnegbottom(@view(uv[(1 + 2 * Mh + 2 * Nh + M):(2 * Mh + 2 * Nh + M + N)]))

    #WRITE OUT MERGINGS EXPLICITLY
    #Bu2 = [🦋(U₁u, U₁l) 0*I
    #       0*I 🦋(U₂u, U₂l)]

    #Bu2 = spzeros(M, N)

    mrng = VectorizedRNG.MutableXoshift(888)
    T = typeof(uv[1])

    Bu2 = SparseBandedMatrix{typeof(uv[1])}(undef, M, N)
    
    🦋2!(view(Bu2, 1 : (M ÷ 4) * 2, 1 : (N ÷ 4) * 2), U₁u, U₁l)
    🦋2!(view(Bu2, M - M ÷ 4 * 2 + 1: M, N - N ÷ 4 * 2 + 1: N), U₂u, U₂l)
    rand!(mrng, diag(view(Bu2, 1 : (M ÷ 4) * 2, 1 : (N ÷ 4) * 2)), static(0), T(-0.05), T(0.1))


    #Bu1 = spzeros(M, N)
    Bu1 = SparseBandedMatrix{typeof(uv[1])}(undef, M, N)
    🦋!(A, Bu1, Uu, Ul)

    #Bv2 = spzeros(M, N)
    Bv2 = SparseBandedMatrix{typeof(uv[1])}(undef, M, N)

    🦋2!(view(Bv2, 1 : (M ÷ 4) * 2, 1 : (N ÷ 4) * 2), V₁u, V₁l)
    🦋2!(view(Bv2, M - M ÷ 4 * 2 + 1: M, N - N ÷ 4 * 2 + 1: N), V₂u, V₂l)
    rand!(mrng, diag(view(Bv2, 1 : (M ÷ 4) * 2, 1 : (N ÷ 4) * 2)), static(0), T(-0.05), T(0.1))

    #Bv1 = spzeros(M, N)
    Bv1 = SparseBandedMatrix{typeof(uv[1])}(undef, M, N)
    🦋!(A, Bv1, Vu, Vl)

    #U = similar(A)
    #U = SparseBandedMatrix{typeof(uv[1])}(undef, M, N)

    #mul!(U, Bu2, Bu1, 1, 0)

    #V = similar(A)
    #V = SparseBandedMatrix{typeof(uv[1])}(undef, M, N)
    #mul!(V, Bv2, Bv1, 1, 0)
    #U = sparse(U)
    #V = sparse(V)
 
    (Bu2 * Bu1)', Bv2 * Bv1
end

function pad!(A)
    M, N = size(A)
    xn = 4 - M % 4
    A_new = similar(A, M + xn, N + xn)
    for j in 1 : N, i in 1 : M
        @inbounds A_new[i, j] = A[i, j]
    end

    for j in M + 1 : M + xn, i in 1:M
        @inbounds A_new[i, j] = 0
        @inbounds A_new[j, i] = 0
    end

    for j in N + 1 : N + xn, i in M + 1 : M + xn
        @inbounds A_new[i,j] = i == j
    end
    A_new
end









#=
using SparseArrays, BenchmarkTools, Random

function get_data1()
    dim = 5000
    x = rand(10:75)
    diag_vals = Vector{Vector{Float64}}(undef, x)
    diag_locs = randperm(dim * 2 - 1)[1:x]
    for j in 1:x
        diag_vals[j] = rand(min(diag_locs[j], 2 * dim - diag_locs[j]))
    end

    x_butterfly = SparseBandedMatrix{Float64}(diag_locs, diag_vals, dim, dim)
    x_dense = copy(x_butterfly)

    y = rand(dim, dim)
    z = zeros(dim, dim)

    @show norm(x_dense*y - x_butterfly * y) 
    
    println("Timing dense multiplication.")
    println("(left-side mul)")
    @btime x_dense*y;
    println("(right-side mul)")
    @btime y*x_dense;
    println("\nTiming butterfly multiplication.")
    println("(left-side mul)")
    @btime x_butterfly*y;
    println("(right-side mul)")
    @btime y*x_butterfly;
    
    nothing
end

function get_data2()
    dim = 1000
    x = rand(10:40)
    diag_vals = Vector{Vector{Float64}}(undef, x)
    diag_locs = randperm(dim * 2 - 1)[1:x]
    for j in 1:x
        diag_vals[j] = rand(min(diag_locs[j], 2 * dim - diag_locs[j]))
    end

    x_butterfly = SparseBandedMatrix{Float64}(diag_locs, diag_vals, dim, dim)
    x_dense = copy(x_butterfly)
    x_sparse = sparse(x_dense)

    y = rand(10:40)
    diag_vals = Vector{Vector{Float64}}(undef, y)
    diag_locs = randperm(dim * 2 - 1)[1:y]
    for j in 1:y
        diag_vals[j] = rand(min(diag_locs[j], 2 * dim - diag_locs[j]))
    end

    y_butterfly = SparseBandedMatrix{Float64}(diag_locs, diag_vals, dim, dim)
    y_dense = copy(y_butterfly)
    y_sparse = sparse(y_dense)

    a = true
    b = false
    @assert isapprox(x_butterfly * y_butterfly, x_dense * y_dense)
    println("Timing butterfly multiplication.")
    @btime mul!(zeros(dim, dim), x_butterfly, y_butterfly, a, b);
    println("\nTiming sparse multiplication.")
    @btime mul!(zeros(dim, dim), x_sparse, y_sparse, a, b);
    println("\nTiming dense multiplication.")
    @btime mul!(zeros(dim, dim), x_dense, y_dense, a, b);

    nothing
end
=#

