module RecursiveFactorizationShim

import RecursiveFactorization
using LinearAlgebra: BlasInt
import Base.@ccallable

# ============================================================================
# @ccallable entry points for RecursiveFactorization.lu!
#
# Interface mirrors LAPACK dgetrf/sgetrf:
#   A:    pointer to column-major matrix data (modified in-place to L\U)
#   m:    number of rows
#   n:    number of columns
#   ipiv: pointer to pivot index array (length min(m,n)), output
#   returns: info (0 = success, k > 0 = U(k,k) is exactly zero)
# ============================================================================

# --- Float64, with pivoting, no threading ---
@ccallable function recursive_lu_f64!(A::Ptr{Float64}, m::Int64, n::Int64,
                                      ipiv::Ptr{Int64})::Int64
    mat = unsafe_wrap(Matrix{Float64}, A, (m, n))
    ipiv_vec = unsafe_wrap(Vector{Int64}, ipiv, min(m, n))
    F = RecursiveFactorization.lu!(mat, ipiv_vec, Val(true), Val(false); check = false)
    return Int64(F.info)
end

# --- Float32, with pivoting, no threading ---
@ccallable function recursive_lu_f32!(A::Ptr{Float32}, m::Int64, n::Int64,
                                      ipiv::Ptr{Int64})::Int64
    mat = unsafe_wrap(Matrix{Float32}, A, (m, n))
    ipiv_vec = unsafe_wrap(Vector{Int64}, ipiv, min(m, n))
    F = RecursiveFactorization.lu!(mat, ipiv_vec, Val(true), Val(false); check = false)
    return Int64(F.info)
end

# --- Float64, with pivoting, threaded ---
@ccallable function recursive_lu_f64_threaded!(A::Ptr{Float64}, m::Int64, n::Int64,
                                               ipiv::Ptr{Int64})::Int64
    mat = unsafe_wrap(Matrix{Float64}, A, (m, n))
    ipiv_vec = unsafe_wrap(Vector{Int64}, ipiv, min(m, n))
    F = RecursiveFactorization.lu!(mat, ipiv_vec, Val(true), Val(true); check = false)
    return Int64(F.info)
end

# --- Float32, with pivoting, threaded ---
@ccallable function recursive_lu_f32_threaded!(A::Ptr{Float32}, m::Int64, n::Int64,
                                               ipiv::Ptr{Int64})::Int64
    mat = unsafe_wrap(Matrix{Float32}, A, (m, n))
    ipiv_vec = unsafe_wrap(Vector{Int64}, ipiv, min(m, n))
    F = RecursiveFactorization.lu!(mat, ipiv_vec, Val(true), Val(true); check = false)
    return Int64(F.info)
end

# --- Float64, no pivoting, no threading ---
@ccallable function recursive_lu_f64_nopiv!(A::Ptr{Float64}, m::Int64, n::Int64,
                                            ipiv::Ptr{Int64})::Int64
    mat = unsafe_wrap(Matrix{Float64}, A, (m, n))
    ipiv_vec = unsafe_wrap(Vector{Int64}, ipiv, min(m, n))
    F = RecursiveFactorization.lu!(mat, ipiv_vec, Val(false), Val(false); check = false)
    return Int64(F.info)
end

# --- Float32, no pivoting, no threading ---
@ccallable function recursive_lu_f32_nopiv!(A::Ptr{Float32}, m::Int64, n::Int64,
                                            ipiv::Ptr{Int64})::Int64
    mat = unsafe_wrap(Matrix{Float32}, A, (m, n))
    ipiv_vec = unsafe_wrap(Vector{Int64}, ipiv, min(m, n))
    F = RecursiveFactorization.lu!(mat, ipiv_vec, Val(false), Val(false); check = false)
    return Int64(F.info)
end

end # module
