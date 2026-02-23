# Juliac binary server management for RecursiveFactorization
# Provides lu!/lu via a juliac-compiled subprocess communicating over pipes.
#
# Protocol (binary, little-endian):
#   Request: cmd(UInt8) + m(Int64) + n(Int64) + A(T[m*n])
#   Response: info(Int64) + A(T[m*n]) + ipiv(Int64[min(m,n)])
#   Commands: 0x00=f64, 0x01=f32, 0x02=f64_threaded, 0x03=f32_threaded,
#             0x04=f64_nopiv, 0x05=f32_nopiv, 0xff=exit

using LinearAlgebra: BlasInt, LU, checknonsingular

# Server process state
const _server_lock = ReentrantLock()
const _server_proc = Ref{Union{Nothing, Base.Process}}(nothing)
const _server_binary = Ref{String}("")

function _juliac_binary_path()
    # Check for binary in package juliac/ directory
    pkg_dir = dirname(dirname(@__FILE__))
    return joinpath(pkg_dir, "juliac", "recfact_server")
end

function _scratch_binary_path()
    # Use a version-specific scratch directory
    scratch_dir = joinpath(first(DEPOT_PATH), "scratchspaces",
        "f2c3362d-daeb-58d1-803e-2bc74f2840b4", # RecursiveFactorization UUID
        "juliac_v$(VERSION.major)_$(VERSION.minor)")
    return joinpath(scratch_dir, "recfact_server")
end

function _find_binary()
    # Check package directory first
    p = _juliac_binary_path()
    isfile(p) && return p
    # Then check scratch space
    p = _scratch_binary_path()
    isfile(p) && return p
    return nothing
end

"""
    RecursiveFactorization.build_binary(; force=false, trim=:unsafe_warn)

Build the juliac binary for RecursiveFactorization. Requires Julia 1.12+.

# Keyword Arguments
- `force::Bool=false`: Rebuild even if binary already exists.
- `trim::Symbol=:unsafe_warn`: Trimming mode (`:safe`, `:unsafe_warn`, or `:no`).
"""
function build_binary(; force::Bool = false, trim::Symbol = :unsafe_warn)
    if VERSION < v"1.12"
        error("Building juliac binary requires Julia 1.12+, currently running $VERSION")
    end

    binary_path = _scratch_binary_path()
    if !force && isfile(binary_path)
        @info "Juliac binary already exists at $binary_path. Use force=true to rebuild."
        return binary_path
    end

    pkg_dir = dirname(dirname(@__FILE__))
    shim_src = joinpath(pkg_dir, "juliac", "shim_exe.jl")
    if !isfile(shim_src)
        error("Shim source not found at $shim_src")
    end

    # Find juliac.jl
    juliac_jl = joinpath(Sys.BINDIR, "..", "share", "julia", "juliac", "juliac.jl")
    if !isfile(juliac_jl)
        error("juliac.jl not found at $juliac_jl. Ensure Julia 1.12+ is installed.")
    end

    # Prepare output directory
    out_dir = dirname(binary_path)
    mkpath(out_dir)

    # Prepare project directory for juliac build
    build_dir = joinpath(out_dir, "build")
    mkpath(build_dir)

    # Create Project.toml for build
    build_project = joinpath(build_dir, "Project.toml")
    open(build_project, "w") do f
        write(f, """
        [deps]
        RecursiveFactorization = "f2c3362d-daeb-58d1-803e-2bc74f2840b4"
        LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
        """)
    end

    # Instantiate the build project with local RecursiveFactorization
    @info "Setting up build environment..."
    run(```$(Base.julia_cmd()) --project=$build_dir -e "
        import Pkg
        Pkg.develop(path=$(repr(pkg_dir)))
        Pkg.instantiate()
    "```)

    trim_flag = if trim === :safe
        "--trim=safe"
    elseif trim === :unsafe_warn
        "--trim=unsafe-warn"
    else
        "--trim=no"
    end

    @info "Building juliac binary (this may take a few minutes)..."
    build_env = copy(ENV)
    build_env["JULIA_PROJECT"] = build_dir
    run(setenv(
        `$(Base.julia_cmd()) $juliac_jl --output-exe $binary_path --experimental $trim_flag $shim_src`,
        build_env))

    if !isfile(binary_path)
        error("Build failed: binary not found at $binary_path")
    end

    @info "Juliac binary built successfully at $binary_path"
    return binary_path
end

function _start_server()
    lock(_server_lock) do
        proc = _server_proc[]
        if proc !== nothing && process_running(proc)
            return
        end

        binary = _find_binary()
        if binary === nothing
            return
        end

        _server_binary[] = binary
        _server_proc[] = open(`$binary`, write = true, read = true)
        # Give the server a moment to initialize
        sleep(0.1)
        if !process_running(_server_proc[])
            _server_proc[] = nothing
            @warn "Juliac server failed to start"
        end
    end
end

function _stop_server()
    lock(_server_lock) do
        proc = _server_proc[]
        if proc === nothing
            return
        end
        if process_running(proc)
            try
                write(proc, UInt8(0xff))
                flush(proc)
            catch
            end
            close(proc.in)
            wait(proc)
        end
        _server_proc[] = nothing
    end
end

function _server_available()
    proc = _server_proc[]
    return proc !== nothing && process_running(proc)
end

function _server_lu!(A::Matrix{T}, ipiv::Vector{Int64},
        pivot::Val{Pivot}, thread::Val{Thread};
        check::Union{Bool, Val{true}, Val{false}} = Val(true)) where {T, Pivot, Thread}
    proc = _server_proc[]
    m, n = size(A)
    mn = min(m, n)

    # Determine command byte
    cmd = if T === Float64
        if !Pivot
            UInt8(0x04)
        elseif Thread
            UInt8(0x02)
        else
            UInt8(0x00)
        end
    else # Float32
        if !Pivot
            UInt8(0x05)
        elseif Thread
            UInt8(0x03)
        else
            UInt8(0x01)
        end
    end

    lock(_server_lock) do
        # Send request
        write(proc, cmd)
        write(proc, Int64(m))
        write(proc, Int64(n))
        write(proc, A)
        flush(proc)

        # Read response
        info_bytes = Vector{UInt8}(undef, 8)
        readbytes!(proc, info_bytes, 8)
        info = reinterpret(Int64, info_bytes)[1]

        A_bytes = Vector{UInt8}(undef, m * n * sizeof(T))
        readbytes!(proc, A_bytes, length(A_bytes))
        A_result = reshape(reinterpret(T, A_bytes), m, n)
        copyto!(A, A_result)

        ipiv_bytes = Vector{UInt8}(undef, mn * sizeof(Int64))
        readbytes!(proc, ipiv_bytes, length(ipiv_bytes))
        ipiv_result = reinterpret(Int64, copy(ipiv_bytes))
        copyto!(ipiv, ipiv_result)

        binfo = BlasInt(info)
        ((check isa Bool && check) || (check === Val(true))) && checknonsingular(binfo)
        return LU(A, BlasInt.(ipiv), binfo)
    end
end

"""
    RecursiveFactorization.juliac_lu!(A, [pivot, thread]; check=true)

LU factorization using the juliac-compiled binary server.
Only supports `Matrix{Float64}` and `Matrix{Float32}`.

Falls back to pure-Julia `lu!` if the server is not available.
"""
function juliac_lu!(A::Matrix{T}, pivot = Val(true), thread = Val(false);
        check::Union{Bool, Val{true}, Val{false}} = Val(true)) where {T <: Union{Float64, Float32}}
    if !_server_available()
        return lu!(A, normalize_pivot(pivot), thread; check = check)
    end
    m, n = size(A)
    mn = min(m, n)
    ipiv = Vector{Int64}(undef, mn)
    npivot = normalize_pivot(pivot)
    return _server_lu!(A, ipiv, npivot, thread; check = check)
end

function juliac_lu!(A::Matrix{T}, ipiv::Vector{Int64},
        pivot = Val(true), thread = Val(false);
        check::Union{Bool, Val{true}, Val{false}} = Val(true)) where {T <: Union{Float64, Float32}}
    if !_server_available()
        return lu!(A, ipiv, normalize_pivot(pivot), thread; check = check)
    end
    npivot = normalize_pivot(pivot)
    return _server_lu!(A, ipiv, npivot, thread; check = check)
end

"""
    RecursiveFactorization.juliac_lu(A, [pivot, thread]; check=true)

LU factorization using the juliac-compiled binary server, on a copy of A.
"""
function juliac_lu(A::AbstractMatrix, pivot = Val(true), thread = Val(false); kwargs...)
    return juliac_lu!(copy(A), pivot, thread; kwargs...)
end

function _init_juliac_server()
    if _find_binary() !== nothing
        _start_server()
    end
end

function _finalize_juliac_server()
    _stop_server()
end
