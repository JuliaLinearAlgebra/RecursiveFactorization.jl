import RecursiveFactorization
using LinearAlgebra: BlasInt

# Low-level I/O using libc read/write on file descriptors
# This avoids needing Core.stdin which isn't available in trimmed binaries

# Set a file descriptor to blocking mode.
# The juliac runtime's libuv may set fds to non-blocking, which breaks raw read/write.
function set_blocking!(fd::Cint)::Nothing
    flags = ccall(:fcntl, Cint, (Cint, Cint), fd, 3)  # F_GETFL = 3
    flags == -1 && error("fcntl F_GETFL failed")
    # Clear O_NONBLOCK (0x800 on Linux)
    new_flags = flags & ~Cint(0x800)
    ret = ccall(:fcntl, Cint, (Cint, Cint, Cint), fd, 4, new_flags)  # F_SETFL = 4
    ret == -1 && error("fcntl F_SETFL failed")
    nothing
end

function fd_read!(fd::Cint, buf::Ptr{UInt8}, nbytes::Int)::Nothing
    remaining = nbytes
    offset = 0
    while remaining > 0
        n = ccall(:read, Cssize_t, (Cint, Ptr{UInt8}, Csize_t), fd, buf + offset, remaining)
        n <= 0 && error("read failed")
        remaining -= n
        offset += n
    end
    nothing
end

function fd_write(fd::Cint, buf::Ptr{UInt8}, nbytes::Int)::Nothing
    remaining = nbytes
    offset = 0
    while remaining > 0
        n = ccall(:write, Cssize_t, (Cint, Ptr{UInt8}, Csize_t), fd, buf + offset, remaining)
        n <= 0 && error("write failed")
        remaining -= n
        offset += n
    end
    nothing
end

function read_value(fd::Cint, ::Type{T}) where {T}
    buf = Ref{T}()
    GC.@preserve buf fd_read!(fd, Ptr{UInt8}(pointer_from_objref(buf)), sizeof(T))
    return buf[]
end

function write_value(fd::Cint, x::T) where {T}
    buf = Ref{T}(x)
    GC.@preserve buf fd_write(fd, Ptr{UInt8}(pointer_from_objref(buf)), sizeof(T))
    nothing
end

function read_matrix!(fd::Cint, A::AbstractArray)
    GC.@preserve A fd_read!(fd, Ptr{UInt8}(pointer(A)), sizeof(eltype(A)) * length(A))
    nothing
end

function write_array(fd::Cint, A::AbstractArray)
    GC.@preserve A fd_write(fd, Ptr{UInt8}(pointer(A)), sizeof(eltype(A)) * length(A))
    nothing
end

function process_f64(fdin::Cint, fdout::Cint, m::Int64, n::Int64, mn::Int64,
                     pivot::Val, thread::Val)
    A = Matrix{Float64}(undef, m, n)
    read_matrix!(fdin, A)
    ipiv = Vector{Int64}(undef, mn)
    F = RecursiveFactorization.lu!(A, ipiv, pivot, thread; check = false)
    write_value(fdout, Int64(F.info))
    write_array(fdout, A)
    write_array(fdout, ipiv)
    return nothing
end

function process_f32(fdin::Cint, fdout::Cint, m::Int64, n::Int64, mn::Int64,
                     pivot::Val, thread::Val)
    A = Matrix{Float32}(undef, m, n)
    read_matrix!(fdin, A)
    ipiv = Vector{Int64}(undef, mn)
    F = RecursiveFactorization.lu!(A, ipiv, pivot, thread; check = false)
    write_value(fdout, Int64(F.info))
    write_array(fdout, A)
    write_array(fdout, ipiv)
    return nothing
end

function (@main)(args::Vector{String})
    fdin = Cint(0)   # stdin fd
    fdout = Cint(1)   # stdout fd

    # The juliac runtime (libuv) may set fds to non-blocking mode.
    # Reset to blocking for our raw read/write calls.
    set_blocking!(fdin)
    set_blocking!(fdout)

    while true
        cmd = read_value(fdin, UInt8)
        cmd == 0xff && break

        m = read_value(fdin, Int64)
        n = read_value(fdin, Int64)
        mn = min(m, n)

        if cmd == 0x00
            process_f64(fdin, fdout, m, n, mn, Val(true), Val(false))
        elseif cmd == 0x01
            process_f32(fdin, fdout, m, n, mn, Val(true), Val(false))
        elseif cmd == 0x02
            process_f64(fdin, fdout, m, n, mn, Val(true), Val(true))
        elseif cmd == 0x03
            process_f32(fdin, fdout, m, n, mn, Val(true), Val(true))
        elseif cmd == 0x04
            process_f64(fdin, fdout, m, n, mn, Val(false), Val(false))
        elseif cmd == 0x05
            process_f32(fdin, fdout, m, n, mn, Val(false), Val(false))
        end
    end
    return 0
end
