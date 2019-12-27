
import GZip, CodecBzip2

const LIBSVM_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
const SEPARATOR = ":"

struct LibSVMData{Index<:Integer, T<:Real}
    # Label to predict
    labels::Vector{T}
    # Features (as sparse matrix)
    index_rows::Vector{Index}
    index_cols::Vector{Index}
    coefs::Vector{T}
end

LibSVMData{Index, T}() where {Index, T} = LibSVMData(T[], Index[], Index[], T[])
ndata(data::LibSVMData) = length(data.labels)
nfeatures(data::LibSVMData) = maximum(data.index_cols)

function to_dense(data::LibSVMData{I, T}) where {I, T}
    n, m = ndata(data), nfeatures(data)
    X = zeros(T, n, m)
    for (i, j, val) in zip(data.index_rows, data.index_cols, data.coefs)
        @inbounds X[i, j] = val
    end
    return X
end

function to_sparse(data::LibSVMData{I, T}) where {I, T}
    n, m = ndata(data), nfeatures(data)
    return sparse(data.index_rows, data.index_cols, data.coefs, n, m)
end

function parse_libsvm(filename::String, T::Type=Float64)
    dataset = LibSVMData{Int, T}()
    gzip_open(filename, "r") do io
         _fetch_dataset(io, dataset)
    end
    return dataset
end

function gzip_open(f::Function, filename::String, mode::String)
    if endswith(filename, ".gz")
        return GZip.open(f, filename, mode)
    elseif endswith(filename, ".bz2")
        return Base.open(f, CodecBzip2.Bzip2DecompressorStream, filename, mode)
    else
        return open(f, filename, mode)
    end
end

function _fetch_dataset(io::IO, data::LibSVMData{I, T}) where {I, T}
    nline = 0
    nindex = 0
    while !eof(io)
        nline += 1
        line = strip(readline(io))
        values_ = split(line, " ", keepempty=false)
        push!(data.labels, parse(T, values_[1]))
        for val in values_[2:end]
            tmp = split(val, SEPARATOR, limit=2)
            ncol = parse(I, tmp[1])
            push!(data.index_rows, nline)
            push!(data.index_cols, ncol)
            push!(data.coefs, parse(T, tmp[2]))
        end
    end
end
