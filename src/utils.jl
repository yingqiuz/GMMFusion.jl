# read files and create fconn
function createfc(
    input::Vector{String}, output::String, seedmask::String, 
    brainmask::String="/well/win/software/packages/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain_mask.nii.gz",
)
    # read masks
    maskfile = niread(seedmask)
    @info "mask size" size(maskfile.raw)
    x, y, z = size(maskfile.raw)
    index = findall(k -> k != 0, maskfile.raw[:])
    nrows = size(index, 1)
    @info "index size" nrows
    maskfile2 = niread(brainmask)
    index2 = findall(k -> k != 0, maskfile2.raw[:])
    ncols = size(index2, 1)
    @info "index2" size(index2, 1)
    fc = zeros(Float32, nrows, ncols)
    # read time series data
    for i ∈ input
        ni = niread(i).raw
        @info "ni size" size(ni)
        ni = reshape(ni, x * y * z, size(ni, 4))
        fc .+= replace_nan(cor(ni[index, :]', ni[index2, :]'))
    end
    fc ./= size(input, 1)
    @info "fc" fc
    h5open(output, "w") do f
        write(f, "fc", fc)
    end
end

function readfc(input::String)
    h5read(input, "fc")
end

function readsc(input::String)
    h5read(input, "sc")
end

replace_nan(v) = map(x -> isnan(x) ? zero(x) : x, v)