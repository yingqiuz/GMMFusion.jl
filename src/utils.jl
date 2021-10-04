using Base: get_compiletime_preferences
rando(d) = qr!(randn(Float32, d, d)).Q

# Fisher's Linear discriminant
function fda(X, y, q)
	n, d = size(X)
	n == size(y, 1) || throw(DimensionMismatch("Dimensions of X and y mismatch."))
	K = length(unique(y))
	R = [y[nn] == kk-1 ? 1 : 0 for nn in 1:n, kk in 1:K]
	nk = sum(R, dims=1)
	m = mean(X', dims=2)
	C = cov(X)
	mk = X' * R ./ nk
	mo = (mk .- m) .* sqrt.(nk ./ n)
	vals, vecs = eigen(mo * mo', C)
	i = sortperm(-vals)
	vecs = vecs[:, i]
	vecs ./= sum(vecs.^ 2, dims=1)
	return vecs[:, 1:q]
end

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

# create single shell data
function create_single_shell(subject::String)
    bvecs = readdlm(subject * "/bvecs", Float32)
    bvals = readdlm(subject * "/bvals", Int)
    index = findall(x -> x < 1600, bvals[:])
    bvecs = bvecs[:, index]
    bvals = bvals[:, index]
    index .-= 1
    vols = [subject * "/2mm_single/data" * lpad(k, 4, "0") * ".nii.gz" for k ∈ index]
    cmd = `fslmerge -t $(subject)/2mm_single/data $(vols)`
    println(cmd)
    run(cmd)

    # save new bvals and bvecs
    open(subject * "/2mm_single/bvecs", "w") do io
        writedlm(io, bvecs)
    end
    open(subject * "/2mm_single/bvals", "w") do io
        writedlm(io, bvals)
    end
end

function create_single_shell_less_dir(subject::String, dir::Int=32)
    bvecs_orig = readdlm(subject * "/bvecs", Float32)
    bvals_orig = readdlm(subject * "/bvals", Int)
    mydir = readdlm(subject * "/2mm_single$(dir)/$(dir)_dirs.txt", Float32)
    index = findall(x -> x < 1600, bvals_orig[:])
    bvecs = bvecs_orig[:, index]
    bvals = bvals_orig[:, index]
    # select 32 directions
    # b1 index
    b1index = findall(x -> x> 100, bvals[:])
    b0index = findall(x -> x< 100, bvals[:])
    b1vecs = bvecs[:, b1index]
    proj = abs.(mydir * b1vecs)
    @info "proj" proj
    # index 2 stores the indices of bvecs close to the 32 dirs
    index32 = vec([el[2] for el in argmax(proj, dims=2)])
    @info "index32" index32
    bvecs32 = b1vecs[:, index32]
    newindex = unique!([el[2] for el in findall(x -> abs(x)>0.98, bvecs32' * b1vecs)])
    @info "newindex" newindex
    index = sort!(index[vcat(b0index, b1index[newindex])])
    @info "index" index
    #b1index = [b2index[k] for k in 1:length(b1index) if !(k in newindex)]
    !ispath(subject * "/2mm_single$(dir)") && mkpath(subject * "/2mm_single$(dir)")
    vols = [subject * "/2mm_single/data" * lpad(k-1, 4, "0") * ".nii.gz" for k ∈ index]
    cmd = `fslmerge -t $(subject)/2mm_single$(dir)/data $(vols)`
    println(cmd)
    run(cmd)
    # save new bvals and bvecs
    open(subject * "/2mm_single$(dir)/bvecs", "w") do io
        writedlm(io, bvecs_orig[:, index])
    end
    open(subject * "/2mm_single$(dir)/bvals", "w") do io
        writedlm(io, bvals_orig[:, index])
    end
end

function add_noise(subject::String, σ::Float32=1f0)
    data = niread(subject * "/2mm/data.nii.gz")
    noise = rand(Rayleigh(σ), size(data.raw)...)
    data.raw .+= noise
    !ispath(subject * "/2mm_noise_$(σ)") && mkpath(subject * "/2mm_noise_$(σ)")
    niwrite(subject * "/2mm_noise_$(σ)/data.nii.gz", data)
end

# clustering of thalamus
function segment(subject::String, thalamus_mask::String, path::String, targets::AbstractArray{String}=["m1", "s1", "cerebellum", "sma"])
    # load mask and data
    mask = niread(join([subject, thalamus_mask], '/'))
    index = findall(x -> x!=0, mask.raw)
    fnames = [subject * "/" * path * "seeds_to_$(k)_1.05.nii.gz" for k ∈ targets]
    X = Matrix{Float32}(undef, size(index, 1), 4)
    for (k, f) ∈ enumerate(fnames)
        X[:, k] .= niread(f).raw[index]
    end
    @avx X .= log.(X)
    X[isinf.(X)] .= 0
    @info "X" X
    res = kmeans(X', 4; maxiter=10000, display=:iter, tol=1e-6)
    @info "res" assignments(res)
    mask.raw[index] .= assignments(res)
    niwrite(subject * "/" * path * "sgmt.nii.gz", mask)
end

function create_conn(fdt::String, wmparc::String, wmparc_bin::String, thalamus::String, output::String)
    thalamus = niread(thalamus).raw[:]
    wmparc = niread(wmparc).raw[:]
    wmparc_bin = niread(wmparc_bin).raw[:]
    n1 = count(thalamus .!= 0)
    n2 = count(wmparc_bin .!= 0)
    @info "tha/wmparc count" n1 n2
    data = zeros(Float32, n1, n2)
    prog = ProgressUnknown("read data...", spinner=true)
    for line in eachline(fdt)
        ProgressMeter.next!(prog)
        x, y, val = split(line, "  ")
        #@info "line" x y val
        data[parse(Int, x), parse(Int, y)] = parse(Float32, val)
    end
    ProgressMeter.finish!(prog)
    # parcels
    wmparc = wmparc[wmparc_bin .!= 0]
    @info "wmparc" size(wmparc)
    labels = sort!(unique(wmparc))
    @info "labels" labels
    conn = zeros(Float32, n1, length(labels))
    @showprogress 0.01 "parcellate..." for (k, label) ∈ enumerate(labels)
        index = findall(x -> x == label, wmparc)
        conn[:, k] = sum(data[:, index], dims=2)
    end
    @info "conn" conn
    # save file
    h5open(output, "w") do file
        write(file, "conn", conn, "labels", labels)
    end
end

function segment_kmeans(filename::String, thalamus_mask::String, output::String, K::Int)
    # load mask and data
    mask = niread(thalamus_mask)
    index = findall(x -> x!=0, mask.raw)
    X = h5read(filename, "conn")
    #labels = h5read(filename, "labels")
    @avx X .= log.(X)
    X[isinf.(X)] .= 0
    @info "X" X
    res = kmeans(X', K; maxiter=10000, display=:iter, tol=1e-6)
    @info "res" assignments(res)
    mask.raw[index] .= assignments(res)
    niwrite(output, mask)
end

function segment_gmm(filename::String, thalamus_mask::String, output::String, K::Int)
    # load mask and data
    mask = niread(thalamus_mask)
    index = findall(x -> x!=0, mask.raw)
    X = h5read(filename, "conn")
    #labels = h5read(filename, "labels")
    @avx X .= log.(X)
    X[isinf.(X)] .= 0
    @info "X" X
    model = GMMFusion.EM(X, K; maxiter=10000, tol=1f-6)
    res = GMMFusion.predict(model, X)
    res = onecold(res', 1:K)
    @info "res" res
    mask.raw[index] .= res
    niwrite(output, mask)
    #!ispath(output) && mkpath(output)
    #for k ∈ 1:K
    #    mask.raw[index] .= res[:, k]
    #    niwrite(output * "/$(k).nii.gz", mask)
    #end
end

function segment_mrf(filename::String, thalamus_mask::String, output::String, K::Int, ω::Float32=1f0)
    # load mask and data
    mask = niread(thalamus_mask)
    index = findall(x -> x!=0, mask.raw)
    X = h5read(filename, "conn")
    n, d = size(X)
    # find adj list
    adj = Array{Tuple}(undef, n)
    neighbours = Array{CartesianIndex}(undef, 6)
    @inbounds for v ∈ 1:n
        x, y, z = [index[v][k] for k ∈ 1:3]
        neighbours .= [
            CartesianIndex(x-1, y, z), CartesianIndex(x+1, y, z), 
            CartesianIndex(x, y-1, z), CartesianIndex(x, y+1, z), 
            CartesianIndex(x, y, z-1), CartesianIndex(x, y, z+1)
        ]
        adj[v] = Tuple(el for el ∈ findall(x -> (x ∈ neighbours), index))
        @debug "adj[$(v)]" adj[v]
    end
    #h5write(filename, "adj", adj)
    #labels = h5read(filename, "labels")
    X ./= sum(X, dims=2)
    X[X .== 0] .= 1f-8
    @avx X .= log.(X)
    X[isinf.(X)] .= 0
    @info "X" X
    @info "isinf.(X)" findall(isinf, X)
    res = kmeans(X', K; maxiter=10000, display=:iter, tol=1e-6)
    R = convert(Array{Float32}, [x == k ? 1 : 0 for x ∈ assignments(res), k ∈ 1:K])
    @info "R" R
    model = GMMFusion.MRFBatch(X=X, adj=adj, R=R, ω=ω, n=n, d=d, K=K, μ=copy(res.centers), Σ=[cholesky!(cov(X)+ I * 1f-6) for _ ∈ 1:K])
    GMMFusion.MrfMixGauss!(model; maxiter=200, tol=1f-6)
    results = Flux.onecold(model.R', 1:K)
    @info "res" results
    mask.raw[index] .= results
    niwrite(output, mask)
    #!ispath(output) && mkpath(output)
    #for k ∈ 1:K
    #    mask.raw[index] .= res[:, k]
    #    niwrite(output * "/$(k).nii.gz", mask)
    #end
end

function vim_segment_read_data(
    subject::String, hemi::String, K::Int, out::String=subject, penalty::Float32=1f0
)
    thalamus = niread(joinpath(subject, "mist_$(hemi)_thalamus_mask.nii.gz"))
    index = findall(x -> x!=0, thalamus.raw)
    n = length(index)
    if isfile(joinpath(subject, out, hemi, "conn2HCPMM1.h5"))
        X = h5read(joinpath(subject, out, hemi, "conn2HCPMM1.h5"), "conn")
    else
        targets = niread(joinpath(subject, "Native", hemi, "stop_1.05_labels.nii.gz")).raw
        targets = targets[findall(x -> x != 0, targets)]
        d = length(targets)
        @info "thalamus/targets dim" n d
        data = zeros(Float32, n, d)
        prog = ProgressUnknown("read data...", spinner=true)
        for line in eachline(joinpath(subject, out, hemi, "fdt_matrix2.dot"))
            ProgressMeter.next!(prog)
            x, y, val = split(line, "  ")
            #@info "line" x y val
            data[parse(Int, x), parse(Int, y)] = parse(Float32, val)
        end
        ProgressMeter.finish!(prog)
        labels = sort!(unique(targets))
        X = zeros(Float32, n, length(labels))
        @showprogress 0.01 "parcellate..." for (l, label) ∈ enumerate(labels)
            ind = findall(x -> x == label, targets)
            Xk = view(X, :, l)
            sum!(Xk, view(data, :, ind))
            #X[:, k] = sum(data[:, ind], dims=2)
        end
        h5open(joinpath(subject, out, hemi, "conn2HCPMM1.h5"), "w") do f
            write(f, "conn", X, "labels", labels)
        end
    end
    # find adj list
    neighbours = Array{CartesianIndex}(undef, 6)
    adj = Array{Tuple}(undef, n)
    @inbounds for v ∈ 1:n
        x, y, z = [index[v][k] for k ∈ 1:3]
        copyto!(neighbours, [
            CartesianIndex(x-1, y, z), CartesianIndex(x+1, y, z), 
            CartesianIndex(x, y-1, z), CartesianIndex(x, y+1, z), 
            CartesianIndex(x, y, z-1), CartesianIndex(x, y, z+1)
        ])
        adj[v] = Tuple(el for el ∈ findall(x -> (x ∈ neighbours), index))
        @debug "adj[$(v)]" adj[v]
    end
    #h5write(filename, "adj", adj)
    #labels = h5read(filename, "labels")
    X = X[:, [8, 181]]
    @info "isnan.(X)" findall(isnan, X)
    #X[findall(isnan, X)] .= 0f0
    #@info "isinf.(X)" Xcopy[findall(isnan, X)]
    #@info "Xmim" minimum(X[X .!= 0]) minimum(X[X .!= 0])*0.1f0
    #X[findall(x -> x==0, X)] .= minimum(X[findall(x -> x!=0, X)]) * 0.1f0
    @avx X .= log.(X)
    X[findall(isinf, X)] .= 0f0
    @info "isinf.(X)" findall(isinf, X)
    @info "isnan.(X)" findall(isnan, X)
    X .-= mean(X, dims=1)
    X ./= std(X, dims=1)
    ## load initial segmentation
    Rinit = Flux.onehotbatch(
        niread(
            joinpath(subject, out, "segmentation/$(hemi)_vim.nii.gz")
        ).raw[index], 0:K-1
    )
    model = GMMFusion.MRFBatchSeg(K=K, X=X, R=convert(Array{Float32}, Rinit'), adj=adj, ω=penalty)
    GMMFusion.MrfMixGauss!(model; maxiter=1000)
    c = count(model.seg .== Flux.onecold(convert(Array{Float32}, Rinit), 1:2) .== 2)
    c1 = count(model.seg .== 2) 
    c2 = count(Flux.onecold(convert(Array{Float32}, Rinit), 1:2) .== 2)
    @info "count mrf" c1
    @info "count vim" c2
    @info "count common" c
    thalamus.raw[index] .= model.seg .- 1f0
    niwrite(join([subject, out, hemi, "vim_sgmt.penalty.$(model.ω).nii.gz"], "/"), thalamus)
    return model.seg
end

function vim_segment_read_data_paired(
    subject::String, hemi::String, K::Int, out::String=subject, penalty::Float32=1f0
)
    thalamus = niread(joinpath(subject, "mist_$(hemi)_thalamus_mask.nii.gz"))
    index = findall(x -> x!=0, thalamus.raw)
    n = length(index)
    if isfile(joinpath(subject, subject, hemi, "conn2HCPMM1.h5"))
        X = h5read(joinpath(subject, subject, hemi, "conn2HCPMM1.h5"), "conn")
    else
        targets = niread(joinpath(subject, "Native", hemi, "stop_1.05_labels.nii.gz")).raw
        targets = targets[findall(x -> x != 0, targets)]
        d = length(targets)
        @info "thalamus/targets dim" n d
        data = zeros(Float32, n, d)
        prog = ProgressUnknown("read data...", spinner=true)
        for line in eachline(joinpath(subject, subject, hemi, "fdt_matrix2.dot"))
            ProgressMeter.next!(prog)
            x, y, val = split(line, "  ")
            #@info "line" x y val
            data[parse(Int, x), parse(Int, y)] = parse(Float32, val)
        end
        ProgressMeter.finish!(prog)
        labels = sort!(unique(targets))
        X = zeros(Float32, n, length(labels))
        @showprogress 0.01 "parcellate..." for (l, label) ∈ enumerate(labels)
            ind = findall(x -> x == label, targets)
            Xk = view(X, :, l)
            sum!(Xk, view(data, :, ind))
            #X[:, k] = sum(data[:, ind], dims=2)
        end
        h5open(joinpath(subject, subject, hemi, "conn2HCPMM1.h5"), "w") do f
            write(f, "conn", X, "labels", labels)
        end
    end

    if isfile(joinpath(subject, out, hemi, "conn2HCPMM1.h5"))
        X2 = h5read(joinpath(subject, out, hemi, "conn2HCPMM1.h5"), "conn")
    else
        targets = niread(joinpath(subject, "Native", hemi, "stop_1.05_labels.nii.gz")).raw
        targets = targets[findall(x -> x != 0, targets)]
        d = length(targets)
        @info "thalamus/targets dim" n d
        data = zeros(Float32, n, d)
        prog = ProgressUnknown("read data...", spinner=true)
        for line in eachline(joinpath(subject, out, hemi, "fdt_matrix2.dot"))
            ProgressMeter.next!(prog)
            x, y, val = split(line, "  ")
            #@info "line" x y val
            data[parse(Int, x), parse(Int, y)] = parse(Float32, val)
        end
        ProgressMeter.finish!(prog)
        labels = sort!(unique(targets))
        X2 = zeros(Float32, n, length(labels))
        @showprogress 0.01 "parcellate..." for (l, label) ∈ enumerate(labels)
            ind = findall(x -> x == label, targets)
            Xk = view(X2, :, l)
            sum!(Xk, view(data, :, ind))
            #X[:, k] = sum(data[:, ind], dims=2)
        end
        h5open(joinpath(subject, out, hemi, "conn2HCPMM1.h5"), "w") do f
            write(f, "conn", X2, "labels", labels)
        end
    end
    # find adj list
    neighbours = Array{CartesianIndex}(undef, 6)
    adj = Array{Tuple}(undef, n)
    @inbounds for v ∈ 1:n
        x, y, z = [index[v][k] for k ∈ 1:3]
        copyto!(neighbours, [
            CartesianIndex(x-1, y, z), CartesianIndex(x+1, y, z), 
            CartesianIndex(x, y-1, z), CartesianIndex(x, y+1, z), 
            CartesianIndex(x, y, z-1), CartesianIndex(x, y, z+1)
        ])
        adj[v] = Tuple(el for el ∈ findall(x -> (x ∈ neighbours), index))
        @debug "adj[$(v)]" adj[v]
    end
    #h5write(filename, "adj", adj)
    #labels = h5read(filename, "labels")
    X = X[:, [8, 181]]
    X2 = X2[:, [8, 181]]
    map([X, X2]) do x 
        @avx x .= log.(x)
        x[findall(isinf, x)] .= 0f0
        x .-= mean(x, dims=1)
        x ./= std(x, dims=1)
    end
    @info "data" X X2
    ## load initial segmentation
    Rinit = Flux.onehotbatch(
        niread(
            joinpath(subject, subject, "segmentation/$(hemi)_vim.nii.gz")
        ).raw[index], 0:K-1
    )
    model = GMMFusion.PairedMRFBatchSeg(K=K, XH=X, XL=X2, R=convert(Array{Float32}, Rinit'), adj=adj, ω=penalty)
    @info "model" model
    GMMFusion.MrfMixGauss!(model; maxiter=1000)
    @info "count mrf" count(model.seg .== 2) 
    @info "count vim" count(Flux.onecold(convert(Array{Float32}, Rinit), 1:2) .== 2)
    @info "count common" count(model.seg .== Flux.onecold(convert(Array{Float32}, Rinit), 1:2) .== 2)
    thalamus.raw[index] .= model.seg .- 1f0
    niwrite(join([subject, out, hemi, "vim_sgmt.penalty.$(model.ω)_with_high.nii.gz"], "/"), thalamus)
end

function vim_segment_save_data(
    subjects::AbstractVector{String}, hemi::String, nall::AbstractArray{Int}, 
    model::Union{MRFBatchSeg{T}, MRFBatch{T}, PairedMRFBatchSeg{T}, PairedMRFBatch{T}}, 
    group::Union{String, Nothing}=nothing
)
    @showprogress for (k, subject) ∈ enumerate(subjects)
        if group === nothing
            out = subject
        else
            out = group
        end
        thalamus = niread(joinpath(subject, "mist_$(hemi)_thalamus_mask.nii.gz"))
        index = findall(x -> x!=0, thalamus.raw)
        seg = model.seg[sum(nall[1:k])-nall[k]+1:sum(nall[1:k])] .- 1
        @info "seg" count(seg .== 1)
        #fill!(thalamus.raw, 0)
        thalamus.raw[index] .= seg
        @info "thalamus" thalamus
        niwrite(joinpath(subject, out, hemi, "vim_sgmt.penalty.$(model.ω).nii.gz"), thalamus)
    end
    nothing
end

function vim_segment_rois(
    subject::String, path::String, hemi::String, K::Int,
    targets::AbstractArray{String}=["m1", "brainstem_cerebellum"]
)
    # load mask and data
    mask = niread(join([subject, "mist_$(hemi)_thalamus_mask.nii.gz"], '/'))
    index = findall(x -> x!=0, mask.raw)
    n = length(index)
    fnames = [join([subject, path, hemi, "$(hemi)_to_$(k)_1_stop.nii.gz"], "/") for k ∈ targets]
    X = Matrix{Float32}(undef, n, length(targets))
    for (k, f) ∈ enumerate(fnames)
        X[:, k] .= niread(f).raw[index]
    end
    @avx X .= log.(X)
    X[findall(isinf, X)] .= 0
    @info "X" X
    X .-= mean(X, dims=1)
    X ./= std(X, dims=1)
    Rinit = Flux.onehotbatch(
            niread(
                joinpath(subject, path, "segmentation", "$(hemi)_vim.nii.gz")
            ).raw[index], 0:K-1
        )
    neighbours = Array{CartesianIndex}(undef, 6)
    # find adj list
    adj = Array{Tuple}(undef, n)
    @inbounds for v ∈ 1:n
        x, y, z = [index[v][k] for k ∈ 1:3]
        copyto!(neighbours, [
            CartesianIndex(x-1, y, z), CartesianIndex(x+1, y, z), 
            CartesianIndex(x, y-1, z), CartesianIndex(x, y+1, z), 
            CartesianIndex(x, y, z-1), CartesianIndex(x, y, z+1)
        ])
        adj[v] = Tuple(el for el ∈ findall(x -> (x ∈ neighbours), index))
        @debug "adj[$(v)]" adj[v]
    end
    #return X, Rinit', adj, n
    model = GMMFusion.MRFBatchSeg(K=K, X=X, R=transpose(Rinit), adj=adj)
    GMMFusion.MrfMixGauss!(model)
    @info "count mrf" count(model.seg .== 2) 
    @info "count vim" count(Flux.onecold(convert(Array{Float32}, Rinit), 1:2) .== 2)
    @info "count common" count(model.seg .== Flux.onecold(convert(Array{Float32}, Rinit), 1:2) .== 2)
    return model.seg
    #mask.raw[index] .= model.seg .- 1f0
    #niwrite(join([subject, path, hemi, "vim_sgmt.penalty.$(model.ω).akram.nii.gz"], "/"), mask)
end