# packages
using LinearAlgebra
using SparseArrays
using KrylovKit
using SuiteSparseGraphBLAS


# sparse diagonalization (regular BLAS and LAPACK)
## return lowest eigenvalues
function SparseDiagonalization(sectorDim::Int64, matH::SparseMatrixCSC{ComplexF64,Int64}, evNum::Int64;
        krylovdimMultiplier::Int64=16, krylovdimMin::Int64=256)::Vector{Float64}

    # enforce Hermiticity
    matH = Hermitian(matH)

    # sparse diagonalization
    krylovdim = max(krylovdimMin, krylovdimMultiplier*evNum)

    vals, vecs = eigsolve(matH, evNum, :SR, krylovdim=krylovdim)
    sort!(vals)

    energies = vals[1:evNum]
    
    return energies
end


# sparse diagonalization (GraphBLAS)
# return lowest eigenvalues
function SparseDiagonalization_GraphBLAS(sectorDim::Int64, matH::GBMatrix{ComplexF64, ComplexF64}, evNum::Int64;
        krylovdimMultiplier::Int64=16, krylovdimMin::Int64=256)::Vector{Float64}
    
    function apply_matH(x::Vector{ComplexF64})::Vector{ComplexF64}
        return matH * x
    end
    
    # sparse diagonalization
    krylovdim = max(krylovdimMin, krylovdimMultiplier*evNum)

    vals, vecs = eigsolve(apply_matH, sectorDim, evNum, :SR, ComplexF64, ishermitian=true, krylovdim=krylovdim)
    sort!(vals)

    energies = vals[1:evNum]
    
    return energies
end



# sparse diagonalization (GraphBLAS)
# return lowest eigenvalues
function SparseDiagonalization_withGSeigenFunc_GraphBLAS(sectorDim::Int64, matH::GBMatrix{ComplexF64, ComplexF64}, evNum::Int64;
        krylovdimMultiplier::Int64=16, krylovdimMin::Int64=256)::Tuple{Vector{Float64}, Vector{ComplexF64}}
    
    function apply_matH(x::Vector{ComplexF64})::Vector{ComplexF64}
        return matH * x
    end

    # sparse diagonalization
    krylovdim = max(krylovdimMin, krylovdimMultiplier*evNum)

    vals, vecs = eigsolve(apply_matH, sectorDim, evNum, :SR, ComplexF64, ishermitian=true, krylovdim=krylovdim)
    energies = real.(vals)

    p = sortperm(energies)
    energies_sorted = energies[p]
    vecs_sorted = vecs[p]

    ψ_gs = copy(vecs_sorted[1])
    if !isempty(ψ_gs)
        i0 = findfirst(x -> abs(x) > 0, ψ_gs)
        if i0 !== nothing
            ψ_gs .*= exp(-1im * angle(ψ_gs[i0]))
        end
    end

    return energies_sorted, ψ_gs
end


function SparseDiagonalization_GraphBLAS_WithVectors(sectorDim::Int64, matH::GBMatrix{ComplexF64, ComplexF64}, evNum::Int64;
    krylovdimMultiplier::Int64=16, krylovdimMin::Int64=256)::Tuple{Vector{Float64}, Vector{Vector{ComplexF64}}}

    function apply_matH(x::Vector{ComplexF64})::Vector{ComplexF64}
        return matH * x
    end

    # sparse diagonalization
    krylovdim = max(krylovdimMin, krylovdimMultiplier*evNum)

    vals, vecs = eigsolve(apply_matH, sectorDim, evNum, :SR, ComplexF64, ishermitian=true, krylovdim=krylovdim)

    p = sortperm(vals)
    sorted_vals = real.(vals[p])
    sorted_vecs = vecs[p]

    energies = sorted_vals[1:evNum]
    vectors = sorted_vecs[1:evNum]

    return energies, vectors
end


# sparse diagonalization (GraphBLAS)
# return lowest eigenvalues
function GS_energy_eigenFunc_GraphBLAS(sectorDim::Int64, matH::GBMatrix{ComplexF64, ComplexF64};
        krylovdimMultiplier::Int64=16, krylovdimMin::Int64=256)::Tuple{Float64, Vector{ComplexF64}}
    
    function apply_matH(x::Vector{ComplexF64})::Vector{ComplexF64}
        return matH * x
    end

    # sparse diagonalization
    krylovdim = max(krylovdimMin, krylovdimMultiplier * 1)


    energy, vec, _ = eigsolve(apply_matH, sectorDim, 1, :SR, ComplexF64, ishermitian=true, krylovdim=krylovdim)
    
    ψ_gs = Vector(vec[1])
    # if !isempty(ψ_gs)
    #     i0 = findfirst(x -> abs(x) > 0, ψ_gs)
    #     if i0 !== nothing
    #         ψ_gs .*= exp(-1im * angle(ψ_gs[i0]))
    #     end
    # end

    return real(energy[1]), ψ_gs
end
