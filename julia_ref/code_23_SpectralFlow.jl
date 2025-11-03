# packages
using LinearAlgebra, SparseArrays, Printf
using CSV, DataFrames
import DataFrames: select!
using SuiteSparseGraphBLAS
using Plots
using Statistics
using Colors
using ProgressMeter
using JLD2

include("code_00_Lattice.jl")
include("code_01_FiniteLattice.jl")
include("code_02_FormFactors.jl")
include("code_10_InteractionMatrix.jl")
include("code_20_ManyBodyHamiltonian.jl")
include("code_21_Diagonalization.jl")


# Scan d for fixed K and save each state to a JLD2 file
function scan_d_and_save_jld2(; u11, u12, u21, u22,
                                Qd::Int64, d_grid = 0.0:0.05:3.0,
                                VC::Float64, K_param::Float64,
                                elecNum_u::Int64, elecNum_d::Int64,
                                LLIdx::Int64
                                )

    tl = TriangularLattice()
    fl = FiniteLattice(tl, u11, u12, u21, u22)
    Qs = FiniteReciprocalLattice(tl, Qd)
    @time sectors = SymmetrySectors_2V_M(fl, elecNum_u, elecNum_d)
    siteNum = fl.siteNum
    

    output_dir = @sprintf("output/Nh-%02d_Nhu-%02d/", siteNum, elecNum_u)
    mkpath(output_dir)
    @printf("Results will be saved in: %s\n", output_dir)

    k_points_to_calculate = [0, 1, 2, 3, 4, 6, 7, 8, 9]
    for d_param in d_grid
        Int_parms = Interaction_Parms(0.0, 0.0, VC, d_param)
        @printf("[d=%.2f] building FF/H ... ", d_param); flush(stdout)
        ff = FormFactors_IdealChernBand_2V(fl, Qs, K_param, LLIdx)
        H4 = InteractionMatrix_IdealChernBand_2V_fromFF(fl, Qs, ff, Int_parms)
        println("OK")

        @showprogress "Diagonalizing K sectors" for itK in k_points_to_calculate
            if itK >= siteNum; continue; end

            sector = sectors[1+itK]
            H = ManyBodyHamiltonian_K_2V1B_2B_modified(fl, H4, sector)
            dim = length(sector)
            if dim == 0; continue; end
            local_nev_per_sector = 0
            
            if itK == 0 || itK == 7
                local_nev_per_sector = 20
            else
                local_nev_per_sector = 20
            end
            nev = min(local_nev_per_sector, dim)
            
            evals, evecs = SparseDiagonalization_GraphBLAS_WithVectors(dim, GBMatrix(H), nev)

            for j in 1:nev
                E = evals[j]
                Ψ = evecs[j]
                id_val = round(Int64, 100.0 * d_param)
                fileName = @sprintf("K_parm-%0.2f_id-%03d_Nh-%02d_Nhu-%02d_iK-%02d_%02d.jld2",
                                    K_param, id_val, siteNum, elecNum_u, itK, j-1)
                full_path = joinpath(output_dir, fileName)

                jldopen(full_path, "w") do file
                    file["E"] = E
                    file["Ψ"] = Ψ
                    file["d"] = d_param
                    file["K"] = itK
                end
            end
        end
    end
    
    @printf("Finished saving all results to %s\n", output_dir)
end


# Flux insertion and spectral flow
function SpectralFlow_2V(;u11::Int, u12::Int, u21::Int, u22::Int,
    evNum::Int, Φ_list::Vector{Float64}, Qd::Int64,
    elecNum_u::Int, elecNum_d::Int, K_param::Float64,
    LLIdx::Int, Int_parms::Interaction_Parms, ref_level_index::Int,
    trs_twist::Bool=true)::Matrix{Float64}

    tl = TriangularLattice()
    Qs = FiniteReciprocalLattice(tl, Qd)
    fl0 = FiniteLattice(tl, u11, u12, u21, u22)
    L1 = fl0.L1
    sliceNum = length(Φ_list)
    siteNum = fl0.siteNum
    spectralFlow = zeros(Float64, sliceNum, evNum)

    # sectors = SymmetrySectors_2V(fl0, elecNum)
    # sector_M = sectors[1+M]


    for (islice, α) in enumerate(Φ_list)
        # @printf("α = %.2f (Φ/2π) ... ", α / L1); flush(stdout)
        @printf("α = %.2f (Φ/2π) ... \n", α); flush(stdout)
        fl = FiniteLattice(tl, u11, u12, u21, u22; isTRS=trs_twist, α1=α, α2=0.0)

        # println("Generating symmetry sectors...")
        @time sector_M = SymmetrySectors_2V_M(fl, elecNum_u, elecNum_d)
        # println("OK")

        # println("Building FF...")
        @time ff = FormFactors_IdealChernBand_2V(fl, Qs, K_param, LLIdx)
        # println("OK")

        # println("Building H...")
        @time H4 = InteractionMatrix_IdealChernBand_2V_fromFF(fl, Qs, ff, Int_parms)
        # println("OK")

        # println("Start diagonalization...")
        all_evals = Float64[]
        @showprogress "Diagonalizing..." for itK in 0:siteNum-1
            sector = sector_M[1+itK]
            dim = length(sector); if dim == 0; continue; end
            H = ManyBodyHamiltonian_K_2V1B_2B_modified(fl, H4, sector)
            nev = min(evNum, dim); if nev == 0; continue; end
            evals = SparseDiagonalization_GraphBLAS(dim, GBMatrix(H), nev)
            append!(all_evals, evals)
        end

        if !isempty(all_evals)
            sort!(all_evals)
            n = min(evNum, length(all_evals))
            spectralFlow[islice, 1:n] .= all_evals[1:n]
        end
    end
    return spectralFlow
end




# (AC band) K sweeping

function scan_K_2V(;u11, u12, u21, u22,
                elecNum::Int64, Qd::Int64, K_grid = 0.0:0.05:3.0,
                Int_parms::Interaction_Parms,
                nev_per_sector::Int64 = 10,
                sector_base::Int64 = 0, M::Int64 = 0, LLIdx::Int64)

    tl = TriangularLattice()
    fl = FiniteLattice(tl, u11, u12, u21, u22)
    Qs = FiniteReciprocalLattice(tl, Qd)
    @time sectors = SymmetrySectors_2V(fl, elecNum)
    siteNum = fl.siteNum
    ν = elecNum / siteNum

    @printf("Raw scan: sectors 1..%d, K ∈ [%.2f, %.2f] step %.2f, nev/sector=%d\n",
            siteNum, first(K_grid), last(K_grid), step(K_grid), nev_per_sector)

    csvname = @sprintf("%dLL_N%d_nu%.2f_d%.2f.csv", LLIdx, siteNum, ν, Int_parms.d)
    open(csvname, "w") do io
        println(io, "K,sector,level,E")

        for K_param in K_grid
            @printf("[K=%.2f] building FF/H ... ", K_param); flush(stdout)
            ff = FormFactors_IdealChernBand_2V(fl, Qs, K_param, LLIdx)
            H4 = InteractionMatrix_IdealChernBand_2V_fromFF(fl, Qs, ff, Int_parms)
            println("OK")

            @showprogress "asdf" for itK in 0:siteNum-1
                sector_M = sectors[1+M]
                sector = sector_M[1+itK]
                dim = length(sector)
                if dim == 0; continue; end
                @printf("--- K : %d, Dimension : %d --- \n", itK, dim)


                H = ManyBodyHamiltonian_K_2V1B_2B_modified(fl, H4, sector)
                nev = min(nev_per_sector, dim)
                evals = SparseDiagonalization_GraphBLAS(dim, GBMatrix(H), nev)

                sort!(evals)

                write_sector_idx = sector_base == 1 ? itK : (itK - 1)

                for (j, E) in enumerate(evals)
                    @printf(io, "%.6f,%d,%d,%.12g\n", K_param, itK, j, E)
                end
            end
        end
    end
    
    @printf("Saved RAW CSV as %s\n", csvname)

    return csvname, siteNum, ν
end



function scan_K_1V(;u11, u12, u21, u22,
                elecNum::Int64, Qd::Int64, K_grid = 0.0:0.05:3.0,
                Int_parms::Interaction_Parms,
                nev_per_sector::Int64 = 10,
                sector_base::Int64 = 0, LLIdx::Int64)

    tl = TriangularLattice()
    fl = FiniteLattice(tl, u11, u12, u21, u22)
    Qs = FiniteReciprocalLattice(tl, Qd)
    @time sector = SymmetrySectors_1V(fl, elecNum)
    siteNum = fl.siteNum
    ν = elecNum / siteNum

    @printf("Raw scan: sectors 1..%d, K ∈ [%.2f, %.2f] step %.2f, nev/sector=%d\n",
            siteNum, first(K_grid), last(K_grid), step(K_grid), nev_per_sector)

    csvname = @sprintf("%dLL_N%d_nu%.2f_d%.2f.csv", LLIdx, siteNum, ν, Int_parms.d)
    open(csvname, "w") do io
        println(io, "K,sector,level,E")

        for K_param in K_grid
            @printf("[K=%.2f] building FF/H ... ", K_param); flush(stdout)
            ff = FormFactors_IdealChernBand_1V(fl, Qs, K_param, LLIdx)
            H4 = InteractionMatrix_IdealChernBand_1V_fromFF(fl, Qs, ff, Int_parms)
            println("OK")

            for itK in 0:siteNum-1
                dim = length(sector[1+itK])
                if dim == 0; continue; end
                @printf("--- K : %d, Dimension : %d --- \n", itK, dim)


                H = ManyBodyHamiltonian_K_1V1B_2B_modified(fl, H4, sector[1+itK])
                nev = min(nev_per_sector, dim)
                evals = SparseDiagonalization_GraphBLAS(dim, GBMatrix(H), nev)

                sort!(evals)

                write_sector_idx = sector_base == 1 ? itK : (itK - 1)

                for (j, E) in enumerate(evals)
                    @printf(io, "%.6f,%d,%d,%.12g\n", K_param, itK, j, E)
                end
            end
        end
    end
    
    @printf("Saved RAW CSV as %s\n", csvname)

    return csvname, siteNum, ν
end



# Scan d for fixed K
function scan_d(;u11, u12, u21, u22,
                Qd::Int64, d_grid = 0.0:0.05:3.0,
                VC::Float64, K_param::Float64,
                nev_per_sector::Int64, 
                elecNum_u::Int64, elecNum_d::Int64, LLIdx::Int64)

    tl = TriangularLattice()
    fl = FiniteLattice(tl, u11, u12, u21, u22)
    Qs = FiniteReciprocalLattice(tl, Qd)
    @time sectors = SymmetrySectors_2V_M(fl, elecNum_u, elecNum_d)
    siteNum = fl.siteNum
    ν = (elecNum_d+elecNum_u) / siteNum

    

    @printf("Raw scan: sectors 1..%d, d ∈ [%.2f, %.2f] step %.2f, nev/sector=%d\n",
            siteNum, first(d_grid), last(d_grid), step(d_grid), nev_per_sector)

    csvname = @sprintf("%dLL_N%d_nu%.2f_K%.2f.csv", LLIdx, siteNum, ν, K_param)
    open(csvname, "w") do io
        println(io, "d,sector,level,E")

        for d_param in d_grid
            Int_parms = Interaction_Parms(0.0, 0.0, VC, d_param)
            @printf("[d=%.2f] building FF/H ... ", d_param); flush(stdout)
            ff = FormFactors_IdealChernBand_2V(fl, Qs, K_param, LLIdx)
            H4 = InteractionMatrix_IdealChernBand_2V_fromFF(fl, Qs, ff, Int_parms)
            println("OK")

            @showprogress "Diagonalizing K symmetry sectors" for itK in 0:siteNum-1
                sector = sectors[1+itK]
                dim = length(sector)
                if dim == 0; continue; end
                # @printf("--- itK : %d, Dimension : %d --- \n", itK, dim)


                H = ManyBodyHamiltonian_K_2V1B_2B_modified(fl, H4, sector)
                nev = min(nev_per_sector, dim)
                evals = SparseDiagonalization_GraphBLAS(dim, GBMatrix(H), nev)
                sort!(evals)

                for (j, E) in enumerate(evals)
                    @printf(io, "%.6f,%d,%d,%.12g\n", d_param, itK, j, E)
                end
            end
        end
    end
    
    @printf("Saved RAW CSV as %s\n", csvname)

    return csvname
end