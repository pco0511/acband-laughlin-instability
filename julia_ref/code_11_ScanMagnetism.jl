include("code_00_Lattice.jl")
include("code_01_FiniteLattice.jl")
include("code_02_FormFactors.jl")
include("code_10_InteractionMatrix.jl")
include("code_20_ManyBodyHamiltonian.jl")
include("code_21_Diagonalization.jl")



using Printf
using Plots



function Energy_correction(fl::FiniteLattice, Param::Interaction_Parms, elecNum::Int64, d::Float64, M::Int64)::Float64
    ℓB2 = (fl.lattice.ℓB)^2
    U = Param.VC
    siteNum = fl.siteNum
    ν = elecNum / siteNum

    m = (M - (elecNum - M)) / (2 * elecNum)

    dE = elecNum / ℓB2 * U * d * ν * m^2

    return dE
end


function scan_Magnetism(fl::FiniteLattice, Qs::Vector{Vector{Float64}}, elecNum::Int64, K_param::Float64, Int_parms::Interaction_Parms, LLIdx::Int64)
    
    siteNum = fl.siteNum
    ν = elecNum / siteNum

    M_values = Int64[]
    E_values = Float64[]

    @printf("Magnetism scan: K=%.2f, elecNum=%d, siteNum=%d\n", K_param, elecNum, siteNum)
    @printf("Pre-calculating FormFactors and InteractionMatrix for K=%.2f ... ", K_param); flush(stdout)
    ff = FormFactors_IdealChernBand_2V(fl, Qs, K_param, LLIdx)
    H4 = InteractionMatrix_IdealChernBand_2V_fromFF(fl, Qs, ff, Int_parms)
    println("Done.")
    tot_sectors = SymmetrySectors_2V(fl, elecNum)
    for M in 0:Int(elecNum/2)
        @printf("\n--- Scanning M = %d ---\n", M)
        sectors_for_M = tot_sectors[1+M]
        min_E_for_M = Inf

        for itK in 0:siteNum-1
            sector = sectors_for_M[1+itK]
            dim = length(sector)

            if dim == 0
                continue
            end
            
            H = ManyBodyHamiltonian_K_2V1B_2B_modified(fl, H4, sector)
            evals = SparseDiagonalization_GraphBLAS(dim, GBMatrix(H), 1)
            ground_E_sector = evals[1]
            
            if ground_E_sector < min_E_for_M
                min_E_for_M = ground_E_sector
            end
        end

        if isinf(min_E_for_M)
             @printf("  No valid states found for M = %d. Skipping.\n", M)
            continue
        end

        d = Int_parms.d
        correction = Energy_correction(fl, Int_parms, elecNum, d, M)
        final_E = min_E_for_M + correction
        
        @printf("-> Min E for M=%d is %.8f (raw) + %.8f (corr) = %.8f\n", M, min_E_for_M, correction, final_E)
        push!(M_values, M)
        push!(E_values, final_E)
    end
    
    if isempty(M_values)
        println("No data to plot.")
        return
    end

    plot(M_values, E_values, 
        xlabel="M (Number of electrons in one valley)", 
        ylabel="Corrected Ground State Energy",
        title=@sprintf("Energy vs. Magnetization (K=%.2f, N=%d, ν=%.2f)", K_param, elecNum, ν),
        legend=false,
        marker=:circle,
        markersize=4,
        linewidth=2
    )
    
    display(current()) 
    
    min_E_total, min_idx = findmin(E_values)
    M_ground = M_values[min_idx]

    @printf("Scan finished. Overall ground state energy is %.8f at M = %d.\n", min_E_total, M_ground)

    return M_values, E_values
end