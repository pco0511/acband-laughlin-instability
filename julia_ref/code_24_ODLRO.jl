include("code_22_ManyBodyQuantities.jl")


using CSV
using DataFrames
using Plots
using LaTeXStrings
using Colors



function Calculate_SC_PairDensityMatrix_1V(fl::FiniteLattice, sector_gs::Vector{BitVector}, ψ_gs::Vector{ComplexF64})::Matrix{ComplexF64}
    
    siteNum = fl.siteNum
    totNum = siteNum 
    ρ = zeros(ComplexF64, siteNum, siteNum)
    
    sqState = falses(totNum) 

    @showprogress "Calculating Pair Density matrix (1V)..." for ik in 0:siteNum-1
        imk = FL_ik_neg(fl, ik)

        for ikp in 0:siteNum-1
            imkp = FL_ik_neg(fl, ikp)
            
            # <GS| c†_{k'} c†_{-k'} c_{-k} c_{k} |GS>
            for idx_s in 0:length(sector_gs)-1
                sqState .= sector_gs[1+idx_s]

                sign_JW = 1.0                
                if sqState[1+ik]
                    if sum(@view sqState[1:ik]) % 2 == 1; sign_JW = -sign_JW; end
                    sqState[1+ik] = false
                    
                    if sqState[1+imk]
                        if sum(@view sqState[1:imk]) % 2 == 1; sign_JW = -sign_JW; end
                        sqState[1+imk] = false
                        
                        if !(sqState[1+imkp])
                            if sum(@view sqState[1:imkp]) % 2 == 1; sign_JW = -sign_JW; end
                            sqState[1+imkp] = true

                            if !(sqState[1+ikp])
                                if sum(@view sqState[1:ikp]) % 2 == 1; sign_JW = -sign_JW; end
                                sqState[1+ikp] = true

                                idx_final = searchsortedfirst(sector_gs, sqState)-1
                                if idx_final == length(sector_gs) || sector_gs[1+idx_final] != sqState
                                    println("**warning: applying two-body operator moved state outside symmetry sector**")
                                end
                                ρ[1+ik, 1+ikp] += sign_JW * conj(ψ_gs[1+idx_final]) * ψ_gs[1+idx_s]
                            end
                        end
                    end
                end
            end
        end
    end
    
    return ρ
end




# Calculating ODLRO
function Calculate_SC_PairDensityMatrix_2V(fl::FiniteLattice, sector_gs::Vector{BitVector}, ψ_gs::Vector{ComplexF64})::Matrix{ComplexF64}
    valleyNum = 2
    siteNum = fl.siteNum
    totNum = valleyNum*siteNum
    ρ = zeros(ComplexF64, siteNum, siteNum)

    sqState = falses(totNum)

    # SC_PairDensityMatrix
    
    @showprogress "Calculating Pair Density matrix..." for ik in 0:siteNum-1
        imk = FL_ik_neg(fl, ik)
        iuk = 0 + valleyNum*ik
        idmk = 1 + valleyNum*imk

        for ikp in 0:siteNum-1
            imkp = FL_ik_neg(fl, ikp)
            iukp = 0 + valleyNum*ikp
            idmkp = 1 + valleyNum*imkp


            # calculation of matrix element <GS| c†_{↑,k'} c†_{↓,-k'} c_{↓,-k} c_{↑,k} |GS>
            for idx_s in 0:length(sector_gs)-1
                sqState .= sector_gs[1+idx_s]

                # Applying annihilation Δ_k
                sign_JW = +1.0
                
                if !(sqState[1+iuk]) || !(sqState[1+idmk]); continue; end

                if sum(@view sqState[1:iuk])%2==1; sign_JW = -sign_JW; end
                sqState[1+iuk] = false
                if sum(@view sqState[1:idmk])%2==1; sign_JW = -sign_JW; end
                sqState[1+idmk] = false

                if sqState[1+iukp] || sqState[1+idmkp]; continue; end

                if sum(@view sqState[1:idmkp])%2==1; sign_JW = -sign_JW; end
                sqState[1+idmkp] = true

                if sum(@view sqState[1:iukp])%2==1; sign_JW = -sign_JW; end
                sqState[1+iukp] = true

                idx_final = searchsortedfirst(sector_gs, sqState) - 1

                if idx_final==length(sector_gs) || sector_gs[1+idx_final]!=sqState
                    println("**warning: applying two-body operator moved state outside symmetry sector**")
                    continue
                end
                
                ρ[1+ik,1+ikp] += sign_JW * conj(ψ_gs[1+idx_final]) * ψ_gs[1+idx_s]
                
            end
        end
    end

    return ρ
end





function plot_ODLRO_spectrum_custom(csv_path::String; 
                                    pdm_top::Int=6, 
                                    plot_title::String="ODLRO Spectrum",
                                    xlabel_str::String="d")
    
    if !isfile(csv_path)
        @error "CSV file not found at: $(csv_path)"
        return nothing
    end
    df = CSV.read(csv_path, DataFrame)

    p = plot(title=plot_title,
             xlabel=xlabel_str,
             legend=false,
             grid=true,
             framestyle=:box,
             tick_direction=:in,
             size = (800, 800),
            #  ylims = (0, 0.08)
             )

    K_values = df[!, 1]

    for j in 1:pdm_top
        lambda_col = Symbol("lambda$(j)")
        
        if !hasproperty(df, lambda_col)
            @warn "Column $(lambda_col) not found in CSV. Skipping."
            continue
        end
        
        lambda_values = df[!, lambda_col] / 12
        
        if j == 1
            # Largest eigenvalue
            plot!(p, K_values, lambda_values,
                  seriestype=:path,    # marker, line
                  linestyle=:dash,     # dashed line
                  color=:red,
                  marker=:square,
                  markersize=3,
                  linewidth=2)
        else
            # Other eigenvalues
            # only for second largest eigenvalue, show label
            label_str = (j == 2) ? L"\lambda_{>1}" : ""

            plot!(p, K_values, lambda_values,
                  seriestype=:scatter,
                  color=:gray,
                  marker=:x,
                  markersize=3)
        end
    end

    return p
end



csv_path = "(K_Scan)_1LL_ODLRO_1V_N12_nu0.50_d0.30.csv"
# csv_path = "(d_Scan)_1LL_ODLRO_V_N12_nu1.00_K0.12.csv"
my_plot = plot_ODLRO_spectrum_custom(csv_path, pdm_top=12, plot_title="ODLRO Spectrum (N=12, K=0.12)")

display(my_plot)


savefig(my_plot, "odlro_spectrum_plot.png")


#=

Example usage:

csv_path = "(K_Scan)_1LL_ODLRO_1V_N12_nu0.50_d0.30.csv"
# csv_path = "(d_Scan)_1LL_ODLRO_V_N12_nu1.00_K0.12.csv"
my_plot = plot_ODLRO_spectrum_custom(csv_path, pdm_top=12, plot_title="ODLRO Spectrum (N=12, K=0.12)")

display(my_plot)


savefig(my_plot, "odlro_spectrum_plot.png")


=#