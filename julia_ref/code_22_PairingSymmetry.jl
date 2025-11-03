# packages
include("code_00_Lattice.jl")
include("code_01_FiniteLattice.jl")
include("code_02_FormFactors.jl")
include("code_10_InteractionMatrix.jl")
include("code_20_ManyBodyHamiltonian.jl")
include("code_21_Diagonalization.jl")

using ProgressMeter
using Plots
pyplot()

function get_brillouin_zone_vertices(tl::Lattice)
    a = 1.0
    b1 = tl.Q1
    b2 = tl.Q2
    b3 = -(b1 + b2)
    K1 = (2*b1 + b2) / 3
    K2 = (b1 + 2*b2) / 3
    K3 = (b2 - b1) / 3
    K4 = -K1
    K5 = -K2
    K6 = -K3
    
    vertices_matrix = hcat(K1, K2, K3, K4, K5, K6)'
    return vertices_matrix
end


function map_to_FBZ(k_point::Vector{Float64}, tl::Lattice)
    b1 = tl.Q1
    b2 = tl.Q2

    min_dist_sq = dot(k_point, k_point)
    k_mapped = k_point

    for n1 in -2:2
        for n2 in -2:2
            Q = n1 * b1 + n2 * b2
            k_shifted = k_point - Q
            dist_sq = dot(k_shifted, k_shifted)

            if dist_sq < min_dist_sq
                min_dist_sq = dist_sq
                k_mapped = k_shifted
            end
        end
    end
    
    return k_mapped
end


function create_fbz_pairing_plot(fl::FiniteLattice, data::Vector{ComplexF64}, title_str::String, fg_size=(500,400); is_colorbar::Bool=true)
    mapped_coords_vec = map(k -> map_to_FBZ(k, fl.lattice), fl.BZ0)
    coords = hcat(mapped_coords_vec...)'

    phase_factor_gamma = (abs(data[1]) > 1e-9) ? (data[1] / abs(data[1])) : 1.0 + 0.0im
    data_rotated = data ./ phase_factor_gamma
    magnitudes = abs.(data_rotated)
    phases = angle.(data_rotated)


    max_mag = maximum(magnitudes)
    marker_sizes = (max_mag > 1e-12) ? (3 .+ 15 * (magnitudes ./ max_mag)) : 3.0


    bz_vertices = get_brillouin_zone_vertices(fl.lattice)
    bz_boundary = vcat(bz_vertices, bz_vertices[1:1, :])

    p = plot(
        aspect_ratio = :equal,
        legend = false,
        framestyle = :box,
        title = title_str,
        xlabel = "kx",
        ylabel = "ky",
        size = fg_size
    )

    # BZ boundary
    plot!(p, bz_boundary[:, 1], bz_boundary[:, 2],
          seriestype = :path,
          linewidth = 2,
          linecolor = :black,
          label = "")

    # Data points with phase color mapping
    scatter!(p, coords[:, 1], coords[:, 2],
        marker_z = phases,
        markersize = marker_sizes,
        color = :hsv, # Use HSV colormap for phase representation
        markerstrokewidth = 0.5,
        markerstrokecolor = :black,
        colorbar = is_colorbar,
        clims = (-π, π),
        colorbar_ticks = ([-π, -2*π/3, -π/3, 0, π/3, 2*π/3, π], ["-π", "-2π/3", "-π/3", "0", "π/3", "2π/3", "π"])
    )
    
    return p
end


function fix_gauge!(Ψ::Vector{ComplexF64})
    if isempty(Ψ) return end
    max_idx = argmax(abs.(Ψ))
    phase_factor = Ψ[max_idx] / abs(Ψ[max_idx])
    Ψ ./= phase_factor
end



function plot_PairingSymmetry_FBZ(fl::FiniteLattice, elecNum::Int, M::Int, K_param::Float64, Qd::Int,
                                LLIdx::Int, Int_parms::Interaction_Parms; nev::Int=1, eps::Float64=1.0e-9)
    
    siteNum = fl.siteNum
    Qs = FiniteReciprocalLattice(tl, Qd)
    ff = FormFactors_IdealChernBand_2V(fl, Qs, K_param, LLIdx)
    H4 = InteractionMatrix_IdealChernBand_2V_fromFF(fl, Qs, ff, Int_parms)

    E_000, Ψ_000, sector_000_gs, itK_000 = find_global_ground_state(fl, H4, elecNum, M)
    # fix_gauge!(Ψ_000)

    E_n00, Ψ_n00, sector_n00_gs, itK_n00 = find_global_ground_state(fl, H4, elecNum - 2, M-1)
    # fix_gauge!(Ψ_n00)

    E_p00, Ψ_p00, sector_p00_gs, itK_p00 = find_global_ground_state(fl, H4, elecNum + 2, M+1)
    # fix_gauge!(Ψ_p00)

    if !(itK_000 == itK_n00 == itK_p00)
        @warn "Momentum sector mismatch GS(N_e) K=$(itK_000), GS(N_e-2) K=$(itK_n00), GS(N_e+2) K=$(itK_p00)"
    end
    if isempty(Ψ_000) || isempty(Ψ_n00) || isempty(Ψ_p00); @error "fail to find GS"; return; end


    println("\nCalculating pairing symmetry...")
    Δs, Δcs = PairingSymmetry(fl, sector_000_gs, Ψ_000, sector_n00_gs, Ψ_n00, sector_p00_gs, Ψ_p00)

    println("Plotting result...")
    p1 = create_fbz_pairing_plot(fl, Δs, "Pairing Gap Function: Δ(k)")
    p2 = create_fbz_pairing_plot(fl, Δcs, "Pair Creation Function: Δ†(k)")

    final_plot = plot(p1, p2, layout=(1, 2), size=(1000, 400), 
                      plot_title="Pairing Symmetry in FBZ (K_param = $K_param, K_gs=$itK_000)")
    display(final_plot)
    
    return Δs, Δcs, final_plot
end


function find_global_ground_state(fl::FiniteLattice, H4, elecNum::Int, M::Int)
    siteNum = fl.siteNum
    
    E_gs = Inf
    Ψ_gs = Vector{ComplexF64}()
    sector_gs = Vector{BitVector}()
    itK_gs = -1

    println("--- Finding global GS begins (N_e = $elecNum) ---")
    sectors = SymmetrySectors_2V(fl, elecNum)[1+M]

    @showprogress "Scanning K sectors for N_e=$elecNum..." for itK in 0:siteNum-1
        current_sector = sectors[1+itK]
        dim = length(current_sector)
        if dim == 0; continue; end

        H = ManyBodyHamiltonian_K_2V1B_2B_modified(fl, H4, current_sector)
        E_local, Ψ_local = GS_energy_eigenFunc_GraphBLAS(dim, GBMatrix(H))
        
        if E_local < E_gs
            E_gs = E_local
            Ψ_gs = Ψ_local
            sector_gs = current_sector
            itK_gs = itK
        end
    end

    if itK_gs == -1
        @warn "No GS at N_e=$elecNum."
    else
        println(">>> N_e=$elecNum has global GS at K=$(itK_gs), E0=$(E_gs)")
    end
    
    return E_gs, Ψ_gs, sector_gs, itK_gs
end



# pairing symmetry by comparing |GS(Ne)> with |GS(Ne-2)> and |GS(Ne+2)>
## wavefunctions are all computed in this function to fix gauge
function PairingSymmetry(fl::FiniteLattice,
        sector_2V_000::Vector{BitVector}, Ψ_000::Vector{ComplexF64},
        sector_2V_n00::Vector{BitVector}, Ψ_n00::Vector{ComplexF64},
        sector_2V_p00::Vector{BitVector}, Ψ_p00::Vector{ComplexF64})::Tuple{Vector{ComplexF64}, Vector{ComplexF64}}

    valleyNum = 2
    siteNum = fl.siteNum
    totNum = valleyNum*siteNum


    
    # pairing gap function
    Δs = zeros(ComplexF64, siteNum)
    Δcs = zeros(ComplexF64, siteNum)
    sqState = zeros(Bool, totNum)
    @showprogress for ik in 0:siteNum-1

        imk = FL_ik_neg(fl, ik)
        
        iuk  = 0 + valleyNum*ik
        idmk = 1 + valleyNum*imk

        # Δ[k] = < N-2, GS | ψ[↓,-k] ψ[↑,k] | N, GS >
        for idx_000 in 0:length(sector_2V_000)-1
            
            sqState .= sector_2V_000[1+idx_000]

            sign_JW = +1
            if sqState[1+iuk]
                if sum(@view sqState[1:iuk])%2==1; sign_JW = -sign_JW; end
                sqState[1+iuk] = false

                if sqState[1+idmk]
                    if sum(@view sqState[1:idmk])%2==1; sign_JW = -sign_JW; end
                    sqState[1+idmk] = false

                    idx_n00 = searchsortedfirst(sector_2V_n00, sqState)-1
                    if idx_n00==length(sector_2V_n00) || sector_2V_n00[1+idx_n00]!=sqState
                        println("**warning: applying two-body operator moved state outside symmetry sector**")
                    end
    
                    Δs[1+ik] += sign_JW * conj(Ψ_n00[1+idx_n00]) * Ψ_000[1+idx_000]
                end
            end
        end

        
        # Δ*[k] = < N+2, GS | ψ[↑,k]† ψ[↓,-k]† | N, GS >
        for idx_000 in 0:length(sector_2V_000)-1
            
            sqState .= sector_2V_000[1+idx_000]

            sign_JW = +1
            if !sqState[1+idmk]
                if sum(@view sqState[1:idmk])%2==1; sign_JW = -sign_JW; end
                sqState[1+idmk] = true

                if !sqState[1+iuk]
                    if sum(@view sqState[1:iuk])%2==1; sign_JW = -sign_JW; end
                    sqState[1+iuk] = true

                    idx_p00 = searchsortedfirst(sector_2V_p00, sqState)-1
                    if idx_p00==length(sector_2V_p00) || sector_2V_p00[1+idx_p00]!=sqState
                        println("**warning: applying two-body operator moved state outside symmetry sector**")
                    end
    
                    Δcs[1+ik] += sign_JW * conj(Ψ_p00[1+idx_p00]) * Ψ_000[1+idx_000]
                end
            end
        end
    end
    
    return Δs, Δcs
end