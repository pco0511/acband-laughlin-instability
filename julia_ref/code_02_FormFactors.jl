# packages
include("code_01_FiniteLattice.jl")



# eta_g factor from Eq. (A13)
# g = n * Q1 + m * Q2
function EtaFactor(fl::FiniteLattice, g::Vector{Float64}; eps::Float64=1.0e-9)::Int64
    n = round(Int64, dot(g, fl.lattice.a1)/(2*π))
    m = round(Int64, dot(g, fl.lattice.a2)/(2*π))
    if norm(g - n*fl.lattice.Q1 - m*fl.lattice.Q2)>eps
        println("**warning: incorrect decomposition of g into Q1 and Q2**")
        return 0
    end
    return ((n + m + n * m) % 2 == 0) ? +1 : -1
end


function calculate_wg_FFT(fl::FiniteLattice, Qs::Vector{Vector{Float64}}, K_param::Float64; grid_res::Int=128)::Vector{ComplexF64}

    Q1 = fl.lattice.Q1
    Q2 = fl.lattice.Q2
    Q3 = -Q1 - Q2
    a1, a2 = fl.lattice.a1, fl.lattice.a2


    # uniform param grid over the UC: r(u,v) = u a1 + v a2, u,v ∈ [0,1)
    r_grid = [(i/grid_res)*a1 + (j/grid_res)*a2 for i in 0:grid_res-1, j in 0:grid_res-1]


    # compute K(r)
    pref = -sqrt(3) * K_param / (2π)
    fvals = Array{Float64}(undef, grid_res, grid_res)
    for i in 0:grid_res-1, j in 0:grid_res-1
        r = r_grid[1+i,1+j]
        K_r = pref * (cos(dot(Q1,r)) + cos(dot(Q2,r)) + cos(dot(Q3,r)))
        fvals[1+i,1+j] = exp(-2 * K_r)   # |B(r)|^2
    end

    gNum = length(Qs)
    w_g = zeros(ComplexF64, gNum)

    # compute w_g = Σ_r B(r) e^{i g·r}
    for ig in 0:gNum-1
        g = Qs[1+ig]
        s = 0.0 + 0.0im
        for i in 1:grid_res, j in 1:grid_res
            r = r_grid[i,j]
            s += fvals[i,j] * exp(1im * dot(g, r))
        end
        w_g[1+ig] = s / (grid_res^2)
    end

    return w_g
end



# Ideal Chern Band Form Factor (Eq. A8, A13, A14)
function FormFactors_IdealChernBand_1V(fl::FiniteLattice, Qs::Vector{Vector{Float64}}, K_param::Float64, LLIdx::Int64; eps::Float64=1.0e-9)
    siteNum = fl.siteNum
    gNum = length(Qs)
    @assert norm(Qs[1]) < 1e-12  # expect g=0 at index 1

    # precompute
    ℓB2 = (fl.lattice.ℓB)^2
    w_g = calculate_wg_FFT(fl, Qs, K_param)

    # (A14) normalization N_k
    Nk = zeros(Float64, siteNum) # real/positive
    for ik in 0:siteNum-1
        k = fl.BZu[1+ik]
        s = 0.0
        @inbounds for ig in 0:gNum-1
            g = Qs[1+ig]
            η = EtaFactor(fl, g)
            phase = ℓB2 * cross([k;0.0], [g;0.0])[3]  # k×g
            gauss = exp(-ℓB2 * dot(g,g) / 4)
            s += real(w_g[1+ig] * η * exp(1im*phase)) * gauss
        end
        Nk[1+ik] = inv(sqrt(s))
    end

    # (A13) f^{g}_{k,p}
    ff_LL = [zeros(ComplexF64, siteNum, siteNum) for _ in 1:gNum]
    for ik in 0:siteNum-1, ip in 0:siteNum-1, ig in 0:gNum-1
        k = fl.BZu[1+ik]; p = fl.BZu[1+ip]
        g = Qs[1+ig]
        P = k - p - g
        Pabs2 = ℓB2 * dot(P, P) / 2
        η = EtaFactor(fl, g)
        phase1 = ℓB2 * cross([k+p;0.0], [g;0.0])[3] / 2      # (k+p)×g /2
        phase2 = ℓB2 * cross([k;0.0], [p;0.0])[3] / 2        # k×p /2

        gauss  = exp(-Pabs2/2)
        if LLIdx == 0
            gauss *= 1
        elseif LLIdx == 1
            gauss *= (1 - Pabs2)
        elseif LLIdx == 2
            gauss *= (1/2) * (2 - 4*Pabs2 + Pabs2^2)
        end
        
        ff_LL[1+ig][1+ik, 1+ip] = η * exp(1im*(phase1 + phase2)) * gauss
    end

    # (A8) Λ_{k,p+g} = N_k N_p Σ_{g'} w_{g'} f^{g+g'}_{k,p}
    ff_Lambda = [zeros(ComplexF64, siteNum, siteNum) for _ in 1:gNum]
    for ik in 0:siteNum-1, ip in 0:siteNum-1, ig in 0:gNum-1
        Nkfac = Nk[1+ik] * Nk[1+ip]
        for igp in 0:gNum-1
            gsum_vec = Qs[1+ig] + Qs[1+igp]
            igsum = RL_QtoiQ(fl.lattice, gsum_vec)   # map back to index
            # println(gNum)
            # println(igsum)
            if igsum < 0 || igsum >= gNum
                # g+g' is not in the truncated set
                continue
            end
            ff_Lambda[1+ig][1+ik,1+ip] += Nkfac * w_g[1+igp] * ff_LL[1+igsum][1+ik,1+ip]
        end
    end

    return ff_Lambda
end



function FormFactors_IdealChernBand_2V(fl::FiniteLattice, Qs::Vector{Vector{Float64}}, K_parm::Float64, LLIdx::Int64; eps::Float64=1.0e-9)::Vector{Vector{Matrix{ComplexF64}}}

    valleyNum = 2
    siteNum = fl.siteNum
    gNum = length(Qs)

    @assert norm(Qs[1]) < 1e-12  # expect g=0 at index 1

    # precompute
    ℓB2 = (fl.lattice.ℓB)^2
    w_g = calculate_wg_FFT(fl, Qs, K_parm)

    # (A14) normalization N_k
    Nk = [zeros(ComplexF64, siteNum) for is in 0:valleyNum-1] # real/positive
    for ik in 0:siteNum-1
        k = fl.BZu[1+ik]
        s = 0.0
        @inbounds for ig in 0:gNum-1
            g = Qs[1+ig]
            η = EtaFactor(fl, g)
            phase = ℓB2 * cross([k;0.0], [g;0.0])[3]  # k×g
            gauss = exp(-ℓB2 * dot(g,g) / 4)
            s += w_g[1+ig] * η * exp(+1im*phase) * gauss
        end
        Nk[1+0][1+ik] = s^(-0.5)
    end

    for ik in 0:siteNum-1
        k = fl.BZd[1+ik]
        s = 0.0
        @inbounds for ig in 0:gNum-1
            g = Qs[1+ig]
            η = EtaFactor(fl, g)
            phase = ℓB2 * cross([k;0.0], [g;0.0])[3]  # k×g
            gauss = exp(-ℓB2 * dot(g,g) / 4)
            s += w_g[1+ig] * η * exp(+1im*phase) * gauss
        end
        Nk[1+1][1+ik] = s^(-0.5)
    end


    # (A13) f^{g}_{k,p}
    ff_LL = [[zeros(ComplexF64, siteNum, siteNum) for _ in 1:gNum] for is in 0:valleyNum-1]


    # K valley (s = 0)
    for ik in 0:siteNum-1, ip in 0:siteNum-1, ig in 0:gNum-1
        k = fl.BZu[1+ik]; p = fl.BZu[1+ip]
        g = Qs[1+ig]
        P = k - p - g
        Pabs2 = ℓB2 * dot(P, P) / 2
        η = EtaFactor(fl, g)
        phase1 = ℓB2 * cross([k+p;0.0], [g;0.0])[3] / 2      # (k+p)×g /2
        phase2 = ℓB2 * cross([k;0.0], [p;0.0])[3] / 2        # k×p /2
        gauss  = exp(-Pabs2/2)
        
        if LLIdx == 0
            gauss *= 1
        elseif LLIdx == 1
            gauss *= (1 - Pabs2)
        elseif LLIdx == 2
            gauss *= (1/2) * (2 - 4*Pabs2 + Pabs2^2)
        end

        ff_LL[1+0][1+ig][1+ik, 1+ip] = η * exp(1im*(phase1 + phase2)) * gauss
    end


    # K' valley (s = 1)
    for ik in 0:siteNum-1, ip in 0:siteNum-1, ig in 0:gNum-1
        k = fl.BZd[1+ik]; p = fl.BZd[1+ip]
        g = Qs[1+ig]
        P = k - p - g
        Pabs2 = ℓB2 * dot(P, P) / 2
        η = EtaFactor(fl, g)
        phase1 = ℓB2 * cross([k+p;0.0], [g;0.0])[3] / 2      # (k+p)×g /2
        phase2 = ℓB2 * cross([k;0.0], [p;0.0])[3] / 2        # k×p /2
        gauss  = exp(-Pabs2/2)
        
        if LLIdx == 0
            gauss *= 1
        elseif LLIdx == 1
            gauss *= (1 - Pabs2)
        elseif LLIdx == 2
            gauss *= (1/2) * (2 - 4*Pabs2 + Pabs2^2)
        end

        ff_LL[1+1][1+ig][1+ik, 1+ip] = η * exp(-1im*(phase1 + phase2)) * gauss
    end

    ff_Λ = [[zeros(ComplexF64, siteNum, siteNum) for _ in 1:gNum] for is in 0:valleyNum-1]
    for ik in 0:siteNum-1, ip in 0:siteNum-1, ig in 0:gNum-1
        Nk_0 = Nk[1+0][1+ik] * Nk[1+0][1+ip]
        Nk_1 = Nk[1+1][1+ik] * Nk[1+1][1+ip]
        # Nkfac_down = Nk_2V[1+1][1+ik] * Nk_2V[1+1][1+ip]
        for igp in 0:gNum-1
            gsum_vec = Qs[1+ig] + Qs[1+igp]
            igsum = RL_QtoiQ(fl.lattice, gsum_vec)   # map back to index
            # println(gNum)
            # println(igsum)
            if igsum < 0 || igsum >= gNum
                # g+g' is not in the truncated set
                continue
            end
            imgsum = RL_iQ_neg(fl.lattice, igsum)
            imk = FL_ik_neg(fl, ik)
            imp = FL_ik_neg(fl, ip)
            img = RL_iQ_neg(fl.lattice, ig)
            ff_Λ[1+0][1+ig][1+ik,1+ip] += Nk_0 * w_g[1+igp] * ff_LL[1+0][1+igsum][1+ik,1+ip]
            ff_Λ[1+1][1+ig][1+ik,1+ip] += Nk_1 * w_g[1+igp] * ff_LL[1+1][1+igsum][1+ik,1+ip]
        end
    end

    return ff_Λ
end
