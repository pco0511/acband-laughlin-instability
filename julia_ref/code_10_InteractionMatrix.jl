include("code_00_Lattice.jl")
include("code_01_FiniteLattice.jl")
include("code_02_FormFactors.jl")

struct Interaction_Parms
    V1::Float64
    V3::Float64
    VC::Float64
    d::Float64
end


function Haldane_PseudoPotential(fl::FiniteLattice, P::Interaction_Parms, q::Vector{Float64}; eps::Float64=1e-9)
    ℓB2 = (fl.lattice.ℓB)^2
    q2 = dot(q,q)
    nq = norm(q)
    x = ℓB2 * q2
    
    V = 0.0

    # Coulomb potential
    if norm(q) > eps
        V += ℓB2^(-1) * P.VC / nq
    end


    # Haldane pseudo-potential
    V += -sqrt(3) * P.V1 * q2 / (4 * π)
    return V
end



function Haldane_PseudoPotential_2V(fl::FiniteLattice, P::Interaction_Parms, q::Vector{Float64}; eps::Float64=1.0e-9)::NTuple{2, Float64}
    d = P.d
    q_norm = norm(q)


    VP = 0.0

    # Coulomb potential
    if norm(q) > eps
        VP += (fl.lattice.ℓB)^(-2) * P.VC / q_norm
    end

    # Haldane pseudo-potential
    VP += - sqrt(3) * P.V1 * norm(q)^2 / (4 * π)

    # screened intervalley interaction
    UP = exp(-q_norm * d) * VP

    return VP, UP
end



# Function to reproduce the Interaction Matrix Element H_{k1,k2;k3,k4} (Eq. A2)
function InteractionMatrix_IdealChernBand_1V_fromFF(
    fl::FiniteLattice, Qs::Vector{Vector{Float64}},
    ff_Lambda::Vector{Matrix{ComplexF64}}, Int_parms::Interaction_Parms; eps::Float64=1e-9)::Array{ComplexF64,4}

    siteNum = fl.siteNum
    gNum = length(Qs)

    # RL vector -> index (returns -1 if not in truncated set)
    @inline function iQ_or_neg1(g::Vector{Float64})
        return RL_QtoiQ(fl.lattice, g)
    end

    H = zeros(ComplexF64, siteNum, siteNum, siteNum, siteNum)

    for ik1 in 0:siteNum-1, ik2 in 0:siteNum-1, ik3 in 0:siteNum-1
        k1 = fl.BZu[1+ik1]; k2 = fl.BZu[1+ik2]; k3 = fl.BZu[1+ik3];

        # enforce momentum conservation to determine k4
        Ksum = k1 + k2 - k3
        k4_BZ, Δg = MomentumSplit(fl.lattice, Ksum)
        ik4 = FL_ktoik(fl, k4_BZ; isFlux=1, eps=eps)
        if ik4 < 0; continue; end
        k4 = fl.BZu[1+ik4]


        total = 0.0 + 0.0im
        for ig in 0:gNum-1
            g = Qs[1+ig]
            gprime = Δg - g
            igp = iQ_or_neg1(gprime)
            if igp < 0 || igp >= gNum; continue; end

            # V_{k1 - (k4+g)} = V(k1 - k4 - g)
            Vval = Haldane_PseudoPotential(fl, Int_parms, k1 - k4 - g)

            Λ1 = ff_Lambda[1+ig ][1+ik1, 1+ik4]   # Λ_{k1, k4+g}
            Λ2 = ff_Lambda[1+igp][1+ik2, 1+ik3]   # Λ_{k2, k3+Δg-g}

            total += Vval * Λ1 * Λ2
        end

        H[1+ik1, 1+ik2, 1+ik3, 1+ik4] = total
    end

    return H
end



function InteractionMatrix_IdealChernBand_2V_fromFF(
    fl::FiniteLattice, Qs::Vector{Vector{Float64}},
    ff_Lambda::Vector{Vector{Matrix{ComplexF64}}}, Int_parms::Interaction_Parms; eps::Float64=1.0e-9)::Matrix{Array{ComplexF64,4}}

    siteNum = fl.siteNum
    gNum = length(Qs)

    k_set_v0 = fl.BZu 
    k_set_v1 = fl.BZd 

    @inline function iQ_or_neg1(g::Vector{Float64})
        return RL_QtoiQ(fl.lattice, g)
    end

    H = [zeros(ComplexF64, siteNum, siteNum, siteNum, siteNum) for _ in 1:2, _ in 1:2]

    # Calculate H[0,0] (intra-valley K)
    for ik1 in 0:siteNum-1, ik2 in 0:siteNum-1, ik3 in 0:siteNum-1
        k1 = k_set_v0[1+ik1]; k2 = k_set_v0[1+ik2]; k3 = k_set_v0[1+ik3]
        ik_sum_12 = FL_ik_sum(fl, ik1, ik2)
        ik_neg_3 = FL_ik_neg(fl, ik3)
        ik4 = FL_ik_sum(fl, ik_sum_12, ik_neg_3)

        Ksum = k1 + k2 - k3
        k4 = k_set_v0[1+ik4]
        Δg = Ksum - k4

        for ig in 0:gNum-1
            g = Qs[1+ig]; gprime_vec = Δg - g; igp = iQ_or_neg1(gprime_vec)
            if igp < 0 || igp >= gNum; continue; end
            V_q = k1 - k_set_v0[1+ik4] - g
            VP, _ = Haldane_PseudoPotential_2V(fl, Int_parms, V_q)
            Λ1 = ff_Lambda[1+0][1+ig][1+ik1, 1+ik4]
            Λ2 = ff_Lambda[1+0][1+igp][1+ik2, 1+ik3]
            H[1+0, 1+0][1+ik1, 1+ik2, 1+ik3, 1+ik4] += VP * Λ1 * Λ2
        end
    end

    # Calculate H[1,1] (intra-valley K')
    for ik1 in 0:siteNum-1, ik2 in 0:siteNum-1, ik3 in 0:siteNum-1
        k1 = k_set_v1[1+ik1]; k2 = k_set_v1[1+ik2]; k3 = k_set_v1[1+ik3]
        ik_sum_12 = FL_ik_sum(fl, ik1, ik2)
        ik_neg_3 = FL_ik_neg(fl, ik3)
        ik4 = FL_ik_sum(fl, ik_sum_12, ik_neg_3)

        Ksum = k1 + k2 - k3
        k4 = k_set_v1[1+ik4]
        Δg = Ksum - k4
        
        for ig in 0:gNum-1
            g = Qs[1+ig]; gprime_vec = Δg - g; igp = iQ_or_neg1(gprime_vec)
            if igp < 0 || igp >= gNum; continue; end
            V_q = k1 - k_set_v1[1+ik4] - g
            VP, _ = Haldane_PseudoPotential_2V(fl, Int_parms, V_q)
            Λ1 = ff_Lambda[1+1][1+ig][1+ik1, 1+ik4]
            Λ2 = ff_Lambda[1+1][1+igp][1+ik2, 1+ik3]
            H[1+1, 1+1][1+ik1, 1+ik2, 1+ik3, 1+ik4] += VP * Λ1 * Λ2
        end
    end

    # Calculate inter-valley
    for ik1 in 0:siteNum-1, ik2 in 0:siteNum-1, ik3 in 0:siteNum-1
        k1 = k_set_v0[1+ik1]; k2 = k_set_v1[1+ik2]; k3 = k_set_v1[1+ik3]
        ik_sum_12 = FL_ik_sum(fl, ik1, ik2)
        ik_neg_3 = FL_ik_neg(fl, ik3)
        ik4 = FL_ik_sum(fl, ik_sum_12, ik_neg_3)

        Ksum = k1 + k2 - k3
        k4 = k_set_v0[1+ik4]
        Δg = Ksum - k4
        
        for ig in 0:gNum-1
            g = Qs[1+ig]; gprime_vec = Δg - g; igp = iQ_or_neg1(gprime_vec)
            if igp < 0 || igp >= gNum; continue; end
            V_q = k1 - k_set_v0[1+ik4] - g
            _, UP = Haldane_PseudoPotential_2V(fl, Int_parms, V_q)
            Λ1_v0 = ff_Lambda[1+0][1+ig][1+ik1, 1+ik4]
            Λ2_v1 = ff_Lambda[1+1][1+igp][1+ik2, 1+ik3]
            H[1+0, 1+1][1+ik1, 1+ik2, 1+ik3, 1+ik4] += UP * Λ1_v0 * Λ2_v1
        end
    end
    

    for ik1 in 0:siteNum-1, ik2 in 0:siteNum-1, ik3 in 0:siteNum-1
        k1 = k_set_v1[1+ik1]; k2 = k_set_v0[1+ik2]; k3 = k_set_v0[1+ik3]
        ik_sum_12 = FL_ik_sum(fl, ik1, ik2)
        ik_neg_3 = FL_ik_neg(fl, ik3)
        ik4 = FL_ik_sum(fl, ik_sum_12, ik_neg_3)

        Ksum = k1 + k2 - k3
        k4 = k_set_v1[1+ik4]
        Δg = Ksum - k4

        for ig in 0:gNum-1
            g = Qs[1+ig]; gprime_vec = Δg - g; igp = iQ_or_neg1(gprime_vec)
            if igp < 0 || igp >= gNum; continue; end
            V_q = k1 - k_set_v1[1+ik4] - g
            _, UP = Haldane_PseudoPotential_2V(fl, Int_parms, V_q)
            Λ1_v0 = ff_Lambda[1+1][1+ig][1+ik1, 1+ik4]
            Λ2_v1 = ff_Lambda[1+0][1+igp][1+ik2, 1+ik3]
            H[1+1, 1+0][1+ik1, 1+ik2, 1+ik3, 1+ik4] += UP * Λ1_v0 * Λ2_v1
        end
    end


    # H[1+1,1+0] .= permutedims(H[1,2], (2,1,4,3))

    return H
end