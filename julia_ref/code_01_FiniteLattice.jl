# packages
include("code_00_Lattice.jl")





# finite lattice
## composite data type
struct FiniteLattice
    
    # (infinite) lattice
    lattice::Lattice
    
    # construction matrix
    u11::Int64
    u12::Int64
    u21::Int64
    u22::Int64
    
    # size of finite lattice
    siteNum::Int64
    L1::Int64
    L2::Int64

    # boundary vectors
    vecL1::Vector{Float64}
    vecL2::Vector{Float64}
    
    # momentum grid lattice vectors
    g1::Vector{Float64}
    g2::Vector{Float64}
    
    # Brillouin zone grid
    BZ0::Vector{Vector{Float64}}
    
    # flux/2π for twisted boundary conditions
    isTRS::Bool
    α1::Float64
    α2::Float64
    BZu::Vector{Vector{Float64}}
    BZd::Vector{Vector{Float64}}
end

## constructor
function FiniteLattice(lattice::Lattice, u11::Int64, u12::Int64, u21::Int64, u22::Int64; isTRS::Bool=true, α1::Float64=0.0, α2::Float64=0.0, eps::Float64=1.0e-9)::FiniteLattice

    # check for structure of u matrix
    siteNum = u11*u22 - u12*u21
    
    L1 = gcd(u11, u12)
    L2 = gcd(u21, u22)

    if siteNum != L1*L2
        println("**warning: FiniteLattice does not satisfy necessary arithmetic operations**")
    end
    
    # boundary vectors
    vecL1 = u11*lattice.a1 + u12*lattice.a2
    vecL2 = u21*lattice.a1 + u22*lattice.a2

    # finite lattice area
    area_FL = cross([vecL1; 0], [vecL2; 0])[3]
    
    # momentum grid lattice vectors
    g1 = (2*π/area_FL)*[+vecL2[2], -vecL2[1]]
    g2 = (2*π/area_FL)*[-vecL1[2], +vecL1[1]]
    
    # Brillouin zone grid
    BZ0 = [zeros(Float64, 2) for ik in 0:siteNum-1]
    BZu = [zeros(Float64, 2) for ik in 0:siteNum-1]
    BZd = [zeros(Float64, 2) for ik in 0:siteNum-1]
    for ik1 in 0:L1-1, ik2 in 0:L2-1
        
        ik = ik1 + L1*ik2
        k0 = ik1*g1 + ik2*g2
        ku = (ik1+α1)*g1 + (ik2+α2)*g2
        kd = (ik1-α1)*g1 + (ik2-α2)*g2
        
        # modulo reciporcal lattice vectors
        f1 = dot(k0, lattice.a1)/(2*π)
        f2 = dot(k0, lattice.a2)/(2*π)
        f1 = f1 - floor(f1 + eps)
        f2 = f2 - floor(f2 + eps)
        k0 = f1*lattice.Q1 + f2*lattice.Q2
        
        f1 = dot(ku, lattice.a1)/(2*π)
        f2 = dot(ku, lattice.a2)/(2*π)
        f1 = f1 - floor(f1 + eps)
        f2 = f2 - floor(f2 + eps)
        ku = f1*lattice.Q1 + f2*lattice.Q2
        
        f1 = dot(kd, lattice.a1)/(2*π)
        f2 = dot(kd, lattice.a2)/(2*π)
        f1 = f1 - floor(f1 + eps)
        f2 = f2 - floor(f2 + eps)
        kd = f1*lattice.Q1 + f2*lattice.Q2
        
        BZ0[1+ik] = k0
        BZu[1+ik] = ku
        BZd[1+ik] = (isTRS ? kd : ku)
    end
    
    return FiniteLattice(lattice, u11, u12, u21, u22, siteNum, L1, L2, vecL1, vecL2, g1, g2, BZ0, isTRS, α1, α2, BZu, BZd)
end

## crystal momentum to index map (only for testing, should not be used in most actual calculations)
function FL_ktoik(fl::FiniteLattice, k::AbstractVector{Float64}; isFlux::Int64=0, eps::Float64=1.0e-9)

    # convert to flux-less crystal momentum in BZ0
    if isFlux==0
        k0 = k
    elseif isFlux==1
        k0 = k - fl.α1*fl.g1 - fl.α2*fl.g2
    else
        k0 = k + fl.α1*fl.g1 + fl.α2*fl.g2
    end
    
    # integer indices
    ik1_float = dot(k0, fl.vecL1)/(2*π)
    ik2_float = dot(k0, fl.vecL2)/(2*π)
    
    ik1 = round(Int, ik1_float)
    ik2 = round(Int, ik2_float)
    
    if abs(ik1_float - ik1)>eps || abs(ik2_float - ik2)>eps
        println("**warning: k not in BZ grid**")
        return -1
    end
    
    ik1 = mod(ik1, fl.L1)
    ik2 = mod(ik2, fl.L2)
    
    ik = ik1 + fl.L1*ik2
    
    return ik
end

## mk = -k w/ integer arithmetic
### as an implicit assumption, BZ0 <-> BZ0 and BZp <-> BZn
function FL_ik_neg(fl::FiniteLattice, ik::Int64)
    
    ik1 = ik % fl.L1
    ik2 = ik ÷ fl.L1
    
    imk1 = mod(-ik1, fl.L1)
    imk2 = mod(-ik2, fl.L2)
    
    imk = imk1 + fl.L1*imk2
    
    return imk
end

## k_ = k + p w/ integer arithmetic
### as an impliicit assumption, k and k_ belong to the same BZ_, and p belongs to BZ0
function FL_ik_sum(fl::FiniteLattice, ik::Int64, ip::Int64)
    
    ik1 = ik % fl.L1
    ik2 = ik ÷ fl.L1
    
    ip1 = ip % fl.L1
    ip2 = ip ÷ fl.L1
    
    ik_1 = mod(ik1+ip1, fl.L1)
    ik_2 = mod(ik2+ip2, fl.L2)
    
    ik_ = ik_1 + fl.L1*ik_2

    return ik_
end



# let us assume there is a fl_upscale which each uab was enhanced by uL
# map momentum index in fl to that in fl_upscale
function FL_ik_upscale(fl::FiniteLattice, ik::Int64, uL::Int64)
    
    ik1 = ik % fl.L1
    ik2 = ik ÷ fl.L1

    ik1_upscale = uL*ik1
    ik2_upscale = uL*ik2
    
    ik_upscale = ik1_upscale + (uL*fl.L1)*ik2_upscale
    
    return ik_upscale
end




# reciprocal lattice operations
## split momentum into BZ and reciprocal lattice part
function FL_MomentumSplit(fl::FiniteLattice, k_total::AbstractVector{Float64}; isFlux::Int64=0, eps::Float64=1.0e-9)::Tuple{Vector{Float64}, Vector{Float64}}
    
    lattice = fl.lattice
    
    if isFlux == 0
        k_crystal = k_total
    elseif isFlux == 1
        k_crystal = k_total - (fl.α1 * fl.g1 + fl.α2 * fl.g2)
    else
        k_crystal = k_total + (fl.α1 * fl.g1 + fl.α2 * fl.g2)
    end

    
    l1_float = dot(k_crystal, lattice.a1)/(2*π)
    l2_float = dot(k_crystal, lattice.a2)/(2*π)

    
    l1 = round(Int, l1_float)
    l2 = round(Int, l2_float)

    Q = l1*lattice.Q1 + l2*lattice.Q2


    k_crystal_BZ = k_total - Q
    
    return k_crystal_BZ, Q
end
