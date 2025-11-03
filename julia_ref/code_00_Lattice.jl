# packages
using LinearAlgebra





# lattice
struct Lattice

    # lattice type (4 = square, 6 = triangular)
    latticeType::Int64
    
    # lattice vectors
    a1::Vector{Float64}
    a2::Vector{Float64}

    # reciprocal lattice vectors
    Q1::Vector{Float64}
    Q2::Vector{Float64}

    # magnetic length (assuming |C| = 1)
    ℓB::Float64
end

## triangular lattice, normalized, a_M = 1
function TriangularLattice()::Lattice
    
    # lattice vectors
    a1 = [-sqrt(3)/2, +1/2]
    a2 = [-sqrt(3)/2, -1/2]
    
    # unit cell area
    area_UC = cross([a1; 0], [a2; 0])[3]
    
    # reciprocal lattice vectors
    Q1 = (2*π/area_UC)*[+a2[2], -a2[1]]
    Q2 = (2*π/area_UC)*[-a1[2], +a1[1]]

    # magnetic length
    ℓB = sqrt(area_UC/(2*π))

    return Lattice(6, a1, a2, Q1, Q2, ℓB)
end





# reciprocal lattice operations
## split momentum into BZ and reciprocal lattice part
function MomentumSplit(lattice::Lattice, K::AbstractVector{Float64}; eps::Float64=1.0e-9)::Tuple{Vector{Float64}, Vector{Float64}}
    
    l1_float = dot(K, lattice.a1)/(2*π)
    l2_float = dot(K, lattice.a2)/(2*π)
    
    l1 = floor(Int, l1_float + eps)
    l2 = floor(Int, l2_float + eps)

    k = (l1_float - l1)*lattice.Q1 + (l2_float - l2)*lattice.Q2
    Q = l1*lattice.Q1 + l2*lattice.Q2
    
    return k, Q
end



# # reciprocal lattice operations
# ## split momentum into BZ and reciprocal lattice part
# function MomentumSplit(fl::FiniteLattice, K::AbstractVector{Float64}; isFlux::Int64=0, eps::Float64=1.0e-9)::Tuple{Vector{Float64}, Vector{Float64}}
    
#     lattice = fl.lattice
#     k0 = K

#     # if isFlux==0
#     #     k0 = K
#     # elseif isFlux==1
#     #     k0 = K - fl.α1*fl.g1 - fl.α2*fl.g2
#     # else
#     #     k0 = K + fl.α1*fl.g1 + fl.α2*fl.g2
#     # end 

#     l1_float = dot(k0, lattice.a1)/(2*π)
#     l2_float = dot(k0, lattice.a2)/(2*π)
    
#     l1 = floor(Int, l1_float + eps)
#     l2 = floor(Int, l2_float + eps)


#     if isFlux==0
#         k = (l1_float - l1)*lattice.Q1 + (l2_float - l2)*lattice.Q2 
#     elseif isFlux==1
#         k = (l1_float - l1)*lattice.Q1 + (l2_float - l2)*lattice.Q2 + fl.α1*fl.g1 + fl.α2*fl.g2
#     else
#         k = (l1_float - l1)*lattice.Q1 + (l2_float - l2)*lattice.Q2 - fl.α1*fl.g1 - fl.α2*fl.g2
#     end

#     Q = l1*lattice.Q1 + l2*lattice.Q2
    
#     return k, Q
# end



## reciprocal lattice index to (l1,l2) map
function RL_iQtol1l2(lattice::Lattice, iQ::Int64; Qd::Int64=20)::Tuple{Int64, Int64}

    if lattice.latticeType==6

        if iQ<0
            println("**warning: iQ is negative**")
            return -1, -1
        end
        
        if iQ==0; return 0,0; end

        
        for Qd_ in 1:Qd-1
    
            QNum_ = 3*Qd_*Qd_ - 3*Qd_ + 1
            
            if iQ>=QNum_ && iQ<QNum_+6*Qd_
            
                if iQ>=QNum_+0*Qd_ && iQ<QNum_+1*Qd_
    
                    l = iQ - (QNum_+0*Qd_)
                    
                    l1 = -(Qd_-l)
                    l2 = -Qd_
                    
                elseif iQ>=QNum_+1*Qd_ && iQ<QNum_+2*Qd_
    
                    l = iQ - (QNum_+1*Qd_)
                    
                    l1 = +l
                    l2 = -(Qd_-l)
                    
                elseif iQ>=QNum_+2*Qd_ && iQ<QNum_+3*Qd_
    
                    l = iQ - (QNum_+2*Qd_)
                    
                    l1 = +Qd_
                    l2 = +l
                    
                elseif iQ>=QNum_+3*Qd_ && iQ<QNum_+4*Qd_
    
                    l = iQ - (QNum_+3*Qd_)
                    
                    l1 = +(Qd_-l)
                    l2 = +Qd_
                    
                elseif iQ>=QNum_+4*Qd_ && iQ<QNum_+5*Qd_
    
                    l = iQ - (QNum_+4*Qd_)
                    
                    l1 = -l
                    l2 = +(Qd_-l)
                    
                elseif iQ>=QNum_+5*Qd_ && iQ<QNum_+6*Qd_
    
                    l = iQ - (QNum_+5*Qd_)
                    
                    l1 = -Qd_
                    l2 = -l
                    
                end
    
                return l1, l2
            end
        end
    
        println("**warning: iQ is too large**")
        return -1, -1
    else
        println("**warning: lattice convention incompatible with existing lattice constructor**")
        return -1, -1
    end
end

## reciprocal lattice (l1,l2) to index map
function RL_l1l2toiQ(lattice::Lattice, l1::Int64, l2::Int64)::Int64

    if lattice.latticeType==6
        
        Qd_ = max(abs(l1), abs(l2), abs(l1-l2))
        QNum_ = 3*Qd_*Qd_ - 3*Qd_ + 1
        
        if Qd_==0
            
            iQ = 0
            
        elseif l2 == -Qd_
        
            l = l1-l2
            iQ = QNum_ + 0*Qd_ + l
            
        elseif l1-l2 == +Qd_
        
            l = l1
            iQ = QNum_ + 1*Qd_ + l
        
        elseif l1 == +Qd_
            
            l = l2
            iQ = QNum_ + 2*Qd_ + l
        
        elseif l2 == +Qd_
            
            l = -(l1-l2)
            iQ = QNum_ + 3*Qd_ + l
        
        elseif l1-l2 == -Qd_
            
            l = -l1
            iQ = QNum_ + 4*Qd_ + l
        
        elseif l1 == -Qd_
            
            l = -l2
            iQ = QNum_ + 5*Qd_ + l
            
        end
        
        return iQ
    else
        println("**warning: lattice convention incompatible with existing lattice constructor**")
        return -1
    end
end

## reciprocal lattice momentum to index map
function RL_QtoiQ(lattice::Lattice, Q::Vector{Float64}; eps::Float64=1.0e-9)::Int64

    # Q = l1*Q1 + l2*Q2
    l1_float = dot(Q, lattice.a1)/(2*π)
    l2_float = dot(Q, lattice.a2)/(2*π)
    
    l1 = round(Int, l1_float)
    l2 = round(Int, l2_float)
    
    if abs(l1_float - l1)>eps || abs(l2_float - l2)>eps
        println("**warning: Q not in reciprocal lattice**")
        return -1
    end

    return RL_l1l2toiQ(lattice, l1, l2)
end

## Q_ = (-Qx, Qy) w/ integer arithmetic
function RL_iQ_C2y(lattice::Lattice, iQ::Int64)
    
    l1, l2 = RL_iQtol1l2(lattice, iQ)
    iQ_ = RL_l1l2toiQ(lattice, -l2, -l1)
    
    return iQ_
end

## Q_ = -Q w/ integer arithmetic
function RL_iQ_neg(lattice::Lattice, iQ::Int64)
    
    l1, l2 = RL_iQtol1l2(lattice, iQ)
    iQ_ = RL_l1l2toiQ(lattice, -l1, -l2)
    
    return iQ_
end

## Q_ = (Qx, -Qy) w/ integer arithmetic
function RL_iQ_C2x(lattice::Lattice, iQ::Int64)
    
    l1, l2 = RL_iQtol1l2(lattice, iQ)
    iQ_ = RL_l1l2toiQ(lattice, l2, l1)
    
    return iQ_
end

## Q'' = Q' + Q w/ integer arithmetic
function RL_iQ_sum(lattice::Lattice, iQ_::Int64, iQ::Int64)
    
    l1_, l2_ = RL_iQtol1l2(lattice, iQ_)
    l1,  l2  = RL_iQtol1l2(lattice, iQ)
    
    iQ__ = RL_l1l2toiQ(lattice, l1_+l1, l2_+l2)

    return iQ__
end





# finite reciprocal lattice
function FiniteReciprocalLattice(lattice::Lattice, Qd::Int64)::Vector{Vector{Float64}}
    
    if lattice.latticeType==6

        Q1 = lattice.Q1
        Q2 = lattice.Q2
        Q0 = -Q1-Q2
        
        QNum = 3*Qd*Qd - 3*Qd + 1
        Qs = [zeros(2) for iQ in 0:QNum-1]
        for Qd_ in 1:Qd-1

            QNum_ = 3*Qd_*Qd_ - 3*Qd_ + 1
            
            for l in 0:Qd_-1
                Qs[1+QNum_+0*Qd_+l] = (Qd_-l)*(+Q0) + l*(-Q2)
                Qs[1+QNum_+1*Qd_+l] = (Qd_-l)*(-Q2) + l*(+Q1)
                Qs[1+QNum_+2*Qd_+l] = (Qd_-l)*(+Q1) + l*(-Q0)
                Qs[1+QNum_+3*Qd_+l] = (Qd_-l)*(-Q0) + l*(+Q2)
                Qs[1+QNum_+4*Qd_+l] = (Qd_-l)*(+Q2) + l*(-Q1)
                Qs[1+QNum_+5*Qd_+l] = (Qd_-l)*(-Q1) + l*(+Q0)
            end
        end

        return Qs
    else
        println("**warning: lattice convention incompatible with existing finite reciprocal lattice constructor**")

        return [[0.0, 0.0]]
    end
end




