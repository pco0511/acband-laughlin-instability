
include("code_00_Lattice")
include("code_01_FiniteLattice")



# calculate BC under single valley
function QuantumGeometry_BerryCurvature_1V(fl::FiniteLattice, wf_1V::Vector{Vector{ComplexF64}})::Vector{Float64}


    u12 = fl.u12
    u21 = fl.u21
    if !(u12) || !(u21); println("use MP grid to compute Berry Curvature"); return; end

    g1 = fl.g1
    g2 = fl.g2
    siteNum = fl.siteNum

    ig1 = FL_ktoik(fl, g1)
    ig2 = FL_ktoik(fl, g2)

    # defining link variable (two directions along to g1 and g2)
    link_var = [zeros(ComplexF64, 2) for _ in 0:siteNum-1]
    for ik in 0:siteNum-1
        # ik is the computational index of k in FBZ sampling (FiniteLattice)
        # here ikp_x(y) indicates the computational basis index k+g1, k+g2, respectively
        ikp_x = FL_ik_sum(fl, ik, ig1)
        ikp_y = FL_ik_sum(fl, ik, ig2)

        # local gauge fixing
        u_x = dot(wf_1V[1+ik], wf_1V[1+ikp_x])
        link_var[1+ik][1+0] = u_x/abs(u_x)
        u_y = dot(wf_1V[1+ik], wf_1V[1+ikp_y])
        link_var[1+ik][1+1] = u_y/abs(u_y)
    end

    BC = [zeros(Float64) for _ in 0:siteNum-1]
    for ik in 0:siteNum-1
        ikp_x = FL_ik_sum(fl, ik, ig1)
        ikp_y = FL_ik_sum(fl, ik, ig2)

        # wilson loop
        link_prod = (link_var[1+ik][1+0] * link_var[1+ikp_x][1+1]) / (link_var[1+ikp_y][1+0] * link_var[1+ik][1+1])
        BC[1+ik] = imag(log(link_prod))
    end

    return BC
end