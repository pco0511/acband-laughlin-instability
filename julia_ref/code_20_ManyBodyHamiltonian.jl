using Combinatorics
using SuiteSparseGraphBLAS
using Printf
using ProgressMeter


# separate the Hilbert space into symmetry sectors
## single valley
function SymmetrySectors_1V(fl::FiniteLattice, elecNum::Int64)::Vector{Vector{BitVector}}
    
    # total number of single hole eigenstates
    siteNum = fl.siteNum
    
    # total momentum sectors
    sectors_1V = [Vector{BitVector}(undef, 0) for itK in 0:siteNum-1]
    
    # iterate over all occupancy states and classify into the sectors
    for fqState in combinations(0:siteNum-1, elecNum)
        
        itK = 0  ## total momentum
        sqState = zeros(Bool, siteNum)  ## occupation basis (second quantization) representation of state
        for ie in 0:elecNum-1
            
            ## convert first quantized state to second quantization representation
            ik = fqState[1+ie]
            sqState[1+ik] = true
            
            ## total magnetization and momentum
            itK = FL_ik_sum(fl, itK, ik)
        end

        # append to sector
        push!(sectors_1V[1+itK], BitVector(sqState))
    end

    
    # sort the states of each sector and store the dimension of the sectors
    for itK in 0:siteNum-1
        sort!(sectors_1V[1+itK])
    end
    
    return sectors_1V
end



## both valleys
function SymmetrySectors_2V(fl::FiniteLattice, elecNum::Int64)::Vector{Vector{Vector{BitVector}}}
    
    # total number of single hole eigenstates
    valleyNum = 2
    siteNum = fl.siteNum
    totNum = valleyNum*siteNum
    
    # total magnetization and momentum sectors
    sectors_2V = [[Vector{BitVector}(undef, 0) for itK in 0:siteNum-1] for itM in 0:elecNum]
    
    # iterate over all occupancy states and classify into the sectors
    sqState = zeros(Bool, totNum)
    for fqState in combinations(0:totNum-1, elecNum)
        
        itM = 0  ## total magnetization
        itK = 0  ## total momentum
        sqState .= false  ## occupation basis (second quantization) representation of state
        
        for ie in 0:elecNum-1
            
            ## convert first quantized state to second quantized state
            isk = fqState[1+ie]
            sqState[1+isk] = true

            ## split index, isk = is + valleyNum*ik
            is = isk % valleyNum
            ik = isk ÷ valleyNum

            ## total magnetization and momentum
            itM += is
            itK = FL_ik_sum(fl, itK, ik)
        end
        
        # append to sector
        push!(sectors_2V[1+itM][1+itK], BitVector(sqState))
    end

    
    # sort the states of each sector and store the dimension of the sectors
    for itM in 0:elecNum, itK in 0:siteNum-1
        sort!(sectors_2V[1+itM][1+itK])
    end
    
    return sectors_2V
end



## both valleys, given number of electrons in each valley
function SymmetrySectors_2V_M(fl::FiniteLattice, elecNum_u::Int64, elecNum_d::Int64)::Vector{Vector{BitVector}}
    
    # total number of single hole eigenstates
    valleyNum = 2
    siteNum = fl.siteNum
    totNum = valleyNum*siteNum
    
    # total momentum sectors
    sectors_2V_M = [Vector{BitVector}(undef, 0) for itK in 0:siteNum-1]
    
    # iterate over all occupancy states and classify into the sectors
    for fqState_u in combinations(0:siteNum-1, elecNum_u), fqState_d in combinations(0:siteNum-1, elecNum_d)
        
        itK = 0  ## total momentum
        sqState = zeros(Bool, totNum)  ## occupation basis (second quantization) representation of state
        
        for ie in 0:elecNum_u-1
            
            ## convert first quantized state to second quantization representation
            ik = fqState_u[1+ie]
            iuk = 0 + valleyNum*ik
            sqState[1+iuk] = true
            
            ## total magnetization and momentum
            itK = FL_ik_sum(fl, itK, ik)
        end
        
        for ie in 0:elecNum_d-1
            
            ## convert first quantized state to second quantization representation
            ik = fqState_d[1+ie]
            idk = 1 + valleyNum*ik
            sqState[1+idk] = true
            
            ## total magnetization and momentum
            itK = FL_ik_sum(fl, itK, ik)
        end

        # append to sector
        push!(sectors_2V_M[1+itK], BitVector(sqState))
    end

    
    # sort the states of each sector and store the dimension of the sectors
    for itK in 0:siteNum-1
        sort!(sectors_2V_M[1+itK])
    end
    
    return sectors_2V_M
end




# matrix representation of the many-body Hamiltonian

# single valley, single band, total momentum resolved
## one-body energies
function ManyBodyHamiltonian_K_1V1B_1B(fl::FiniteLattice, ϵ_u::Vector{Float64}, sector_1V::Vector{BitVector})::SparseMatrixCSC{ComplexF64,Int64}
    
    # total number of single hole eigenstates
    siteNum = fl.siteNum

    
    # construct list (coo format)
    matIs = zeros(Int64, 0)
    matJs = zeros(Int64, 0)
    matVs = zeros(ComplexF64, 0)
    sqState = zeros(Bool, siteNum)
    for idx in 0:length(sector_1V)-1
        
        sqState .= sector_1V[1+idx]

        val = 0.0
        for ik in 0:siteNum-1
            if !sqState[1+ik]; continue; end
            val += ϵ_u[1+ik]
        end
        
        push!(matIs, idx)
        push!(matJs, idx)
        push!(matVs, val)
    end
    
    # convert list (coo format) to csc format
    return SparseArrays.sparse!(matIs .+ 1, matJs .+ 1, matVs)
end


## two-body interaction
function ManyBodyHamiltonian_K_1V1B_2B_modified(fl::FiniteLattice,
        H4::Array{ComplexF64,4},
        sector_1V::Vector{BitVector};
        normalize_by_N::Bool=true
        )::SparseMatrixCSC{ComplexF64,Int64}
    
    siteNum = fl.siteNum
    facN = normalize_by_N ? (1/siteNum) : 1.0
    
    # thread-local COO buffers
    threadNum = Threads.nthreads()
    matIss = [zeros(Int64, 0) for threadId in 1:threadNum]
    matJss = [zeros(Int64, 0) for threadId in 1:threadNum]
    matVss = [zeros(ComplexF64, 0) for threadId in 1:threadNum]
    sqStates = [zeros(Bool, siteNum) for threadId in 1:threadNum]
    
    
    function ManyBodyHamiltonian_K_1V1B_2B_idx(idx::Int64)::Nothing
        
        threadId = Threads.threadid()
        
        sqState = sqStates[threadId]
        sqState .= sector_1V[1+idx]
        
        # we use an ordered dict to prevent the coo format from growing due to multiple entries of (I,J) pairs
        matVs_idx = Dict{Int,ComplexF64}()
        
        
        # JW sign factor
        sign_JW_0 = +1
        for ik0 in 0:siteNum-1
            
            # ψ[ik0] |sqState> = sign_JW_1 |sqState_1>
            if !sqState[1+ik0]; continue; end
            sign_JW_1 = sign_JW_0
            if sum(@view sqState[1:ik0])%2==1; sign_JW_1 = -sign_JW_1; end
            sqState[1+ik0] = false
            
            for ik1 in 0:ik0-1
                
                # sign_JW_1 ψ[ik1] |sqState_1> = sign_JW_2 |sqState_2>
                if !sqState[1+ik1]; continue; end
                sign_JW_2 = sign_JW_1
                if sum(@view sqState[1:ik1])%2==1; sign_JW_2 = -sign_JW_2; end
                sqState[1+ik1] = false
                
                for ip in 0:siteNum-1
                    
                    # two body interaction with momentum transfer of p
                    # k0' = k0 + p,   k1' = k1 - p
                    ik0_ = FL_ik_sum(fl, ik0, ip)
                    ik1_ = FL_ik_sum(fl, ik1, FL_ik_neg(fl, ip))

                    # sign_JW_2 γ†[ik1_] |sqState_2> = sign_JW_3 |sqState_3>
                    if sqState[1+ik1_]; continue; end
                    sign_JW_3 = sign_JW_2
                    if sum(@view sqState[1:ik1_])%2==1; sign_JW_3 = -sign_JW_3; end
                    sqState[1+ik1_] = true
                    
                    # sign_JW_3 γ†[ik0_] |sqState_3> = sign_JW_4 |sqState_4>
                    if sqState[1+ik0_]
                        sqState[1+ik1_] = false
                        continue
                    end
                    
                    sign_JW_4 = sign_JW_3
                    if sum(@view sqState[1:ik0_])%2==1; sign_JW_4 = -sign_JW_4; end
                    sqState[1+ik0_] = true
                    
                    idx_ = searchsortedfirst(sector_1V, BitVector(sqState))-1
                    
                    if idx_==length(sector_1V) || sector_1V[1+idx_]!=sqState
                        println("**warning: applying two-body operator moved state outside symmetry sector**")
                    end
                    
                    Velem = H4[1+ik0_, 1+ik1_, 1+ik1, 1+ik0]
                    # interaction, <sector[idx_]|V|sector[idx]>  +=  sign_JW_4 * fV
                    ## double counting is avoided by restricting to ik1<ik0
                    val = sign_JW_4 * (1/siteNum)  * Velem * facN
                    if haskey(matVs_idx, idx_)
                        matVs_idx[idx_] += val
                    else
                        matVs_idx[idx_] = val
                    end
                    
                    sqState[1+ik0_] = false
                    sqState[1+ik1_] = false
                end
                
                sqState[1+ik1] = true
            end
            
            sqState[1+ik0] = true
        end
        
        for (idx_,val) in matVs_idx
            push!(matIss[threadId], idx_)
            push!(matJss[threadId], idx)
            push!(matVss[threadId], val)
        end

        return
    end
    
    
    Threads.@threads for idx in 0:length(sector_1V)-1
        ManyBodyHamiltonian_K_1V1B_2B_idx(idx)
    end
    
    matIs = vcat(matIss...)
    matJs = vcat(matJss...)
    matVs = vcat(matVss...)
    
    # convert list (coo format) to csc format
    return SparseArrays.sparse!(matIs .+ 1, matJs .+ 1, matVs)
end








## two-body interaction
function ManyBodyHamiltonian_K_2V1B_2B_modified(fl::FiniteLattice,
        H4::Matrix{Array{ComplexF64,4}},
        sector_2V::Vector{BitVector};
        normalize_by_N::Bool=true
        )::SparseMatrixCSC{ComplexF64,Int64}
    

    valleyNum = 2
    siteNum = fl.siteNum
    totNum = valleyNum * siteNum
    

    # thread-local COO buffers
    threadNum = Threads.nthreads()
    matIss = [zeros(Int64, 0) for threadId in 1:threadNum]
    matJss = [zeros(Int64, 0) for threadId in 1:threadNum]
    matVss = [zeros(ComplexF64, 0) for threadId in 1:threadNum]
    sqStates = [zeros(Bool, totNum) for threadId in 1:threadNum]
    
    Threads.@threads for idx in 0:length(sector_2V)-1
        threadId = Threads.threadid()
        sqState = sqStates[threadId]
        sqState .= sector_2V[1+idx]
        
        matVs_idx = ManyBodyHamiltonian_MK_2V1B_2B_col(fl, H4, sector_2V, sqState)

        for (idx_, val) in matVs_idx
            push!(matIss[threadId], idx_)
            push!(matJss[threadId], idx)
            push!(matVss[threadId], val)
        end
    end
    
    matIs = vcat(matIss...)
    matJs = vcat(matJss...)
    matVs = vcat(matVss...)
    
    # convert list (coo format) to csc format
    return SparseArrays.sparse!(matIs .+ 1, matJs .+ 1, matVs)
end





## two-body interaction, single column
function ManyBodyHamiltonian_MK_2V1B_2B_col(fl::FiniteLattice, H4::Matrix{Array{ComplexF64,4}}, sector_2V::Vector{BitVector}, sqState::Vector{Bool})::Dict{Int64, ComplexF64}
    facN = 1.0
    valleyNum = 2
    siteNum = fl.siteNum
    totNum = valleyNum*siteNum

    
    # we use an ordered dict to prevent the coo format from growing due to multiple entries of (I,J) pairs
    matVs_idx = Dict{Int,ComplexF64}()
    
    # JW sign factor
    sign_JW_0 = +1
    for isk0 in 0:totNum-1
        
        # ψ[isk0] |sqState> = sign_JW_1 |sqState_1>
        if !sqState[1+isk0]; continue; end
        sign_JW_1 = sign_JW_0
        if sum(@view sqState[1:isk0])%2==1; sign_JW_1 = -sign_JW_1; end
        sqState[1+isk0] = false
        
        # collapse index
        is0 = isk0 % valleyNum
        ik0 = isk0 ÷ valleyNum
        
        for isk1 in 0:isk0-1
            
            # sign_JW_1 ψ[isk1] |sqState_1> = sign_JW_2 |sqState_2>
            if !sqState[1+isk1]; continue; end
            sign_JW_2 = sign_JW_1
            if sum(@view sqState[1:isk1])%2==1; sign_JW_2 = -sign_JW_2; end
            sqState[1+isk1] = false
            
            # collapse index
            is1 = isk1 % valleyNum
            ik1 = isk1 ÷ valleyNum
            
            for ip in 0:siteNum-1
                
                # two body interaction with momentum transfer of p
                # k0' = k0 + p,   k1' = k1 - p
                ik0_ = FL_ik_sum(fl, ik0, ip)
                ik1_ = FL_ik_sum(fl, ik1, FL_ik_neg(fl, ip))
                
                    
                # combine index
                isk1_ = is1 + valleyNum*ik1_

                # sign_JW_2 γ†[isk1_] |sqState_2> = sign_JW_3 |sqState_3>
                if sqState[1+isk1_]; continue; end
                sign_JW_3 = sign_JW_2
                if sum(@view sqState[1:isk1_])%2==1; sign_JW_3 = -sign_JW_3; end
                sqState[1+isk1_] = true
                
                    
                # combine index
                isk0_ = is0 + valleyNum*ik0_
                
                # sign_JW_3 γ†[isk0_] |sqState_3> = sign_JW_4 |sqState_4>
                if sqState[1+isk0_]
                    sqState[1+isk1_] = false
                    continue
                end
                
                sign_JW_4 = sign_JW_3
                if sum(@view sqState[1:isk0_])%2==1; sign_JW_4 = -sign_JW_4; end
                sqState[1+isk0_] = true
                
                idx_ = searchsortedfirst(sector_2V, BitVector(sqState))-1
                
                if idx_==length(sector_2V) || sector_2V[1+idx_]!=sqState
                    println("**warning: applying two-body operator moved state outside symmetry sector**")
                end

                Velem = H4[1+is0, 1+is1][1+ik0_, 1+ik1_, 1+ik1, 1+ik0]
                # interaction, <sector[idx_]|V|sector[idx]>  +=  sign_JW_4 * fV
                ## double counting is avoided by restricting to ik1<ik0
                val = sign_JW_4 * (1/siteNum)  * Velem * facN
                if haskey(matVs_idx, idx_)
                    matVs_idx[idx_] += val
                else
                    matVs_idx[idx_] = val
                end
                    
                sqState[1+isk0_] = false
                sqState[1+isk1_] = false
            end
            
            sqState[1+isk1] = true
        end
        
        sqState[1+isk0] = true
    end

    return matVs_idx
end


