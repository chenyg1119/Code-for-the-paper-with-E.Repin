module CORE


function uni(n) #QR decomposition. Q being the unitary matrix.
    H = randn(n,n) + 1im*randn(n,n)
    Q,R = qr(H) ### QR decomposition
    #return Q*Q.' #symmetric
    return Q #arbiturary
    #return diagm(Complex{Float64}[1,1,1,1])
end

#s = uni(4)




#call unitary matrix such that the random values are unanimous in all expression
# s here is 4 by 4 energy independent
# better run separately to for the sake of minor fixes in other codes while pinning s




function ground(E,x,y,z,s) # returns the ground energy

    e_phi(x,y) = exp.([x,y,0.5im*Float64(z),0]) # array list of phase-exponentials
    ephi = diagm(e_phi(1im*x/2.,1im*y/2.))
    ephiInv = inv(ephi)
    Asqr = 0.5*(1-E/sqrt(E^2+1))
    Asqrem = 0.5*(1+E/sqrt(E^2+1))
    Q =  Asqr*ephiInv*s*ephi  +  Asqrem*ephi*s.'*ephiInv
    return -0.5*log(det(Q))
end

function current1(E,x,y,z,s) # returns the current w.r.t the first phase change

    e_phi(x,y) = exp.([x,y,0.5im*Float64(z),0]) # array list of phase-exponentials
    ephi = diagm(e_phi(1im*x/2.,1im*y/2.))
    ephiInv = inv(ephi)
    Asqr = 0.5*(1-E/sqrt(E^2+1))
    Asqrem = 0.5*(1+E/sqrt(E^2+1))

    phiProj1 = diagm(Complex{Float64}[1im/2,0,0,0])

    Q =  Asqr*ephiInv*s*ephi  +  Asqrem*ephi*s.'*ephiInv
    Qinv = inv(Q)
    comm1 = ephiInv*s*ephi*phiProj1  -   phiProj1*ephiInv*s*ephi
    Qdphi1 =  Asqr*comm1 + Asqrem*comm1.'

    return -0.5*trace(Qinv*Qdphi1)
end


function current2(E,x,y,z,s) ## returns the current w.r.t the second phase change

    e_phi(x,y) = exp.([x,y,0.5im*Float64(z),0]) # array list of phase-exponentials
    ephi = diagm(e_phi(1im*x/2.,1im*y/2.))
    ephiInv = inv(ephi)
    Asqr = 0.5*(1-E/sqrt(E^2+1))
    Asqrem = 0.5*(1+E/sqrt(E^2+1))

    phiProj2 = diagm(Complex{Float64}[0,1im/2,0,0])

    Q =  Asqr*ephiInv*s*ephi  +  Asqrem*ephi*s.'*ephiInv
    Qinv = inv(Q)
    comm2 = ephiInv*s*ephi*phiProj2  -   phiProj2*ephiInv*s*ephi
    Qdphi2 =  Asqr*comm2 + Asqrem*comm2.'

    return -0.5*trace(Qinv*Qdphi2)
end


function invar(E,x,y,z,s) # invariant Q^{-1}.dQ/de.Q^{-1}.dQ/d(phi_1).Q^{-1}.dQ/d(phi_2), antisymmetrised upon 1,2
    ephi = diagm(exp.([0.5im*Float64(x),0.5im*Float64(y),0.5im*Float64(z),0.])) # diag matrix of phase-exponentials
    ephiInv = inv(ephi)
    sphi = ephiInv*s*ephi
    Asqr = 0.5*(1.-E/sqrt(E^2+1.))
    Asqrem = 0.5*(1.+E/sqrt(E^2+1.))
    Q =  Asqr*sphi  +  Asqrem*sphi.'
    Qinv = inv(Q)
    Qde = 1/(2*(sqrt(E^2+1))^3)*(sphi.'-sphi)

    phiProj1 = diagm(Complex{Float64}[0.5*1im,0.,0.,0.])
    phiProj2 = diagm(Complex{Float64}[0.,0.5*1im,0.,0.])

    comm1 = sphi*phiProj1  -   phiProj1*sphi
    comm2 = sphi*phiProj2  -   phiProj2*sphi
    #comm(n) = ephiInv*s*ephi*phiProj(n)  -   phiProj(n)*ephiInv*s*ephi
    Qdphi1 =  Asqr*comm1 + Asqrem*comm1.'
    Qdphi2 =  Asqr*comm2 + Asqrem*comm2.'

    #Qdphi(n) =  Asqr*comm(n) + Asqrem*comm(n)'

    inv_ = Qinv*Qde*Qinv*Qdphi1*Qinv*Qdphi2  -  Qinv*Qde*Qinv*Qdphi2*Qinv*Qdphi1
    return -0.25*trace(inv_)
end


function SE(E_,x,y,z,scale_,s_)
    E = Float64(E_)
    scale = Float64(scale_)
    s = Array{Complex{Float64}}(s_)
    F::Base.LinAlg.Eigen{Complex{Float64},Complex{Float64},Array{Complex{Float64},2},Array{Complex{Float64},1}} = eigfact(s)
    d::Array{Complex{Float64},1} = F[:values]
    v::Array{Complex{Float64},2} = F[:vectors]
    sgn::Array{Float64,2} = diagm(sign.(sin.(0.5*angle.(d))))
    e0 = scale*diagm(cos.(0.5*angle.(d)))*sgn
    gamma = 2*scale*diagm(sin.(0.5*angle.(d)))*sgn
    pole = e0 + 0.5*im*gamma
    S = v*((im*E*eye(4)-pole)/(im*E*eye(4)-conj(pole)))*v'
    return S
end



function LargeE(E_,x,y,z,scale_,s_) # invariant Q^{-1}.dQ/de.Q^{-1}.dQ/d(phi_1).Q^{-1}.dQ/d(phi_2), antisymmetrised upon 1,2
    E = Float64(E_)
    scale = Float64(scale_)
    s = Array{Complex{Float64}}(s_)
    F::Base.LinAlg.Eigen{Complex{Float64},Complex{Float64},Array{Complex{Float64},2},Array{Complex{Float64},1}}= eigfact(s)
    d::Array{Complex{Float64},1} = F[:values]
    v::Array{Complex{Float64},2} = F[:vectors]
    sgn::Array{Float64,2} = diagm(sign.(sin.(0.5*angle.(d))))
    e0 = scale*diagm(cos.(0.5*angle.(d)))*sgn
    gamma = 2*scale*diagm(sin.(0.5*angle.(d)))*sgn
    pole = e0 + 0.5*im*gamma
    S = v*((im*E*eye(4)-pole)/(im*E*eye(4)-conj(pole)))*v'
    Sem = v*((-im*E*eye(4)-pole)/(-im*E*eye(4)-conj(pole)))*v'
    ephi = diagm(exp.([0.5im*Float64(x),0.5im*Float64(y),0.5im*Float64(z),0.])) # diag matrix of phase-exponentials
    ephiInv = inv(ephi)
    sphi = ephiInv*S*ephi
    semphi = ephiInv*Sem*ephi
    Semde = ephiInv*v*(gamma/(-im*E*eye(4)-conj(pole))^2)*v'*ephi

    phiProj1 = diagm(Complex{Float64}[0.5*1im,0.,0.,0.])
    phiProj2 = diagm(Complex{Float64}[0.,0.5*1im,0.,0.])

    comm1 = sphi*phiProj1  -   phiProj1*sphi   #dS/dphi1
    comm2 = sphi*phiProj2  -   phiProj2*sphi
    commem1 = semphi*phiProj1  -   phiProj1*semphi #E -> -E
    commem2 = semphi*phiProj2  -   phiProj2*semphi

    inv_::Array{Complex{Float64},2} = Semde'*sphi*commem1'*comm2 - Semde'*sphi*commem2'*comm1
    return -0.5*trace(inv_)
end

function test(E_,x,y,z,scale_,s_)
    E = Float64(E_)
    scale = Float64(scale_)
    s = Array{Complex{Float64}}(s_)
    F = eigfact(s)
    d::Array{Complex{Float64},1} = F[:values]
    v::Array{Complex{Float64},2} = F[:vectors]
    sgn::Array{Float64,2} = diagm(sign.(sin.(0.5*angle.(d))))
    e0 = scale*diagm(cos.(0.5*angle.(d)))*sgn
    gamma = 2*scale*diagm(sin.(0.5*angle.(d)))*sgn
    pole = e0 + 0.5*im*gamma
    S = v*((im*E*eye(4)-pole)/(im*E*eye(4)-conj(pole)))*v'
    Semde = v*(-gamma/((-im*E*eye(4)-conj(pole))^2))*v'
    ephi = diagm(exp.([0.5im*Float64(x),0.5im*Float64(y),0.5im*Float64(z),0.])) # diag matrix of phase-exponentials
    ephiInv = inv(ephi)
    phiProj1 = diagm(Complex{Float64}[0.5*1im,0.,0.,0.])
    phiProj2 = diagm(Complex{Float64}[0.,0.5*1im,0.,0.])
    sphi = ephiInv*S*ephi
    Semdephi = ephiInv*Semde*ephi
    Semdagphi = inv(sphi)
    ans = trace(Semdephi'*Semdagphi*(Semdagphi*phiProj1-phiProj1*Semdagphi)*(sphi*phiProj2-phiProj2*sphi) - Semdephi'*Semdagphi*(Semdagphi*phiProj2-phiProj2*Semdagphi)*(sphi*phiProj1-phiProj1*sphi))
    return -0.5*ans
end






function invarE(E::Float64, x::Float64, y::Float64, z::Float64, scale::Float64, s::Array{Complex{Float64},2}) # invariant Q^{-1}.dQ/de.Q^{-1}.dQ/d(phi_1).Q^{-1}.dQ/d(phi_2), antisymmetrised upon 1,2
    F::Base.LinAlg.Eigen{Complex{Float64},Complex{Float64},Array{Complex{Float64},2},Array{Complex{Float64},1}} = eigfact(s)
    d::Array{Complex{Float64},1} = F[:values]::Array{Complex{Float64},1}
    v::Array{Complex{Float64},2} = F[:vectors]::Array{Complex{Float64},2}
    sgn::Array{Float64,2} = diagm(sign.(sin.(0.5*angle.(d))))
    e0 = scale*diagm(cos.(0.5*angle.(d)))*sgn
    gamma = 2.*scale*diagm(sin.(0.5*angle.(d)))*sgn
    pole = e0 + 0.5*im*gamma
    S = v*((im*E*eye(4)-pole)/(im*E*eye(4)-conj(pole)))*v'
    Sem = v*((-im*E*eye(4)-pole)/(-im*E*eye(4)-conj(pole)))*v'  #energy depend S transpose has to change E -> -E
    Sde = v*(-gamma/((im*E*eye(4)-conj(pole))^2))*v'
    Semde = v*(gamma/((-im*E*eye(4)-conj(pole))^2))*v'  #dS_{-E}/de
    ephi = diagm(exp.([0.5im*x,0.5im*y,0.5im*z,0.])) # diag matrix of phase-exponentials
    ephiInv = inv(ephi)
    sphi = ephiInv*S*ephi
    semphi = ephiInv*Sem*ephi  #sphi with -E
    sdephi = ephiInv*Sde*ephi
    semdephi = ephiInv*Semde*ephi   #sdephi with -E

    Asqr = 0.5*(1.-E/sqrt(E^2+1.))
    Asqrem = 0.5*(1.+E/sqrt(E^2+1.))
    Q =  Asqr*sphi  +  Asqrem*semphi.'
    Qinv = inv(Q)
    Qde = 1./(2.*(sqrt(E^2+1.))^3)*(semphi.'-sphi) + Asqr*sdephi + Asqrem*semdephi.'

    phiProj1 = diagm(Complex{Float64}[0.5*1im,0.,0.,0.])
    phiProj2 = diagm(Complex{Float64}[0.,0.5*1im,0.,0.])

    comm1 = sphi*phiProj1  -   phiProj1*sphi   #dS/dphi1
    comm2 = sphi*phiProj2  -   phiProj2*sphi
    commem1 = semphi*phiProj1  -   phiProj1*semphi #E -> -E
    commem2 = semphi*phiProj2  -   phiProj2*semphi

    Qdphi1 =  Asqr*comm1 + Asqrem*commem1.'
    Qdphi2 =  Asqr*comm2 + Asqrem*commem2.'


    #Qdphi(n) =  Asqr*comm(n) + Asqrem*comm(n)'

    inv_::Array{Complex{Float64},2} = Qinv*Qde*Qinv*Qdphi1*Qinv*Qdphi2  -  Qinv*Qde*Qinv*Qdphi2*Qinv*Qdphi1
    return -0.25*trace(inv_)
end

function boundary(E_,x,y,z,scale_,s_) # invariant Q^{-1}.dQ/de.Q^{-1}.dQ/d(phi_1).Q^{-1}.dQ/d(phi_2), antisymmetrised upon 1,2
    E = Float64(E_)
    scale = Float64(scale_)
    s = Array{Complex{Float64}}(s_)
    F::Base.LinAlg.Eigen{Complex{Float64},Complex{Float64},Array{Complex{Float64},2},Array{Complex{Float64},1}} = eigfact(s)
    d::Array{Complex{Float64},1} = F[:values]
    v::Array{Complex{Float64},2} = F[:vectors]
    sgn::Array{Float64,2} = diagm(sign.(sin.(0.5*angle.(d))))
    e0 = scale*diagm(cos.(0.5*angle.(d)))*sgn
    gamma = 2*scale*diagm(sin.(0.5*angle.(d)))*sgn
    pole = e0 + 0.5*im*gamma
    S = v*((im*E*eye(4)-pole)/(im*E*eye(4)-conj(pole)))*v'
    Sem = v*((-im*E*eye(4)-pole)/(-im*E*eye(4)-conj(pole)))*v'  #energy depende transpose has to change E -> -E
    Sde = v*(-gamma/((im*E*eye(4)-conj(pole))^2))*v'
    Semde = v*(gamma/((-im*E*eye(4)-conj(pole))^2))*v'  #dS_{-E}/de
    ephi = diagm(exp.([0.5im*Float64(x),0.5im*Float64(y),0.5im*Float64(z),0.])) # diag matrix of phase-exponentials
    ephiInv = inv(ephi)
    sphi = ephiInv*S*ephi
    semphi = ephiInv*Sem*ephi  #sphi with -E
    sdephi = ephiInv*Sde*ephi
    semdephi = ephiInv*Semde*ephi   #sdephi with -E
    Asqr = 0.5*(1.-E/sqrt(E^2+1.))
    Asqrem = 0.5*(1.+E/sqrt(E^2+1.))
    Q =  Asqr*sphi  +  Asqrem*semphi.'
    Qinv = inv(Q)
    Qde = 1/(2*(sqrt(E^2+1))^3)*(semphi.'-sphi) + Asqr*sdephi + Asqrem*semdephi.'

    phiProj1 = diagm(Complex{Float64}[0.5*1im,0.,0.,0.])
    phiProj2 = diagm(Complex{Float64}[0.,0.5*1im,0.,0.])

    comm1::Array{Complex{Float64},2} = sphi*phiProj1  -   phiProj1*sphi    #dS/dphi1
    comm2::Array{Complex{Float64},2} = sphi*phiProj2  -   phiProj2*sphi
    commem1 = semphi*phiProj1  -   phiProj1*semphi #E -> -E
    commem2 = semphi*phiProj2  -   phiProj2*semphi
    comme1::Array{Complex{Float64},2} = sdephi*phiProj1  -   phiProj1*sdephi  #d^2 S/ dedphi1
    comme2::Array{Complex{Float64},2} = sdephi*phiProj2  -   phiProj2*sdephi

    #comm(n) = ephiInv*s*ephi*phiProj(n)  -   phiProj(n)*ephiInv*s*ephi
    Qdphi1 =  Asqr*comm1 + Asqrem*commem1.'
    Qdphi2 =  Asqr*comm2 + Asqrem*commem2.'

    Bound::Array{Complex{Float64},2} = Asqr*(-Qinv*Qdphi2*Qinv*(sdephi*phiProj1+phiProj1*sdephi) + Qinv*comme2*phiProj1 + Qinv*phiProj1*comme2) - Asqr*(-Qinv*Qdphi1*Qinv*(sdephi*phiProj2+phiProj2*sdephi) + Qinv*comme1*phiProj2 + Qinv*phiProj2*comme1)

    #Qdphi(n) =  Asqr*comm(n) + Asqrem*comm(n)'

    inv_::Array{Complex{Float64},2} = 2*Bound
    return -0.25*trace(inv_)
end




function invarEcomp1(E_,x,y,z,scale_,s_) # invariant Q^{-1}.dQ/de.Q^{-1}.dQ/d(phi_1).Q^{-1}.dQ/d(phi_2), antisymmetrised upon 1,2
    E = Float64(E_)
    scale = Float64(scale_)
    s = Array{Complex{Float64}}(s_)
    F::Base.LinAlg.Eigen{Complex{Float64},Complex{Float64},Array{Complex{Float64},2},Array{Complex{Float64},1}} = eigfact(s)
    d::Array{Complex{Float64},1} = F[:values]
    v::Array{Complex{Float64},2} = F[:vectors]
    sgn::Array{Float64,2} = diagm(sign.(sin.(0.5*angle.(d))))
    e0 = scale*diagm(cos.(0.5*angle.(d)))*sgn
    gamma = 2*scale*diagm(sin.(0.5*angle.(d)))*sgn
    pole = e0 + 0.5*im*gamma
    S = v*((im*0.5*(1-E^2)*eye(4)-pole*E)/(im*0.5*(1-E^2)*eye(4)-E*conj(pole)))*v'
    Sem = v*((-im*0.5*(1-E^2)*eye(4)-pole*E)/(-im*0.5*(1-E^2)*eye(4)-E*conj(pole)))*v'  #energy depend S transpose has to change E -> -E
    Sde = v*(-(gamma*E^2)/((im*0.5*(1-E^2)*eye(4)-E*conj(pole))^2))*v'
    Semde = v*((gamma*E^2)/((-im*0.5*(1-E^2)*eye(4)-E*conj(pole))^2))*v'  #dS_{-E}/de
    ephi = diagm(exp.([0.5im*Float64(x),0.5im*Float64(y),0.5im*Float64(z),0.])) # diag matrix of phase-exponentials
    ephiInv = inv(ephi)
    sphi = ephiInv*S*ephi
    semphi = ephiInv*Sem*ephi  #sphi with -E
    sdephi = ephiInv*Sde*ephi
    semdephi = ephiInv*Semde*ephi   #sdephi with -E

    Asqr = E^2/(E^2+1)
    Asqrem = 1./(1+E^2)
    Q =  Asqr*sphi  +  Asqrem*semphi.'
    Qinv = inv(Q)
    Qde = 4*E^3/((1+E^2)^3)*(semphi.'-sphi) + Asqr*sdephi + Asqrem*semdephi.'

    phiProj1 = diagm(Complex{Float64}[0.5*1im,0.,0.,0.])
    phiProj2 = diagm(Complex{Float64}[0.,0.5*1im,0.,0.])

    comm1 = sphi*phiProj1  -   phiProj1*sphi   #dS/dphi1
    comm2 = sphi*phiProj2  -   phiProj2*sphi
    commem1 = semphi*phiProj1  -   phiProj1*semphi #E -> -E
    commem2 = semphi*phiProj2  -   phiProj2*semphi

    Qdphi1 =  Asqr*comm1 + Asqrem*commem1.'
    Qdphi2 =  Asqr*comm2 + Asqrem*commem2.'


    #Qdphi(n) =  Asqr*comm(n) + Asqrem*comm(n)'

    inv_::Array{Complex{Float64},2} = Qinv*Qde*Qinv*Qdphi1*Qinv*Qdphi2  -  Qinv*Qde*Qinv*Qdphi2*Qinv*Qdphi1
    return -0.25*trace(inv_)::Complex{Float64}
end

function invarEcomp2(E_,x,y,z,scale_,s_) # invariant Q^{-1}.dQ/de.Q^{-1}.dQ/d(phi_1).Q^{-1}.dQ/d(phi_2), antisymmetrised upon 1,2
    E = Float64(E_)
    scale = Float64(scale_)
    s = Array{Complex{Float64}}(s_)
    F::Base.LinAlg.Eigen{Complex{Float64},Complex{Float64},Array{Complex{Float64},2},Array{Complex{Float64},1}} = eigfact(s)
    d::Array{Complex{Float64},1} = F[:values]
    v::Array{Complex{Float64},2} = F[:vectors]
    sgn::Array{Float64,2} = diagm(sign.(sin.(0.5*angle.(d))))
    e0 = scale*diagm(cos.(0.5*angle.(d)))*sgn
    gamma = 2*scale*diagm(sin.(0.5*angle.(d)))*sgn
    pole = e0 + 0.5*im*gamma
    S = v*((im*0.5*(E^2-1)*eye(4)-pole*E)/(im*0.5*(E^2-1)*eye(4)-E*conj(pole)))*v'
    Sem = v*((-im*0.5*(E^2-1)*eye(4)-pole*E)/(-im*0.5*(E^2-1)*eye(4)-E*conj(pole)))*v'  #energy depend S transpose has to change E -> -E
    Sde = v*(-(gamma*E^2)/((im*0.5*(E^2-1)*eye(4)-E*conj(pole))^2))*v'
    Semde = v*((gamma*E^2)/((-im*0.5*(E^2-1)*eye(4)-E*conj(pole))^2))*v'  #dS_{-E}/de
    ephi = diagm(exp.([0.5im*Float64(x),0.5im*Float64(y),0.5im*Float64(z),0.])) # diag matrix of phase-exponentials
    ephiInv = inv(ephi)
    sphi = ephiInv*S*ephi
    semphi = ephiInv*Sem*ephi  #sphi with -E
    sdephi = ephiInv*Sde*ephi
    semdephi = ephiInv*Semde*ephi   #sdephi with -E

    Asqr = 1./(1+E^2)
    Asqrem = E^2/(1.+E^2)
    Q =  Asqr*sphi  +  Asqrem*semphi.'
    Qinv = inv(Q)
    Qde = 4*E^3/((1+E^2)^3)*(semphi.'-sphi) + Asqr*sdephi + Asqrem*semdephi.'

    phiProj1 = diagm(Complex{Float64}[0.5*1im,0.,0.,0.])
    phiProj2 = diagm(Complex{Float64}[0.,0.5*1im,0.,0.])

    comm1 = sphi*phiProj1  -   phiProj1*sphi   #dS/dphi1
    comm2 = sphi*phiProj2  -   phiProj2*sphi
    commem1 = semphi*phiProj1  -   phiProj1*semphi #E -> -E
    commem2 = semphi*phiProj2  -   phiProj2*semphi

    Qdphi1 =  Asqr*comm1 + Asqrem*commem1.'
    Qdphi2 =  Asqr*comm2 + Asqrem*commem2.'


    #Qdphi(n) =  Asqr*comm(n) + Asqrem*comm(n)'

    inv_::Array{Complex{Float64},2} = Qinv*Qde*Qinv*Qdphi1*Qinv*Qdphi2  -  Qinv*Qde*Qinv*Qdphi2*Qinv*Qdphi1
    return -0.25*trace(inv_)::Complex{Float64}
end

function boundarycomp1(E_,x,y,z,scale_,s_) # invariant Q^{-1}.dQ/de.Q^{-1}.dQ/d(phi_1).Q^{-1}.dQ/d(phi_2), antisymmetrised upon 1,2
    E = Float64(E_)
    scale = Float64(scale_)
    s = Array{Complex{Float64}}(s_)
    F::Base.LinAlg.Eigen{Complex{Float64},Complex{Float64},Array{Complex{Float64},2},Array{Complex{Float64},1}} = eigfact(s)
    d::Array{Complex{Float64},1} = F[:values]
    v::Array{Complex{Float64},2} = F[:vectors]
    sgn::Array{Float64,2} = diagm(sign.(sin.(0.5*angle.(d))))
    e0 = scale*diagm(cos.(0.5*angle.(d)))*sgn
    gamma = 2*scale*diagm(sin.(0.5*angle.(d)))*sgn
    pole = e0 + 0.5*im*gamma
    S = v*((im*0.5*(1-E^2)*eye(4)-pole*E)/(im*0.5*(1-E^2)*eye(4)-E*conj(pole)))*v'
    Sem = v*((-im*0.5*(1-E^2)*eye(4)-pole*E)/(-im*0.5*(1-E^2)*eye(4)-E*conj(pole)))*v'  #energy depend S transpose has to change E -> -E
    Sde = v*(-(gamma*E^2)/((im*0.5*(1-E^2)*eye(4)-E*conj(pole))^2))*v'
    Semde = v*((gamma*E^2)/((-im*0.5*(1-E^2)*eye(4)-E*conj(pole))^2))*v'  #dS_{-E}/de
    ephi = diagm(exp.([0.5im*Float64(x),0.5im*Float64(y),0.5im*Float64(z),0.])) # diag matrix of phase-exponentials
    ephiInv = inv(ephi)
    sphi = ephiInv*S*ephi
    semphi = ephiInv*Sem*ephi  #sphi with -E
    sdephi = ephiInv*Sde*ephi
    semdephi = ephiInv*Semde*ephi   #sdephi with -E

    Asqr = E^2/(E^2+1)
    Asqrem = 1./(1+E^2)
    Q =  Asqr*sphi  +  Asqrem*semphi.'
    Qinv = inv(Q)
    Qde = E^3/(2*(0.5*(1+E^2))^3)*(semphi.'-sphi) + Asqr*sdephi + Asqrem*semdephi.'

    phiProj1 = diagm(Complex{Float64}[0.5*1im,0.,0.,0.])
    phiProj2 = diagm(Complex{Float64}[0.,0.5*1im,0.,0.])

    comm1::Array{Complex{Float64},2} = sphi*phiProj1  -   phiProj1*sphi    #dS/dphi1
    comm2::Array{Complex{Float64},2} = sphi*phiProj2  -   phiProj2*sphi
    commem1 = semphi*phiProj1  -   phiProj1*semphi #E -> -E
    commem2 = semphi*phiProj2  -   phiProj2*semphi
    comme1::Array{Complex{Float64},2} = sdephi*phiProj1  -   phiProj1*sdephi  #d^2 S/ dedphi1
    comme2::Array{Complex{Float64},2} = sdephi*phiProj2  -   phiProj2*sdephi

    #comm(n) = ephiInv*s*ephi*phiProj(n)  -   phiProj(n)*ephiInv*s*ephi
    Qdphi1 =  Asqr*comm1 + Asqrem*commem1.'
    Qdphi2 =  Asqr*comm2 + Asqrem*commem2.'

    Bound::Array{Complex{Float64},2} = Asqr*(-Qinv*Qdphi2*Qinv*(sdephi*phiProj1+phiProj1*sdephi) + Qinv*comme2*phiProj1 + Qinv*phiProj1*comme2) - Asqr*(-Qinv*Qdphi1*Qinv*(sdephi*phiProj2+phiProj2*sdephi) + Qinv*comme1*phiProj2 + Qinv*phiProj2*comme1)

    #Qdphi(n) =  Asqr*comm(n) + Asqrem*comm(n)'

    inv_::Array{Complex{Float64},2} = 2*Bound
    return -0.25*trace(inv_)
end

function boundarycomp2(E_,x,y,z,scale_,s_) # invariant Q^{-1}.dQ/de.Q^{-1}.dQ/d(phi_1).Q^{-1}.dQ/d(phi_2), antisymmetrised upon 1,2
    E = Float64(E_)
    scale = Float64(scale_)
    s = Array{Complex{Float64}}(s_)
    F::Base.LinAlg.Eigen{Complex{Float64},Complex{Float64},Array{Complex{Float64},2},Array{Complex{Float64},1}} = eigfact(s)
    d::Array{Complex{Float64},1} = F[:values]
    v::Array{Complex{Float64},2} = F[:vectors]
    sgn::Array{Float64,2} = diagm(sign.(sin.(0.5*angle.(d))))
    e0 = scale*diagm(cos.(0.5*angle.(d)))*sgn
    gamma = 2*scale*diagm(sin.(0.5*angle.(d)))*sgn
    pole = e0 + 0.5*im*gamma
    S = v*((im*0.5*(E^2-1)*eye(4)-pole*E)/(im*0.5*(E^2-1)*eye(4)-E*conj(pole)))*v'
    Sem = v*((-im*0.5*(E^2-1)*eye(4)-pole*E)/(-im*0.5*(E^2-1)*eye(4)-E*conj(pole)))*v'  #energy depend S transpose has to change E -> -E
    Sde = v*(-(gamma*E^2)/((im*0.5*(E^2-1)*eye(4)-E*conj(pole))^2))*v'
    Semde = v*((gamma*E^2)/((-im*0.5*(E^2-1)*eye(4)-E*conj(pole))^2))*v'  #dS_{-E}/de
    ephi = diagm(exp.([0.5im*Float64(x),0.5im*Float64(y),0.5im*Float64(z),0.])) # diag matrix of phase-exponentials
    ephiInv = inv(ephi)
    sphi = ephiInv*S*ephi
    semphi = ephiInv*Sem*ephi  #sphi with -E
    sdephi = ephiInv*Sde*ephi
    semdephi = ephiInv*Semde*ephi   #sdephi with -E

    Asqr = 1./(1+E^2)
    Asqrem = E^2/(1.+E^2)
    Q =  Asqr*sphi  +  Asqrem*semphi.'
    Qinv = inv(Q)
    Qde = E^3/(2*(0.5*(1+E^2))^3)*(semphi.'-sphi) + Asqr*sdephi + Asqrem*semdephi.'

    phiProj1 = diagm(Complex{Float64}[0.5*1im,0.,0.,0.])
    phiProj2 = diagm(Complex{Float64}[0.,0.5*1im,0.,0.])

    comm1::Array{Complex{Float64},2} = sphi*phiProj1  -   phiProj1*sphi    #dS/dphi1
    comm2::Array{Complex{Float64},2} = sphi*phiProj2  -   phiProj2*sphi
    commem1 = semphi*phiProj1  -   phiProj1*semphi #E -> -E
    commem2 = semphi*phiProj2  -   phiProj2*semphi
    comme1::Array{Complex{Float64},2} = sdephi*phiProj1  -   phiProj1*sdephi  #d^2 S/ dedphi1
    comme2::Array{Complex{Float64},2} = sdephi*phiProj2  -   phiProj2*sdephi

    #comm(n) = ephiInv*s*ephi*phiProj(n)  -   phiProj(n)*ephiInv*s*ephi
    Qdphi1 =  Asqr*comm1 + Asqrem*commem1.'
    Qdphi2 =  Asqr*comm2 + Asqrem*commem2.'

    Bound::Array{Complex{Float64},2} = Asqr*(-Qinv*Qdphi2*Qinv*(sdephi*phiProj1+phiProj1*sdephi) + Qinv*comme2*phiProj1 + Qinv*phiProj1*comme2) - Asqr*(-Qinv*Qdphi1*Qinv*(sdephi*phiProj2+phiProj2*sdephi) + Qinv*comme1*phiProj2 + Qinv*phiProj2*comme1)

    #Qdphi(n) =  Asqr*comm(n) + Asqrem*comm(n)'

    inv_::Array{Complex{Float64},2} = 2*Bound
    return -0.25*trace(inv_)
end




function invarEindep(E::Float64, x::Float64, y::Float64, z::Float64, scale::Float64, s::Array{Complex{Float64},2}) # invariant Q^{-1}.dQ/de.Q^{-1}.dQ/d(phi_1).Q^{-1}.dQ/d(phi_2), antisymmetrised upon 1,2
    F::Base.LinAlg.Eigen{Complex{Float64},Complex{Float64},Array{Complex{Float64},2},Array{Complex{Float64},1}} = eigfact(s)
    d::Array{Complex{Float64},1} = F[:values]::Array{Complex{Float64},1}
    v::Array{Complex{Float64},2} = F[:vectors]::Array{Complex{Float64},2}
    sgn::Array{Float64,2} = diagm(sign.(sin.(0.5*angle.(d))))
    e0 = diagm(cos.(0.5*angle.(d)))*sgn
    gamma = 2.*diagm(sin.(0.5*angle.(d)))*sgn
    pole = e0 + 0.5*im*gamma
    S = v*((im*E*eye(4)-pole)/(im*E*eye(4)-conj(pole)))*v'
    Sem = v*((-im*E*eye(4)-pole)/(-im*E*eye(4)-conj(pole)))*v'  #energy depend S transpose has to change E -> -E
    Sde = v*(-gamma/((im*E*eye(4)-conj(pole))^2))*v'
    Semde = v*(gamma/((-im*E*eye(4)-conj(pole))^2))*v'  #dS_{-E}/de
    ephi = diagm(exp.([0.5im*x,0.5im*y,0.5im*z,0.])) # diag matrix of phase-exponentials
    ephiInv = inv(ephi)
    sphi = ephiInv*S*ephi
    semphi = ephiInv*Sem*ephi  #sphi with -E
    sdephi = ephiInv*Sde*ephi
    semdephi = ephiInv*Semde*ephi   #sdephi with -E

    Asqr = 0.5*(1.-scale*E/sqrt(scale^2*E^2+1.))
    Asqrem = 0.5*(1.+scale*E/sqrt(scale^2*E^2+1.))
    Q =  Asqr*sphi  +  Asqrem*semphi.'
    Qinv = inv(Q)
    Qde = Asqr*sdephi + Asqrem*semdephi.' + 1./(2.*(sqrt(scale^2*E^2+1.))^3)*(semphi.'-sphi)*scale

    phiProj1 = diagm(Complex{Float64}[0.5*1im,0.,0.,0.])
    phiProj2 = diagm(Complex{Float64}[0.,0.5*1im,0.,0.])

    comm1 = sphi*phiProj1  -   phiProj1*sphi   #dS/dphi1
    comm2 = sphi*phiProj2  -   phiProj2*sphi
    commem1 = semphi*phiProj1  -   phiProj1*semphi #E -> -E
    commem2 = semphi*phiProj2  -   phiProj2*semphi

    Qdphi1 =  Asqr*comm1 + Asqrem*commem1.'
    Qdphi2 =  Asqr*comm2 + Asqrem*commem2.'


    #Qdphi(n) =  Asqr*comm(n) + Asqrem*comm(n)'

    inv_::Array{Complex{Float64},2} = Qinv*Qde*Qinv*Qdphi1*Qinv*Qdphi2  -  Qinv*Qde*Qinv*Qdphi2*Qinv*Qdphi1
    return -0.25*trace(inv_)
end









function invarcut(ie,x,y,z,s) # invariant Q^{-1}.dQ/de.Q^{-1}.dQ/d(phi_1).Q^{-1}.dQ/d(phi_2), antisymmetrised upon 1,2
    ephi = diagm(exp.([0.5im*Float64(x),0.5im*Float64(y),0.5im*Float64(z),0.])) # diag matrix of phase-exponentials
    ephiInv = inv(ephi)
    sphi = ephiInv*s*ephi
    Asqr_p = 0.5*(1.-ie/sqrt(Float64(ie)^2-1.)) # A^2_e analy connti. e-> ie cut positive real, equivalent to A^2_-e negative real
    Asqr_m = 0.5*(1.+ie/sqrt(Float64(ie)^2-1.)) # A^2_e nagative real, equiv to A^2_-e posi. real
    Q_p =  Asqr_p*sphi  +  Asqr_m*sphi.' #Q analy connti. e-> ie cut positive real
    Q_m =  Asqr_m*sphi  +  Asqr_p*sphi.' #Q analy connti. e-> ie cut negative real
    Qp_inv = inv(Q_p)
    Qm_inv = inv(Q_m)
    Qp_de = 0.5*im/((Float64(ie)^2-1.)^1.5)*(sphi.'-sphi) #positive real dQ/de
    Qm_de = -0.5*im/((Float64(ie)^2-1.)^1.5)*(sphi.'-sphi)

    phiProj1 = diagm(Complex{Float64}[1im/2.,0.,0.,0.])
    phiProj2 = diagm(Complex{Float64}[0.,1im/2.,0.,0.])

    comm1 = sphi*phiProj1  -   phiProj1*sphi
    comm2 = sphi*phiProj2  -   phiProj2*sphi
    #comm(n) = ephiInv*s*ephi*phiProj(n)  -   phiProj(n)*ephiInv*s*ephi
    Qp_dphi1 =  Asqr_p*comm1 + Asqr_m*comm1.'
    Qm_dphi1 =  Asqr_m*comm1 + Asqr_p*comm1.'
    Qp_dphi2 =  Asqr_p*comm2 + Asqr_m*comm2.'
    Qm_dphi2 =  Asqr_m*comm2 + Asqr_p*comm2.'

    inv_p = Qp_inv*Qp_de*Qp_inv*Qp_dphi1*Qp_inv*Qp_dphi2  -  Qp_inv*Qp_de*Qp_inv*Qp_dphi2*Qp_inv*Qp_dphi1
    inv_m = Qm_inv*Qm_de*Qm_inv*Qm_dphi1*Qm_inv*Qm_dphi2  -  Qm_inv*Qm_de*Qm_inv*Qm_dphi2*Qm_inv*Qm_dphi1
    return -0.25*trace(inv_p-inv_m)
end


function avrginvar(x,y,z,s) #the average invariant part at the boundary: 0.5I_1 * S^\dag * 0.5I_2 S
    ephi = diagm(exp.([0.5im*Float64(x),0.5im*Float64(y),0.5im*Float64(z),0.])) # diag matrix of phase-exponentials
    ephiInv = inv(ephi)
    sphi = ephiInv*s*ephi
    phiProj1 = diagm(Complex{Float64}[0.5*im,0.,0.,0.])
    phiProj2 = diagm(Complex{Float64}[0.,0.5*im,0.,0.])
    avrg_=(phiProj1*sphi'*phiProj2*sphi-phiProj2*sphi'*phiProj1*sphi)
    return -0.5*trace(avrg_)/(2.*pi)
end





function ssconj(x,y,z,s) # sum over all the positive ABS energies
    e_phi(x,y) = exp.([x,y,0.5im*Float64(z),0]) # array list of phase-exponentials
    ephi = diagm(e_phi(1im*x/2.,1im*y/2.))
    ephiInv = inv(ephi)

    Lambda = ephiInv*s*ephi*ephi*conj(s)*ephiInv
    chi = real(-0.5im*log.(eigvals(Lambda)))
    positive = chi[chi.>0]
    return sum(cos.(positive))+(2-ndims(positive))
end

function LowestABS(x,y,z,s) # Lowest ABS energies
    ephi = diagm(exp.([0.5im*Float64(x),0.5im*Float64(y),0.5im*Float64(z),0.])) # diag matrix of phase-exponentials
    ephiInv = inv(ephi)

    Lambda = ephiInv*s*ephi*ephi*conj(s)*ephiInv
    chi = real(-0.5im*log.(eigvals(Lambda)))
    positive = chi[chi.>0]
    ABS = cos.(max(positive...))
    return ABS
end


function HighestABS(x,y,z,s) # Lowest ABS energies
    ephi = diagm(exp.([0.5im*Float64(x),0.5im*Float64(y),0.5im*Float64(z),0.])) # diag matrix of phase-exponentials
    ephiInv = inv(ephi)

    Lambda = ephiInv*s*ephi*ephi*conj(s)*ephiInv
    chi = real(-0.5im*log.(eigvals(Lambda)))
    positive = chi[chi.>0]
    ABS = cos.(min(positive...))
    return ABS
end

function HighestIs1(x,y,z,s) # Lowest ABS energies
    ephi = diagm(exp.([0.5im*Float64(x),0.5im*Float64(y),0.5im*Float64(z),0.])) # diag matrix of phase-exponentials
    ephiInv = inv(ephi)

    Lambda = ephiInv*s*ephi*ephi*conj(s)*ephiInv
    chi = real(-0.5im*log.(eigvals(Lambda)))
    positive = chi[chi.>0]
    ABS = cos.(min(positive...))
    if abs(ABS-1.)<0.00003
        return 1.
    else return 0.
    end
end

function LowestIs0(x,y,z,s) # Lowest ABS energies
    ephi = diagm(exp.([0.5im*Float64(x),0.5im*Float64(y),0.5im*Float64(z),0.])) # diag matrix of phase-exponentials
    ephiInv = inv(ephi)

    Lambda = ephiInv*s*ephi*ephi*conj(s)*ephiInv
    chi = real(-0.5im*log.(eigvals(Lambda)))
    positive = chi[chi.>0]
    ABS = cos.(max(positive...))
    if abs(ABS-0)<0.02
        return 1.
    else return 0.
    end
end

function HisL(x,y,z,s) # Lowest ABS energies
    ephi = diagm(exp.([0.5im*Float64(x),0.5im*Float64(y),0.5im*Float64(z),0.])) # diag matrix of phase-exponentials
    ephiInv = inv(ephi)

    Lambda = ephiInv*s*ephi*ephi*conj(s)*ephiInv
    chi = real(-0.5im*log.(eigvals(Lambda)))
    positive = chi[chi.>0]
    ABSL = cos.(max(positive...))
    ABSH = cos.(min(positive...))
    if abs(ABSL-ABSH)<0.02
        return 1.
    else return 0.
    end
end

function invarSum(x,y,z,s) #This should return the same value as function "intInvar" yet faster
    e_phi(x,y) = exp.([x,y,0.5im*Float64(z),0]) # array list of phase-exponentials
    ephi = diagm(e_phi(1im*x/2.,1im*y/2.))
    ephiInv = inv(ephi)
    sphi = ephiInv*s*ephi
    phiProj1 = diagm(Complex{Float64}[1im/2,0,0,0])
    phiProj2 = diagm(Complex{Float64}[0,1im/2,0,0])

    Lambda = sphi*conj(sphi)
    D,V = eig(Lambda)
    dLambda1 = 2*sphi*phiProj1*conj(sphi)-phiProj1*Lambda-Lambda*phiProj1
    dLambda2 = 2*sphi*phiProj2*conj(sphi)-phiProj2*Lambda-Lambda*phiProj2

    summation = 0         #This is summing over eigenvalues and eigenvectors' indeces
    for i = 1:4
        for j = 1:4
            if j == i
                continue
            end
            summation += 2*log.(D[i])/((D[i]-D[j])^2)*((V'[i,:].'*dLambda1*V[:,j])*(V'[j,:].'*dLambda2*V[:,i])
                        -(V'[i,:].'*dLambda2*V[:,j])*(V'[j,:].'*dLambda1*V[:,i]))
                        +1/(D[j]*(D[j]-D[i]))*((V'[i,:].'*dLambda1*V[:,j])*(V'[j,:].'*dLambda2*V[:,i])
                                    -(V'[i,:].'*dLambda2*V[:,j])*(V'[j,:].'*dLambda1*V[:,i]))
                        -(1-D[i]/D[j])*(1/(D[j]-D[i]))*((V'[i,:].'*(phiProj1-ephi*s.'*ephiInv*phiProj1*ephi*conj(s)*ephiInv)*V[:,j])*(V'[j,:].'*dLambda2*V[:,i])
                        -(V'[i,:].'*(phiProj2-ephi*s.'*ephiInv*phiProj2*ephi*conj(s)*ephiInv)*V[:,j])*(V'[j,:].'*dLambda1*V[:,i]))
        end
    end
    return -0.25*summation/(2.*pi)

end

function invarSum1(x,y,z,s) #This should return the same value as function "intSum" yet faster
    ephi = diagm(exp.([0.5im*Float64(x),0.5im*Float64(y),0.5im*Float64(z),0.])) # diag matrix of phase-exponentials
    ephiInv = inv(ephi)
    sphi = ephiInv*s*ephi
    phiProj1 = diagm(Complex{Float64}[1im/2,0,0,0])
    phiProj2 = diagm(Complex{Float64}[0,1im/2,0,0])

    Lambda = sphi*conj(sphi)
    D,V = eig(Lambda)
    dLambda1 = 2*sphi*phiProj1*conj(sphi)-phiProj1*Lambda-Lambda*phiProj1
    dLambda2 = 2*sphi*phiProj2*conj(sphi)-phiProj2*Lambda-Lambda*phiProj2

    P1 = V'*phiProj1*V
    P2 = V'*phiProj2*V
    Pbar1 = -conj(V')*sphi'*phiProj1*sphi*conj(V) #this Pbar[j,i] = <jbar| 0.5im*I |ibar> = -<i| s^T * 0.5im*I*s^* |j>
    Pbar2 = -conj(V')*sphi'*phiProj2*sphi*conj(V)

       #This is summing over eigenvalues and eigenvectors' indeces
    summation1 = 0
    for m = 1:4
        for n = 1:4
            if n == m
                continue
            end
            summation1 +=
             ((-2*log.(D[n]))/((D[m]-D[n])^2)*4*D[m]*D[n]*P1[m,n]*P2[n,m]+(2*log.(D[m]))/((D[m]-D[n])^2)*((D[m]+D[n])^2)*P1[m,n]*P2[n,m]+(2*log.(D[m])-2*log.(D[n]))/((D[m]-D[n])^2)*(2*D[m]*D[n]+2*D[m]^2)*Pbar1[n,m]*P2[n,m]) - ((-2*log.(D[n]))/((D[m]-D[n])^2)*4*D[m]*D[n]*P2[m,n]*P1[n,m]+(2*log.(D[m]))/((D[m]-D[n])^2)*((D[m]+D[n])^2)*P2[m,n]*P1[n,m]+(2*log.(D[m])-2*log.(D[n]))/((D[m]-D[n])^2)*(2*D[m]*D[n]+2*D[m]^2)*Pbar2[n,m]*P1[n,m]) + (-4*((D[m]+D[n])/(D[m]-D[n]))*P1[m,n]*P2[n,m]+(1/(D[n]*(D[n]-D[m])))*((D[m]+D[n])^2+4*D[m]*D[n])*P2[n,m]*Pbar1[n,m]) - (-4*((D[m]+D[n])/(D[m]-D[n]))*P2[m,n]*P1[n,m]+(1/(D[n]*(D[n]-D[m])))*((D[m]+D[n])^2+4*D[m]*D[n])*P1[n,m]*Pbar2[n,m])
        end
    end
    return -0.25*(summation1)/(2.*pi)

end














function invarSumP(x,y,z,s) # Invariant (Q^{-1}dQ)^3 summed over only Positive energies
    e_phi(x,y) = exp.([x,y,0.5im*Float64(z),0]) # array list of phase-exponentials
    ephi = diagm(e_phi(1im*x/2.,1im*y/2.))
    ephiInv = inv(ephi)
    phiProj1 = diagm(Complex{Float64}[1im/2,0,0,0])
    phiProj2 = diagm(Complex{Float64}[0,1im/2,0,0])

    Lambda = ephiInv*s*ephi*ephi*conj(s)*ephiInv
    D,V = eig(Lambda)
    chi = real(-0.5im*log.(D))
    dLambda1 = 2*ephiInv*s*ephi*phiProj1*ephi*conj(s)*ephiInv-phiProj1*Lambda-Lambda*phiProj1
    dLambda2 = 2*ephiInv*s*ephi*phiProj2*ephi*conj(s)*ephiInv-phiProj2*Lambda-Lambda*phiProj2

    summation = 0
    summation2 = 0         #This is summing over eigenvalues and eigenvectors' indeces
    for i = 1:4
        if chi[i]<0
            continue
        end
        for j = 1:4
            if j == i
                continue
            end
            summation += 4*log.(D[i])/((D[i]-D[j])^2)*((V'[i,:].'*dLambda1*V[:,j])*(V'[j,:].'*dLambda2*V[:,i])
                        -(V'[i,:].'*dLambda2*V[:,j])*(V'[j,:].'*dLambda1*V[:,i]))
        end
    end
    for m = 1:4
        if chi[m]>0
            continue
        end
        for n = 1:4
            if chi[n]<0
                continue
            end
            if n == m
                continue
            end
            summation2 += 1/(D[n]*(D[n]-D[m]))*((V'[m,:].'*dLambda1*V[:,n])*(V'[n,:].'*dLambda2*V[:,m])
                          -(V'[m,:].'*dLambda2*V[:,n])*(V'[n,:].'*dLambda1*V[:,m]))
                          -(1-D[m]/D[n])*(1/(D[n]-D[m]))*((V'[m,:].'*(phiProj1-ephi*s.'*ephiInv*phiProj1*ephi*conj(s)*ephiInv)*V[:,n])*(V'[n,:].'*dLambda2*V[:,m])
                          -(V'[m,:].'*(phiProj2-ephi*s.'*ephiInv*phiProj2*ephi*conj(s)*ephiInv)*V[:,n])*(V'[n,:].'*dLambda1*V[:,m]))
        end
    end
    return -0.25*(summation2)/(2.*pi)

end


function BerryP(x,y,z,s) # Berry curvature only positive ABS energies
    e_phi(x,y) = exp.([x,y,0.5im*Float64(z),0]) # array list of phase-exponentials
    ephi = diagm(e_phi(1im*x/2.,1im*y/2.))
    ephiInv = inv(ephi)
    phiProj1 = diagm(Complex{Float64}[1im/2,0,0,0])
    phiProj2 = diagm(Complex{Float64}[0,1im/2,0,0])

    Lambda = ephiInv*s*ephi*ephi*conj(s)*ephiInv
    D,V = eig(Lambda)
    chi = real(-0.5im*log.(D))
    dLambda1 = 2*ephiInv*s*ephi*phiProj1*ephi*conj(s)*ephiInv-phiProj1*Lambda-Lambda*phiProj1
    dLambda2 = 2*ephiInv*s*ephi*phiProj2*ephi*conj(s)*ephiInv-phiProj2*Lambda-Lambda*phiProj2

    summation = 0         #This is summing over eigenvalues and eigenvectors' indeces
    for i = 1:4
        if chi[i]<0
            continue
        end
        for j = 1:4
            if j == i
                continue
            end
            summation += 2im*pi/((D[i]-D[j])^2)*((V'[i,:].'*dLambda1*V[:,j])*(V'[j,:].'*dLambda2*V[:,i])
                        -(V'[i,:].'*dLambda2*V[:,j])*(V'[j,:].'*dLambda1*V[:,i]))
        end
    end
    return -0.25*summation/(2.*pi)

end


function BerryBranch(x,y,z,s) # Berry curvature only positive ABS energies
    e_phi(x,y) = exp.([x,y,0.5im*Float64(z),0]) # array list of phase-exponentials
    ephi = diagm(e_phi(1im*x/2.,1im*y/2.))
    ephiInv = inv(ephi)
    sphi = ephiInv*s*ephi
    phiProj1 = diagm(Complex{Float64}[1im/2,0,0,0])
    phiProj2 = diagm(Complex{Float64}[0,1im/2,0,0])

    Lambda = sphi*conj(sphi)
    D,V = eig(Lambda)
    dLambda1 = 2*sphi*phiProj1*conj(sphi)-phiProj1*Lambda-Lambda*phiProj1
    dLambda2 = 2*sphi*phiProj2*conj(sphi)-phiProj2*Lambda-Lambda*phiProj2

    P1 = V'*phiProj1*V
    P2 = V'*phiProj2*V
    Pbar1 = -conj(V')*sphi'*phiProj1*sphi*conj(V)
    Pbar2 = -conj(V')*sphi'*phiProj2*sphi*conj(V)

    summation = 0         #This is summing over eigenvalues and eigenvectors' indeces
    for i = 1:4
        if imag(D[i])<=0
            continue
        end
        for j = 1:4
            if j == i
                continue
            end # the 2 factor in "2* 2im* pi" comes from the factor in "2log<di | di>"
            summation += 2*2im*pi*((4*D[i]*D[j]*Pbar1[j,i]*Pbar2[i,j] + (D[i]+D[j])^2*P1[i,j]*P2[j,i] + 2*(D[i]*D[j]+D[j]^2)*P1[i,j]*Pbar2[i,j] + 2*(D[i]^2+D[i]*D[j])*P2[j,i]*Pbar1[j,i]) - (4*D[i]*D[j]*Pbar2[j,i]*Pbar1[i,j] + (D[i]+D[j])^2*P2[i,j]*P1[j,i] + 2*(D[i]*D[j]+D[j]^2)*P2[i,j]*Pbar1[i,j] + 2*(D[i]^2+D[i]*D[j])*P1[j,i]*Pbar2[j,i]))/((D[i]-D[j])^2)
        end
    end
    return -0.25*summation/(2.*pi)

end


function invarSumSec(x,y,z,s) #This should return the same value as function "intInvar" yet faster
    e_phi(x,y) = exp.([x,y,0.5im*Float64(z),0]) # array list of phase-exponentials
    ephi = diagm(e_phi(1im*x/2.,1im*y/2.))
    ephiInv = inv(ephi)
    sphi = ephiInv*s*ephi
    phiProj1 = diagm(Complex{Float64}[1im/2,0,0,0])
    phiProj2 = diagm(Complex{Float64}[0,1im/2,0,0])

    Lambda = sphi*conj(sphi)
    D,V = eig(Lambda)
    chi = real(-0.5im*log.(D))
    dLambda1 = 2*sphi*phiProj1*conj(sphi)-phiProj1*Lambda-Lambda*phiProj1
    dLambda2 = 2*sphi*phiProj2*conj(sphi)-phiProj2*Lambda-Lambda*phiProj2

    summation = 0         #This is summing over eigenvalues and eigenvectors' indeces
    for i = 1:4
        for j = 1:4
            if j == i
                continue
            end
            summation += 1/(D[j]*(D[j]-D[i]))*((V'[i,:].'*dLambda1*V[:,j])*(V'[j,:].'*dLambda2*V[:,i])
                        -(V'[i,:].'*dLambda2*V[:,j])*(V'[j,:].'*dLambda1*V[:,i]))   -(1-D[i]/D[j])*(1/(D[j]-D[i]))*((V'[i,:].'*(phiProj1-ephi*s.'*ephiInv*phiProj1*ephi*conj(s)*ephiInv)*V[:,j])*(V'[j,:].'*dLambda2*V[:,i])
                          -(V'[i,:].'*(phiProj2-ephi*s.'*ephiInv*phiProj2*ephi*conj(s)*ephiInv)*V[:,j])*(V'[j,:].'*dLambda1*V[:,i]))
        end
    end
    return -0.25*summation/(2.*pi)
end

function invarSumP1(x,y,z,s) # Invariant (Q^{-1}dQ)^3 summed over only Positive energies
    e_phi(x,y) = exp.([x,y,0.5im*Float64(z),0]) # array list of phase-exponentials
    ephi = diagm(e_phi(1im*x/2.,1im*y/2.))
    ephiInv = inv(ephi)
    phiProj1 = diagm(Complex{Float64}[1im/2,0,0,0])
    phiProj2 = diagm(Complex{Float64}[0,1im/2,0,0])

    Lambda = ephiInv*s*ephi*ephi*conj(s)*ephiInv
    D,V = eig(Lambda)
    chi = real(-0.5im*log.(D))
    dLambda1 = 2*ephiInv*s*ephi*phiProj1*ephi*conj(s)*ephiInv-phiProj1*Lambda-Lambda*phiProj1
    dLambda2 = 2*ephiInv*s*ephi*phiProj2*ephi*conj(s)*ephiInv-phiProj2*Lambda-Lambda*phiProj2

    summation = 0       #This is summing over eigenvalues and eigenvectors' indeces
    for i = 1:4
        if chi[i]>0
            continue
        end
        for j = 1:4
            if j == i
                continue
            end
            summation += 2*(2*log.(D[i])/((D[i]-D[j])^2)*((V'[i,:].'*dLambda1*V[:,j])*(V'[j,:].'*dLambda2*V[:,i])
                        -(V'[i,:].'*dLambda2*V[:,j])*(V'[j,:].'*dLambda1*V[:,i]))
                        +1/(D[j]*(D[j]-D[i]))*((V'[i,:].'*dLambda1*V[:,j])*(V'[j,:].'*dLambda2*V[:,i])
                          -(V'[i,:].'*dLambda2*V[:,j])*(V'[j,:].'*dLambda1*V[:,i]))
                          -(1-D[i]/D[j])*(1/(D[j]-D[i]))*((V'[i,:].'*(phiProj1-ephi*s.'*ephiInv*phiProj1*ephi*conj(s)*ephiInv)*V[:,j])*(V'[j,:].'*dLambda2*V[:,i])
                          -(V'[i,:].'*(phiProj2-ephi*s.'*ephiInv*phiProj2*ephi*conj(s)*ephiInv)*V[:,j])*(V'[j,:].'*dLambda1*V[:,i])))
        end
    end
    return -0.25*summation/(2.*pi)

end



using QuadGK
using Cuba


function ChernS(z::Float64,s::Array{Complex{Float64},2})
    result::Array{Float64,1}, = cuhre((x::Array{Float64,1},f::Array{Float64,1})->f[1] = ((2.*x[1]^2-2.*x[1]+1.)/((x[1]-x[1]^2)^2))*2*pi*real(invar((2.*x[1]-1.)/(x[1]-x[1]^2),2.*pi*x[2],2.*pi*x[3],z,s)::Complex{Float64}),3,abstol = 1e-12, reltol = 1e-10)

    return result[1] #the result should /2pi for energy integral and *(2pi)^2 for [0,1] phase compactify. So, overall 2pi
end

function ChernL(z::Float64,s::Array{Complex{Float64},2})
    result::Array{Float64,1}, = cuhre((x::Array{Float64,1},f::Array{Float64,1})->f[1] = 4*pi^2*real(avrginvar(2.*pi*x[1],2.*pi*x[2],z,s)::Complex{Float64}),2,abstol = 1e-8, reltol = 1e-7)

    return result[1] #the result should /2pi for energy integral and *(2pi)^2 for [0,1] phase compactify. So, overall 2pi
end

function ChernSum(z::Float64,s::Array{Complex{Float64},2})
    result::Array{Float64,1}, = cuhre((x::Array{Float64,1},f::Array{Float64,1})->f[1] = 4*pi^2*real(invarSum1(2.*pi*x[1],2.*pi*x[2],z,s)::Complex{Float64}),2,abstol = 1e-8, reltol = 1e-7)

    return result[1] #the result should /2pi for energy integral and *(2pi)^2 for [0,1] phase compactify. So, overall 2pi
end

function ChernBerry(z::Float64,s::Array{Complex{Float64},2})
    result::Array{Float64,1}, = cuhre((x::Array{Float64,1},f::Array{Float64,1})->f[1] = 4*pi^2*real(BerryBranch(2.*pi*x[1],2.*pi*x[2],z,s)::Complex{Float64}),2,abstol = 1e-8, reltol = 1e-7)

    return result[1] #the result should /2pi for energy integral and *(2pi)^2 for [0,1] phase compactify. So, overall 2pi
end

function ChernCut(z::Float64,s::Array{Complex{Float64},2})
    result::Array{Float64,1}, = cuhre((x::Array{Float64,1},f::Array{Float64,1})->f[1] = 1./((1.-x[1])^2)*2*pi*real(invarcut(1./(1.-x[1]),2.*pi*x[2],2.*pi*x[3],z,s)::Complex{Float64}) - 1./((1.-x[1])^2)*2*pi*real(invarcut(1+1e-10,2.*pi*x[2],2.*pi*x[3],z,s))*1e-10/(1./(1.-x[1])-1),3,abstol = 1e-8, reltol = 1e-7)

    return result[1] #the result should /2pi for energy integral and *(2pi)^2 for [0,1] phase compactify. So, overall 2pi
end

function ChernE(z::Float64,scale::Float64,s::Array{Complex{Float64},2})
    result1::Array{Float64,1}, = cuhre((x::Array{Float64,1},f::Array{Float64,1})->f[1] = 2*pi*real(invarEcomp2(x[1],2.*pi*x[2],2.*pi*x[3],z,scale,s)::Complex{Float64}*(x[1]^2+1)/(2.*x[1]^2)),3,atol = 1e-8, rtol = 1e-7)

    result2::Array{Float64,1}, = cuhre((x::Array{Float64,1},f::Array{Float64,1})->f[1] = 2*pi*real(invarEcomp1(x[1],2.*pi*x[2],2.*pi*x[3],z,scale,s)::Complex{Float64}*(x[1]^2+1)/(2.*x[1]^2)),3,abstol = 1e-8, reltol = 1e-7)

    return result1[1] + result2[1] #the result should /2pi for energy integral and *(2pi)^2 for [0,1] phase compactify. So, overall 2pi
end

function ChernEindep(z::Float64,scale::Float64,s::Array{Complex{Float64},2})
    result::Array{Float64,1}, = cuhre((x::Array{Float64,1},f::Array{Float64,1})->f[1] = 20*2*pi*real(invarEindep(20*x[1]-10,2.*pi*x[2],2.*pi*x[3],z,scale,s)::Complex{Float64}),3,abstol = 1e-8, reltol = 1e-7)

    return result[1] #the result should /2pi for energy integral and *(2pi)^2 for [0,1] phase compactify. So, overall 2pi
end

function ChernBoundE(z::Float64,scale::Float64,s::Array{Complex{Float64},2})
    result1::Array{Float64,1}, = cuhre((x::Array{Float64,1},f::Array{Float64,1})->f[1] = 2*pi*real(boundarycomp2(x[1],2.*pi*x[2],2.*pi*x[3],z,scale,s)::Complex{Float64}*(x[1]^2+1)/(2.*x[1]^2)),3,atol = 1e-8, rtol = 1e-7)

    result2::Array{Float64,1}, = cuhre((x::Array{Float64,1},f::Array{Float64,1})->f[1] = 2*pi*real(boundarycomp1(x[1],2.*pi*x[2],2.*pi*x[3],z,scale,s)::Complex{Float64}*(x[1]^2+1)/(2.*x[1]^2)),3,abstol = 1e-8, reltol = 1e-7)

    return result1[1] + result2[1] #the result should /2pi for energy integral and *(2pi)^2 for [0,1] phase compactify. So, overall 2pi
end


function intGrnd(x,y,z,s)
    intg, err = quadgk(e->ground(e,x,y,z,s), -50, 50, atol=0.0000001)
    return intg/(2.*pi)
end

function intCurrent1(x,y,z,s)
    intg, err = quadgk(e->current1(e,x,y,z,s), -50, 50, atol=0.0000001)
    return intg/(2.*pi)
end

function intCurrent2(x,y,z,s)
    intg, err = quadgk(e->current2(e,x,y,z,s), -50, 50, atol=0.0000001)
    return intg/(2.*pi)
end


function intInvar(x,y,z,s) # integrate invariant (Q^{-1}dQ)^3 w.r.t energy for phases (x,y), constant S matrix
    intgS::Complex{Float64}, err = quadgk(E->invar(E,x,y,z,s), -2., 2., atol=0.00000001)::Tuple{Complex{Float64},Float64}
    intgL1::Complex{Float64}, err = quadgk(E->invar(E,x,y,z,s), -50., -2., atol=0.00000001)::Tuple{Complex{Float64},Float64}
    intgL2::Complex{Float64}, err = quadgk(E->invar(E,x,y,z,s), 2., 50., atol=0.00000001)::Tuple{Complex{Float64},Float64}
    return (intgS+intgL1+intgL2)/(2.*pi)
end

function intInvarES(x::Float64, y::Float64, z::Float64, scale::Float64, s::Array{Complex{Float64},2}) # integrate invariant of the response function at all scale energy dependent scale<1
    intg1::Float64, = quadgk(E::Float64->real(invarE(E,x,y,z,scale,s)), -2.*scale, 2.*scale)::Tuple{Float64,Float64}
    intg2::Float64, = quadgk(E->real(invarE(E,x,y,z,scale,s)), -2., -2.*scale)::Tuple{Float64,Float64}
    intg3::Float64, = quadgk(E->real(invarE(E,x,y,z,scale,s)), -100., -2.)::Tuple{Float64,Float64}
    intg4::Float64, = quadgk(E->real(invarE(E,x,y,z,scale,s)), 2.*scale, 2.)::Tuple{Float64,Float64}
    intg5::Float64, = quadgk(E->real(invarE(E,x,y,z,scale,s)), 2., 100.)::Tuple{Float64,Float64}
    return (intg1+intg2+intg3+intg4+intg5)/(2.*pi)
end

function intInvarEB(x::Float64, y::Float64, z::Float64, scale::Float64, s::Array{Complex{Float64},2}) # integrate invariant of the response function at all scale energy dependent large scale>1
    intg1::Float64, = quadgk(E::Float64->real(invarE(E,x,y,z,scale,s)), -2., 2., atol=0.0000001)::Tuple{Float64,Float64}
    intg2::Float64, = quadgk(E->real(invarE(E,x,y,z,scale,s)), -2.*scale, -2., atol=0.0000001)::Tuple{Float64,Float64}
    intg3::Float64, = quadgk(E->real(invarE(E,x,y,z,scale,s)), 2.,2.*scale, atol=0.0000001)::Tuple{Float64,Float64}
    intg4::Float64, = quadgk(E->real(invarE(E,x,y,z,scale,s)), -100.*scale,-2.*scale, atol=0.0000001)::Tuple{Float64,Float64}
    intg5::Float64, = quadgk(E->real(invarE(E,x,y,z,scale,s)), 2.*scale, 100.*scale, atol=0.0000001)::Tuple{Float64,Float64}
    return (intg1+intg2+intg3+intg4+intg5)/(2.*pi)

end


function intBoundE(x,y,z,scale,s) # integrate boundary at all scale
    if scale < 1.
        intg1, err = quadgk(E->boundary(E,x,y,z,scale,s), -2.*scale, 2.*scale)::Tuple{Complex{Float64},Float64}
        intg2, err = quadgk(E->boundary(E,x,y,z,scale,s), -2., -2.*scale)::Tuple{Complex{Float64},Float64}
        intg3, err = quadgk(E->boundary(E,x,y,z,scale,s), -100., -2.)::Tuple{Complex{Float64},Float64}
        intg4, err = quadgk(E->boundary(E,x,y,z,scale,s), 2.*scale, 2.)::Tuple{Complex{Float64},Float64}
        intg5, err = quadgk(E->boundary(E,x,y,z,scale,s), 2., 100.)::Tuple{Complex{Float64},Float64}
        sum = (intg1+intg2+intg3+intg4+intg5)/(2.*pi)
    else
        intg1, err = quadgk(E->boundary(E,x,y,z,scale,s), -2., 2.)::Tuple{Complex{Float64},Float64}
        intg2, err = quadgk(E->boundary(E,x,y,z,scale,s), -2.*scale, -2.)::Tuple{Complex{Float64},Float64}
        intg3, err = quadgk(E->boundary(E,x,y,z,scale,s), 2.,2.*scale)::Tuple{Complex{Float64},Float64}
        intg4, err = quadgk(E->boundary(E,x,y,z,scale,s), -100.*scale,-2.*scale)::Tuple{Complex{Float64},Float64}
        intg5, err = quadgk(E->boundary(E,x,y,z,scale,s), 2.*scale, 100.*scale)::Tuple{Complex{Float64},Float64}
        sum = (intg1+intg2+intg3+intg4+intg5)/(2.*pi)
    end
    return sum
end


function intInvarEcomp(x::Float64,y::Float64,z::Float64,scale::Float64,s::Array{Complex{Float64},2}) # integrate invarE at all scale
    intg1::Complex{Float64}, = quadgk(E::Float64->invarEcomp1(E,x,y,z,scale,s)*(E^2+1)/(-2.*E^2), 0., 1.)::Tuple{Complex{Float64},Float64}
    intg2::Complex{Float64}, = quadgk(E::Float64->invarEcomp2(E,x,y,z,scale,s)*(E^2+1)/(2.*E^2), 0., 1.)::Tuple{Complex{Float64},Float64}
    return (-intg1+intg2)/(2.*pi)
end

function intBoundEcomp(x,y,z,scale,s) # integrate invarE at all scale
    intg1::Complex{Float64}, err = quadgk(E->boundarycomp1(E,x,y,z,scale,s)*(E^2+1)/(-2.*E^2), 0., 1.)::Tuple{Complex{Float64},Float64}
    intg2::Complex{Float64}, err = quadgk(E->boundarycomp2(E,x,y,z,scale,s)*(E^2+1)/(2.*E^2), 0., 1.)::Tuple{Complex{Float64},Float64}

    return (-intg1+intg2)/(2.*pi)
end








function intInvarEL(x,y,z,scale,s) # integrate invarE at large scale
    intg1::Complex{Float64}, err = quadgk(E->invarE(E,x,y,z,scale,s), 50., 100000., atol=0.00000001)::Tuple{Complex{Float64},Float64}
    intg2::Complex{Float64}, err = quadgk(E->invarE(E,x,y,z,scale,s), -100000., -50., atol=0.00000001)::Tuple{Complex{Float64},Float64}
    return (intg1+intg2)/(2.*pi)
end

function intBoundEL(x,y,z,scale,s) # integrate boundary at large scale
    intg1::Complex{Float64}, err = quadgk(E->boundary(E,x,y,z,scale,s), 50., 100000., atol=0.00000001)::Tuple{Complex{Float64},Float64}
    intg2::Complex{Float64}, err = quadgk(E->boundary(E,x,y,z,scale,s), -100000., -50., atol=0.00000001)::Tuple{Complex{Float64},Float64}

    return (intg1+intg2)/(2.*pi)
end

function intLargeE(x,y,z,scale,s) # integrate largeE
    intg::Complex{Float64}, err = quadgk(E->LargeE(E,x,y,z,scale,s), 20., 10000., atol=0.00000001)::Tuple{Complex{Float64},Float64}
    return -intg/(2.*pi)
end




function intInvarCut(x,y,z,s)
    intg, err = quadgk(ie->invarcut(ie,x,y,z,s), 1.00000001, 50.00000001, atol=0.00000001)
    return intg/(2.*pi)
end

function intInvarerr(x,y,z,s) # integrate invariant (Q^{-1}dQ)^3 w.r.t energy for phases (x,y), limit(-50, 50)
    intg, err = quadgk(e->invar(e,x,y,z,s), -50, 50, atol=0.0000001)
    return err/(2.*pi)
end







function invarSumB(x,y,z,s) #This should return the same value as function "intInvar" yet faster
    e_phi(x,y) = exp.([x,y,0.5im*Float64(z),0]) # array list of phase-exponentials
    ephi = diagm(e_phi(1im*x/2.,1im*y/2.))
    ephiInv = inv(ephi)
    phiProj1 = diagm(Complex{Float64}[1im/2,0,0,0])
    phiProj2 = diagm(Complex{Float64}[0,1im/2,0,0])

    Lambda = ephiInv*s*ephi*ephi*conj(s)*ephiInv
    D,V = eig(Lambda)
    dLambda1 = 2*ephiInv*s*ephi*phiProj1*ephi*conj(s)*ephiInv-phiProj1*Lambda-Lambda*phiProj1
    dLambda2 = 2*ephiInv*s*ephi*phiProj2*ephi*conj(s)*ephiInv-phiProj2*Lambda-Lambda*phiProj2

    dL1 = V'*dLambda1*V
    dL2 = V'*dLambda2*V

    summation = 0         #This is summing over eigenvalues and eigenvectors' indeces
    for i = 1:4
        for j = 1:4
            if j == i
                continue
            end
            summation += 2*log.(D[i])/((D[i]-D[j])^2)*(dL1[i,j]*dL2[j,i]
                        -dL2[i,j]*dL1[j,i])
        end
    end
    return -0.25*summation/(2.*pi)

end







#intInvar(1.2,1.3,s)

GrndGrid(xx,yy,z,s) = map(x->intGrnd(x...,z,s),Iterators.product(yy,xx))

Current1Grid(xx,yy,z,s) = map(x->intCurrent1(x...,z,s),Iterators.product(yy,xx))

Current2Grid(xx,yy,z,s) = map(x->intCurrent2(x...,z,s),Iterators.product(yy,xx))


InvarGrid(xx,yy,z,s) = map(x->intInvar(x...,z,s),Iterators.product(yy,xx))
#@time InvarGrid = pmap(x->intInvar(x...,s),Iterators.product(1:3,1:4))
InvarCutGrid(xx,yy,z,s) = map(x->intInvarCut(x...,z,s),Iterators.product(yy,xx))

InvarGriderr(xx,yy,z,s) = map(x->intInvarerr(x...,z,s),Iterators.product(yy,xx))

#intInvar(1.2,1.3)
avrgInvarGrid(xx,yy,z,s) = map(x->avrginvar(x...,z,s),Iterators.product(yy,xx))

LambdaGrid(xx,yy,z,s) = map(x->ssconj(x...,z,s),Iterators.product(yy,xx))
LowestABSGrid(xx,yy,z,s) = map(x->LowestABS(x...,z,s),Iterators.product(yy,xx))
HighestABSGrid(xx,yy,z,s) = map(x->HighestABS(x...,z,s),Iterators.product(yy,xx))
HighestIs1Grid(xx,yy,z,s) = map(x->HighestIs1(x...,z,s),Iterators.product(yy,xx))
LowestIs0Grid(xx,yy,z,s) = map(x->LowestIs0(x...,z,s),Iterators.product(yy,xx))
HisLGrid(xx,yy,z,s) = map(x->HisL(x...,z,s),Iterators.product(yy,xx))

InvarSumGrid(xx,yy,z,s) = map(x->invarSum(x...,z,s),Iterators.product(yy,xx))
InvarSumPGrid(xx,yy,z,s) = map(x->invarSumP(x...,z,s),Iterators.product(yy,xx))
InvarSum1Grid(xx,yy,z,s) = map(x->invarSum1(x...,z,s),Iterators.product(yy,xx))
InvarSumBGrid(xx,yy,z,s) = map(x->invarSumB(x...,z,s),Iterators.product(yy,xx))

invarSumP1Grid(xx,yy,z,s) = map(x->invarSumP1(x...,z,s),Iterators.product(yy,xx))

BerryPGrid(xx,yy,z,s) = map(x->BerryP(x...,z,s),Iterators.product(yy,xx))
BerryBranchGrid(xx,yy,z,s) = map(x->BerryBranch(x...,z,s),Iterators.product(yy,xx))
InvarSumSecGrid(xx,yy,z,s) = map(x->invarSumSec(x...,z,s),Iterators.product(yy,xx))

IntInvarESGrid(xx,yy,z,scale,s) = map(x->intInvarES(x...,z,scale,s),Iterators.product(yy,xx))
IntInvarEBGrid(xx,yy,z,scale,s) = map(x->intInvarEB(x...,z,scale,s),Iterators.product(yy,xx))
IntBoundEGrid(xx,yy,z,scale,s) = map(x->intBoundE(x...,z,scale,s),Iterators.product(yy,xx))

IntInvarEcompGrid(xx,yy,z,scale,s) = map(x->intInvarEcomp(x...,z,scale,s),Iterators.product(yy,xx))



export avrginvar, avrgInvarGrid, BerryBranchGrid, BerryPGrid, Current1Grid, Current2Grid, GrndGrid, HighestABSGrid, HighestIs1Grid, intBoundE, intBoundEcomp, IntBoundEGrid, intBoundEL, intInvar, intInvarE, intInvarE1, intInvarEcomp, intInvarEL, IntInvarEBGrid, IntInvarESGrid, IntInvarEcompGrid, intLargeE, invar, invarcut, InvarCutGrid, invarE, invarEcomp1, invarEcomp2, invarEGrid, InvarGrid, InvarGriderr, invarSum1, InvarSum1Grid, InvarSumBGrid, InvarSumGrid, InvarSumPGrid, invarSumP1Grid, InvarSumSecGrid, LambdaGrid, LowestABSGrid, uni, s, SE, HisLGrid, LowestIs0Grid, invarEindep, ChernE, ChernEindep, ChernBoundE

end

#using QDPHI
