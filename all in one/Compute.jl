module COMPUTE

#Find your own path where you save the julia core file
include("/Users/yuguangchen/Projects/Tex/new structure/simulation/non-zero phi3/s matrix generating/all in one/core.jl")



using PyCall
#If you use Python 3, replace all "cPickle" to "pickle"
@pyimport pickle



#define global variables
N = 200
x = linspace(0,2*pi,N)
y = linspace(0,2*pi,N)



#dump the new data to filename
function mypickle(filename, obj)
    out = open(filename,"w")
    pickle.dump(obj, out, protocol=pickle.HIGHEST_PROTOCOL)
    close(out)
end

#read saved s matrix from the filename. You have to use myunpickle(\path\filename.extension) to properly call it.
function myunpickle(filename)
    r = nothing
    @pywith pybuiltin("open")(filename,"rb") as f begin
        r = pickle.load(f)
    end
    return r
end



#find your own file of "S.pickle" file, which specifies the s matrix
#Be aware of the naming system. The example uses particular the symmtric one near level crossing, and candidate number 1.
s = myunpickle("/Users/yuguangchen/Projects/Tex/new structure/simulation/non-zero phi3/s matrix generating/all in one/S matrix/S_weyl_1.pickle")


#energy independent or energy slowly varying S matrix

#Summation of small scale of the invariant
#Be aware of the naming system when you save the array.
function summation(phi3::Float64)
    InvarSum = CORE.InvarSum1Grid(x,y,phi3,s::Array{Complex{Float64},2})
    mypickle("/Users/yuguangchen/Projects/Tex/new structure/simulation/non-zero phi3/s matrix generating/all in one/Summation/Summation_S_weyl_1.pickle", InvarSum)
end

#Integrated summation over 2 phases
function chernsum(phi3::Float64)
    Chern = CORE.ChernSum(phi3,s::Array{Complex{Float64},2})
    mypickle("/Users/yuguangchen/Projects/Tex/new structure/simulation/non-zero phi3/s matrix generating/all in one/ChernSum/Chernsum_S_sym_weyl_1.pickle", Chern)
end


#Non-topologogical contribution -- the large scale conductant, integrated over energy
function landauer(phi3::Float64)
    conductance = CORE.avrgInvarGrid(x,y,phi3,s::Array{Complex{Float64},2})
    mypickle("/Users/yuguangchen/Projects/Tex/new structure/simulation/non-zero phi3/s matrix generating/all in one/Landauer/Landauer_S_weyl_1.pickle", conductance)
end

#Landuaer conductance integrated with phases.
function chernlandauer(phi3::Float64)
    Chern = CORE.ChernL(phi3,s::Array{Complex{Float64},2})
    mypickle("/Users/yuguangchen/Projects/Tex/new structure/simulation/non-zero phi3/s matrix generating/all in one/ChernLandauer/Chernlandauer_S_sym_weyl_1.pickle", Chern)
end


#Berry curvature of discrete energy levels only
function berry(phi3::Float64)
    berry = CORE.BerryBranchGrid(x,y,phi3,s::Array{Complex{Float64},2})
    mypickle("/Users/yuguangchen/Projects/Tex/new structure/simulation/non-zero phi3/s matrix generating/all in one/Berry/Berry_S_weyl_1.pickle", berry)
end

#curvature of discrete energy -- berry -- integrated over 2 phases
function chernberry(phi3::Float64)
    Chern = CORE.ChernBerry(x,y,phi3,s::Array{Complex{Float64},2})
    mypickle("/Users/yuguangchen/Projects/Tex/new structure/simulation/non-zero phi3/s matrix generating/all in one/ChernBerry/Chernberry_S_sym_weyl_1.pickle", Chern)
end

#Lowest ABS
function lowestabs(phi3::Float64)
    abs = CORE.LowestABSGrid(x,y,phi3,s::Array{Complex{Float64},2})
    mypickle("/Users/yuguangchen/Projects/Tex/new structure/simulation/non-zero phi3/s matrix generating/all in one/LowestABS/Lowestabs_S_weyl_1.pickle", abs)
end

#HighestABS
function highestabs(phi3::Float64)
    abs = CORE.HighestABSGrid(x,y,phi3,s::Array{Complex{Float64},2})
    mypickle("/Users/yuguangchen/Projects/Tex/new structure/simulation/non-zero phi3/s matrix generating/all in one/HighestABS/Highestabs_S_weyl_1.pickle", abs)
end



#energy dependent S matrix

#invariant part of a small-scale energy dependent S matriz (scale<1)
function invariantsmall(phi3::Float64,scale::Float64)
    curvature = CORE.IntInvarESGrid(x,y,phi3,scale,s::Array{Complex{Float64},2})
    mypickle("/Users/yuguangchen/Projects/Tex/new structure/simulation/non-zero phi3/s matrix generating/all in one/S_e small_curv/Invariantsmall_S_sym_weyl_1.pickle", curvature)
end

#invariant part of a large-scale energy dependent S matriz (scale>1)
function invariantlarge(phi3::Float64,scale::Float64)
    curvature = CORE.IntInvarEBGrid(x,y,phi3,scale,s::Array{Complex{Float64},2})
    mypickle("/Users/yuguangchen/Projects/Tex/new structure/simulation/non-zero phi3/s matrix generating/all in one/S_e_large_curv/Invariantlarge_S_weyl_1.pickle", curvature)
end

#Energy dependent curvature integrated over phases -- Chern number
function cherns_e(phi3::Float64, scale::Float64)
    Chern = CORE.ChernE(phi3, scale, s::Array{Complex{Float64},2})
    mypickle("/Users/yuguangchen/Projects/Tex/new structure/simulation/non-zero phi3/s matrix generating/all in one/ChernS_e/Cherns_e_S_weyl_1.pickle", Chern)
end


end
