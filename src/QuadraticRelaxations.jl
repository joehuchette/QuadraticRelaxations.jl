module QuadraticRelaxations

import JuMP, MathOptInterface
const MOI = MathOptInterface
const MOIU = MOI.Utilities
const VI = MOI.VariableIndex
const SV = MOI.SingleVariable
const SAT = MOI.ScalarAffineTerm{Float64}
const SAF = MOI.ScalarAffineFunction{Float64}
const SQT = MOI.ScalarQuadraticTerm{Float64}
const SQF = MOI.ScalarQuadraticFunction{Float64}
const VQT = MOI.VectorQuadraticTerm{Float64}
const VQF = MOI.VectorQuadraticFunction{Float64}

import LinearAlgebra, SparseArrays
import Mosek, MosekTools
import PiecewiseLinearOpt

abstract type AbstractRelaxation end

abstract type AbstractShiftMethod end

include("moi_util.jl")
include("reformulation.jl")
include("shift.jl")
# include("diagonalize.jl")

include("relaxations/secant.jl")
include("relaxations/hongbo.jl")
include("relaxations/neural_net.jl")
include("relaxations/standard_nmdt.jl")
include("relaxations/tightened_nmdt.jl")

include("box_qp_util.jl")

end # module
