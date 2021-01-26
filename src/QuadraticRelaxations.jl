module QuadraticRelaxations

import JuMP, MathOptInterface
import LinearAlgebra, SparseArrays
import PiecewiseLinearOpt

const MOI = MathOptInterface
const MOIU = MOI.Utilities
const VI = MOI.VariableIndex
const SV = MOI.SingleVariable
const SAT = MOI.ScalarAffineTerm{Float64}
const SAF = MOI.ScalarAffineFunction{Float64}
const SQT = MOI.ScalarQuadraticTerm{Float64}
const SQF = MOI.ScalarQuadraticFunction{Float64}
const VAT = MOI.VectorAffineTerm{Float64}
const VAF = MOI.VectorAffineFunction{Float64}
const VQT = MOI.VectorQuadraticTerm{Float64}
const VQF = MOI.VectorQuadraticFunction{Float64}

abstract type AbstractRelaxation end

abstract type AbstractShiftMethod end

include("moi_util.jl")
include("reformulation.jl")
include("shift.jl")

include("relaxations/exact.jl")
include("relaxations/secant.jl")
include("relaxations/cda.jl")
include("relaxations/neural_net.jl")
include("relaxations/standard_nmdt.jl")
include("relaxations/tightened_nmdt.jl")

end # module
