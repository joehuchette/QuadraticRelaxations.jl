struct MinEigenvalueShift <: AbstractShiftMethod end

function compute_diagonal_shift(Q::Matrix{Float64}, ::MinEigenvalueShift)
    @assert issymmetric(Q)
    n = size(Q, 1)
    return LinearAlgebra.eigmin(Q) .* ones(n)
end

struct SemidefiniteShift <: AbstractShiftMethod
    sdp_optimizer::MOI.AbstractOptimizer
end

function compute_diagonal_shift(Q::Matrix{Float64}, shift::SemidefiniteShift)
    @assert issymmetric(Q)
    n = size(Q, 1)
    sdp_model = JuMP.Model(shift.sdp_optimizer)
    δ = JuMP.@variable(sdp_model, [i = 1:n])
    JuMP.@objective(sdp_model, Min, sum(δ))
    JuMP.@SDconstraint(sdp_model, Q + LinearAlgebra.diagm(δ .- 1e-4) >= 0)
    JuMP.optimize!(sdp_model)
    @assert JuMP.primal_status(sdp_model) == MOI.FEASIBLE_POINT
    return JuMP.value.(δ)
end
