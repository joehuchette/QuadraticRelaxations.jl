struct MinEigenvalueShift <: AbstractShiftMethod end

function compute_diagonal_shift(Q::Matrix{Float64}, ::MinEigenvalueShift)
    @assert LinearAlgebra.issymmetric(Q)
    n = size(Q, 1)
    shift = -LinearAlgebra.eigmin(Q) .* ones(n)
    @assert LinearAlgebra.eigmin(Q + LinearAlgebra.diagm(shift)) >= -1e-4
    return shift
end

struct SemidefiniteShift <: AbstractShiftMethod
    sdp_optimizer_factory::Any
end

function compute_diagonal_shift(Q::Matrix{Float64}, shift::SemidefiniteShift)
    @assert LinearAlgebra.issymmetric(Q)
    n = size(Q, 1)
    sdp_model = JuMP.Model(shift.sdp_optimizer_factory)
    δ = JuMP.@variable(sdp_model, [i = 1:n])
    JuMP.@objective(sdp_model, Min, sum(δ))
    JuMP.@SDconstraint(sdp_model, Q + LinearAlgebra.diagm(δ .- 1e-4) >= 0)
    JuMP.optimize!(sdp_model)
    @assert JuMP.primal_status(sdp_model) == MOI.FEASIBLE_POINT
    shift = JuMP.value.(δ)
    @assert LinearAlgebra.eigmin(Q + LinearAlgebra.diagm(shift)) >= -1e-4
    return shift
end
