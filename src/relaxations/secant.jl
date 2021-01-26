struct SecantRelaxation <: AbstractRelaxation end

function reformulate_unary_quadratic_term!(
    model::MOI.ModelLike,
    reformulation::Reformulation{SecantRelaxation},
    _x::VI,
    _y::VI,
    l::Float64,
    u::Float64,
)
    x, y = _map_to_interval_if_needed(model, _x, _y, l, u)
    # Secant cut: y <= x
    MOI.add_constraint(model, 1.0SV(x) - SV(y), MOI.GreaterThan(0.0))
    return nothing
end
