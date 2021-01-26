struct ExactFormulation <: AbstractRelaxation end

function reformulate_unary_quadratic_term!(
    model::MOI.ModelLike,
    reformulation::Reformulation{ExactFormulation},
    x::VI,
    y::VI,
    l::Float64,
    u::Float64,
)
    MOI.add_constraint(
        model,
        1.0 * SV(y) - 1.0 * SV(x) * SV(x),
        MOI.EqualTo(0.0),
    )
    return nothing
end
