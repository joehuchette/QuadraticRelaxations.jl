struct NeuralNetRelaxation <: AbstractRelaxation
    num_layers::Int
end

function reformulate_unary_quadratic_term!(
    model::MOI.ModelLike,
    reformulation::Reformulation{NeuralNetRelaxation},
    _x::VI,
    _y::VI,
    l::Float64,
    u::Float64,
)
    x, y = _map_to_interval_if_needed(model, _x, _y, l, u)
    # Secant cut: y <= x
    MOI.add_constraint(model, 1.0SV(x) - SV(y), MOI.GreaterThan(0.0))
    return _formulate_zero_one_quadratic!(
        model,
        x,
        y,
        reformulation.relaxation.num_layers,
        !reformulation.soc_lower_bound,
    )
end

function _formulate_zero_one_quadratic!(
    model::MOI.ModelLike,
    x::VI,
    y::VI,
    num_layers::Int,
    impose_lower_bound::Bool,
)
    z_v, z_c = MOI.add_constrained_variables(
        model,
        [MOI.ZeroOne() for i in 1:num_layers],
    )
    g_v, g_c = MOI.add_constrained_variables(
        model,
        [MOI.Interval(0.0, 1.0) for i in 1:num_layers],
    )

    gv = Dict(i => SV(g_v[i]) for i in 1:num_layers)
    gv[0] = SV(x)

    agg_aff = 1.0 * SV(y) - SV(x) + sum(gv[s] / 2.0^(2s) for s in 1:num_layers)

    if impose_lower_bound
        agg_c = MOI.add_constraint(model, agg_aff, MOI.EqualTo(0.0))
    else
        agg_c = MOI.add_constraint(model, agg_aff, MOI.LessThan(0.0))
    end

    for i in 1:num_layers
        α_i = SV(z_v[i])
        MOI.add_constraint(model, gv[i] - 2.0gv[i-1], MOI.LessThan(0.0))
        MOIU.normalize_and_add_constraint(
            model,
            gv[i] + 2.0gv[i-1] - 2.0,
            MOI.LessThan(0.0),
        )
        MOI.add_constraint(
            model,
            gv[i] + 2.0gv[i-1] - 2.0 * α_i,
            MOI.GreaterThan(0.0),
        )
        MOIU.normalize_and_add_constraint(
            model,
            gv[i] - 2.0gv[i-1] + 2.0 * α_i,
            MOI.GreaterThan(0.0),
        )
    end
    return z_v
end
