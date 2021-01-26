struct StandardNMDT <: AbstractRelaxation
    num_layers::Int
end

function reformulate_unary_quadratic_term!(
    model::MOI.ModelLike,
    reformulation::Reformulation{StandardNMDT},
    _x::VI,
    _y::VI,
    _x_l::Float64,
    _x_u::Float64,
)
    x, y = _map_to_interval_if_needed(model, _x, _y, _x_l, _x_u)
    x_l, x_u = 0.0, 1.0

    L = reformulation.relaxation.num_layers
    α_v, α_c =
        MOI.add_constrained_variables(model, [MOI.ZeroOne() for i in 1:L])
    Δx_v, Δx_c =
        MOI.add_constrained_variable(model, MOI.Interval(0.0, 2.0^(-L)))
    u_v = MOI.add_variables(model, L)
    Δz_v = MOI.add_variable(model)

    α = [SV(α_v[i]) for i in 1:L]
    u = [SV(u_v[i]) for i in 1:L]
    x = 1.0 * SV(x)
    y = 1.0 * SV(y)
    Δx = SV(Δx_v)
    Δz = SV(Δz_v)

    MOI.add_constraint(
        model,
        x - Δx - sum(2.0^(-i) * α[i] for i in 1:L),
        MOI.EqualTo(0.0),
    )
    MOI.add_constraint(
        model,
        y - Δz - sum(2.0^(-i) * u[i] for i in 1:L),
        MOI.EqualTo(0.0),
    )
    for i in 1:L
        a, b, c = x, α[i], u[i]
        a_l, a_u = x_l, x_u
        b_l, b_u = 0.0, 1.0
        MOIU.normalize_and_add_constraint(
            model,
            c - (a_u * b + a * b_u - a_u * b_u),
            MOI.GreaterThan(0.0),
        )
        MOIU.normalize_and_add_constraint(
            model,
            c - (a_l * b + a * b_l - a_l * b_l),
            MOI.GreaterThan(0.0),
        )
        MOIU.normalize_and_add_constraint(
            model,
            c - (a_u * b + a * b_l - a_u * b_l),
            MOI.LessThan(0.0),
        )
        MOIU.normalize_and_add_constraint(
            model,
            c - (a_l * b + a * b_u - a_l * b_u),
            MOI.LessThan(0.0),
        )
    end
    a, b, c = x, Δx, Δz
    a_l, a_u = x_l, x_u
    b_l, b_u = 0.0, 2.0^(-L)
    MOIU.normalize_and_add_constraint(
        model,
        c - (a_u * b + a * b_u - a_u * b_u),
        MOI.GreaterThan(0.0),
    )
    MOIU.normalize_and_add_constraint(
        model,
        c - (a_l * b + a * b_l - a_l * b_l),
        MOI.GreaterThan(0.0),
    )
    MOIU.normalize_and_add_constraint(
        model,
        c - (a_u * b + a * b_l - a_u * b_l),
        MOI.LessThan(0.0),
    )
    MOIU.normalize_and_add_constraint(
        model,
        c - (a_l * b + a * b_u - a_l * b_u),
        MOI.LessThan(0.0),
    )
    return α_v
end
