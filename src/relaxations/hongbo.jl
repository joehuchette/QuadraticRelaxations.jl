struct HongboRelaxation <: AbstractRelaxation
    num_layers::Int
end

function reformulate_unary_quadratic_term!(
    model::MOI.ModelLike,
    reformulation::Reformulation{HongboRelaxation},
    _x::VI,
    _y::VI,
    l::Float64,
    u::Float64,
)
    num_layers = reformulation.relaxation.num_layers
    ξ_v = MOI.add_variables(model, num_layers)
    η_v = MOI.add_variables(model, num_layers)
    λ_1_v, λ_1_c = MOI.add_constrained_variables(
        model,
        [MOI.GreaterThan(0.0) for i in 1:num_layers],
    )
    λ_2_v, λ_2_c = MOI.add_constrained_variables(
        model,
        [MOI.GreaterThan(0.0) for i in 1:num_layers],
    )
    z_v, z_c = MOI.add_constrained_variables(
        model,
        [MOI.ZeroOne() for i in 1:num_layers],
    )
    x = SV(_x)
    y = SV(_y)
    ξ = [SV(ξ_v[i]) for i in 1:num_layers]
    η = [SV(η_v[i]) for i in 1:num_layers]
    λ_1 = [SV(λ_1_v[i]) for i in 1:num_layers]
    λ_2 = [SV(λ_2_v[i]) for i in 1:num_layers]
    z = [SV(z_v[i]) for i in 1:num_layers]

    θ_min = (
        if l > 0
            atan(l^2 - 1) / (2l)
        elseif l == 0
            -π / 2
        else
            atan(l^2 - 1) / (2l) - π
        end
    )
    θ_max = (
        if u > 0
            atan(u^2 - 1) / (2u)
        elseif u == 0
            -π / 2
        else
            atan(u^2 - 1) / (2u) - π
        end
    )
    @assert -3π / 2 < θ_min < π / 2
    @assert -3π / 2 < θ_max < π / 2
    θ_d = θ_max - θ_min
    θ_mid = (θ_max + θ_min) / 2

    ν = num_layers
    # RLP inequality (1)
    MOIU.normalize_and_add_constraint(
        model,
        y - (l + u) * x + l * u,
        MOI.LessThan(0.0),
    )
    # Equation (11)
    # Suggestion from Robert: Set this to equation
    MOIU.normalize_and_add_constraint(
        model,
        ξ[ν] * cos(θ_d / 2^(ν + 1)) + η[ν] * sin(θ_d / 2^(ν + 1)) -
        (y + 1.0) / 2.0 * cos(θ_d / 2^(ν + 1)),
        MOI.GreaterThan(0.0),
    )
    if !reformulation.soc_lower_bound
        # Equation 12
        MOIU.normalize_and_add_constraint(
            model,
            ξ[ν] * cos(θ_d / 2^ν) + η[ν] * sin(θ_d / 2^ν) - (y + 1.0) / 2.0,
            MOI.LessThan(0.0),
        )
        # Equation 13
        MOIU.normalize_and_add_constraint(
            model,
            ξ[ν] - (y + 1.0) / 2.0,
            MOI.LessThan(0.0),
        )
    end
    # Equation block (20)
    MOIU.normalize_and_add_constraint(
        model,
        ξ[1] - x * cos(θ_mid) - (y / 2.0 - 0.5) * sin(θ_mid),
        MOI.EqualTo(0.0),
    )
    C = max(l^2, u^2) / 2 + 0.5
    MOIU.normalize_and_add_constraint(
        model,
        (1.0λ_2[1] - 1.0λ_1[1]) * C + x * sin(θ_mid) -
        (y / 2.0 - 0.5) * cos(θ_mid),
        MOI.EqualTo(0.0),
    )
    MOI.add_constraint(
        model,
        η[1] - (1.0λ_1[1] + 1.0λ_2[1]) * C,
        MOI.EqualTo(0.0),
    )
    MOIU.normalize_and_add_constraint(
        model,
        1.0λ_1[1] - (1.0 - z[1]),
        MOI.LessThan(0.0),
    )
    MOI.add_constraint(model, 1.0λ_2[1] - 1.0z[1], MOI.LessThan(0.0))
    for j in 1:(ν-1)
        # Equation block 21
        MOI.add_constraint(
            model,
            ξ[j+1] - ξ[j] * cos(θ_d / 2^(j + 1)) - η[j] * sin(θ_d / 2^(j + 1)),
            MOI.EqualTo(0.0),
        )
        C_j = C * sin(θ_d / 2^(j + 1))
        MOI.add_constraint(
            model,
            (1.0λ_2[j+1] - 1.0λ_1[j+1]) * C_j + ξ[j] * sin(θ_d / 2^(j + 1)) -
            η[j] * cos(θ_d / 2^(j + 1)),
            MOI.EqualTo(0.0),
        )
        MOI.add_constraint(
            model,
            η[j+1] - (1.0λ_1[j+1] + 1.0λ_2[j+1]) * C_j,
            MOI.EqualTo(0.0),
        )
        MOIU.normalize_and_add_constraint(
            model,
            1.0λ_1[j] - (1.0 - z[j+1]),
            MOI.LessThan(0.0),
        )
        MOI.add_constraint(model, 1.0λ_2[j] - 1.0z[j+1], MOI.LessThan(0.0))
    end
end
