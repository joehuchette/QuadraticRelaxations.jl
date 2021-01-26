module BoxQP

import JuMP

using LinearAlgebra

using QuadraticRelaxations

import MathOptInterface
const MOI = MathOptInterface

function parse_box_qp(filename::String)
    open(filename) do fp
        n = parse(Int, readline(fp))
        c = parse.(Float64, split(readline(fp)))
        @assert n == length(c)
        Q = zeros(Float64, n, n)
        for i in 1:n
            Q[:, i] = parse.(Float64, split(readline(fp)))
        end
        return Q, c
    end
end

function formulate_box_qp(
    Q::Matrix{T},
    c::Vector{T},
    optimizer_factory;
    epigraph::Bool = true,
) where {T}
    n = length(c)
    @assert n == size(Q, 1) == size(Q, 2)
    model = JuMP.Model(JuMP.with_optimizer(optimizer_factory))
    JuMP.@variable(model, 0 ≤ x[1:n] ≤ 1)
    if epigraph
        JuMP.@variable(model, y)
        JuMP.@constraint(model, y ≥ -0.5dot(x, Q * x) - dot(c, x))
        JuMP.@objective(model, Min, y)
    else
        JuMP.@objective(model, Min, -0.5dot(x, Q * x) - dot(c, x))
    end
    return model
end

function solve_and_verify(
    filename::String,
    optimizer_factory;
    num_layers::Int = 3,
    diagonalize::Bool = true,
    soc_lower_bound::Bool = false,
)
    Q, c = parse_box_qp(filename)
    model, x, y = formulate_box_qp(Q, c, optimizer_factory)
    QuadraticRelaxations.formulate_quadratics!(
        model,
        num_layers,
        diagonalize = diagonalize,
        soc_lower_bound = soc_lower_bound,
    )
    JuMP.optimize!(model)
    x_val = JuMP.value.(x)
    true_val = -0.5dot(x_val, Q * x_val) - dot(c, x_val)
    @show true_val, JuMP.value(y)
    @assert abs(true_val - JuMP.value(y)) < 1e-2
    return
end

function verify_quadratic(
    x_val::Vector{Float64},
    x_squared_approx::Vector{Float64},
    y_val::Float64,
    Q::Matrix{Float64},
    c::Vector{Float64},
    ϵ::Float64,
)
    for i in 1:length(x_val)
        component_error = abs(x_squared_approx[i] - x_val[i]^2)
        component_error ≤ ϵ || return false
        println("|y_$i-x_$(i)²| = ", component_error)
    end
    true_val = -0.5dot(x_val, Q * x_val) - dot(c, x_val)
    println(
        "|true_obj(opt_solution) - linearized_obj(opt_solution)| = ",
        abs(true_val - y_val),
    )
    @assert abs(true_val - y_val) < 1e-2
    return true
end

end  # module BoxQP
