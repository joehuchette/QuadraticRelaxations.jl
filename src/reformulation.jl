struct Reformulation{T<:AbstractRelaxation,S<:AbstractShiftMethod}
    relaxation::T
    shift::S
    soc_lower_bound::Bool
end

function reformulate_quadratics!(
    model::JuMP.Model,
    reformulation::Reformulation,
)
    moi_backend = JuMP.backend(model)
    @assert typeof(moi_backend) <: MOIU.CachingOptimizer
    @assert moi_backend.state in (MOIU.NO_OPTIMIZER, MOIU.EMPTY_OPTIMIZER)
    vq_ci, vq_f, vq_s = vectorize_quadratics!(moi_backend.model_cache.model)
    obj = try
        MOI.get(model, MOI.ObjectiveFunction{SAF}())
        nothing
    catch InexactError
        MOI.get(
            model,
            MOI.ObjectiveFunction{SQF}(),
        )
    end
    # NOTE: Previously, this code only MOI.delete'd if obj === nothing. That
    # doesn't seem right.
    MOI.delete(moi_backend, vq_ci)
    _relax_quadratic!(moi_backend, reformulation, vq_f, vq_s, obj)
    MOIU.attach_optimizer(model)
    return nothing
end

# Assumption: Constraints we wish to reformulate have been bridged to the form
# `VectorQuadraticFunction`-in-`Nonpositives`.
function _relax_quadratic!(
    model,
    reformulation::Reformulation,
    f::VQF,
    s::MOI.Nonpositives,
    obj::Union{Nothing,SQF} = nothing,
)
    squared_vars = Dict{VI,VI}()
    # Walk all constraints in vector and populate squared_vars dict
    sqt_terms = vcat(
        [t -> t.scalar_term for t in f.quadratic_terms],
        obj.quadratic_terms,
    )
    for st in sqt_terms
        for x in (st.variable_index_1, st.variable_index_2)
            if !haskey(squared_vars, x)
                l, u = MOIU.get_bounds(model, Float64, x)
                y_lb = l < 0 < u ? 0.0 : min(l^2, u^2)
                y_ub = max(l^2, u^2)
                y_v, y_c = MOI.add_constrained_variable(
                    model,
                    MOI.Interval{Float64}(y_lb, y_ub),
                )
                squared_vars[x] = y_v
                if reformulation.soc_lower_bound
                    MOI.add_constraint(
                        model,
                        MOI.VectorOfVariables([y_v, x]),
                        MOI.SecondOrderCone(2),
                    )
                end
                reformulate_unary_quadratic_term!(
                    model,
                    reformulation,
                    x,
                    y_v,
                    l,
                    u,
                )
            end
        end
    end

    n = length(keys(squared_vars))
    q_funcs = MOIU.scalarize(f)
    canonical_index = Dict{VI,Int}()
    for q_func in q_funcs
        Q = _get_Q_matrix(q_func, canonical_index, n)
        δ_value = compute_diagonal_shift(Q, reformulation.shift)
        quad_shift = SQF(SAF[],
            [
                SQT(2 * δ_value[canonical_index[xi]], xi, xi) for xi in keys(canonical_index)
            ],
            0.0,
        )
        aff_shift = SAF(
            [
                SAT(
                    -δ_value[canonical_index[xi]],
                    squared_vars[xi],
                ) for xi in keys(canonical_index)
            ],
            0.0,
        )
        shifted_q_func = MOIU.canonical(q_func + quad_shift + aff_shift)
        MOIU.normalize_and_add_constraint(
            model,
            shifted_q_func,
            MOI.LessThan(0.0),
        )
    end
    # Now handle objective
    if obj !== nothing
        Q = _get_Q_matrix(obj, canonical_index, n)
        δ_value = compute_diagonal_shift(Q, reformulation.shift)
        quad_shift = SQF(
            SAT[],
            [
                SQT(2 * δ_value[canonical_index[xi]], xi, xi) for xi in keys(canonical_index)
            ],
            0.0,
        )
        aff_shift = SAF(
            [
                SAT(
                    -δ_value[canonical_index[xi]],
                    squared_vars[xi],
                ) for xi in keys(canonical_index)
            ],
            0.0,
        )
        shifted_q_func = MOIU.canonical(obj + quad_shift + aff_shift)
        MOI.set(
            model,
            MOI.ObjectiveFunction{SQF}(),
            shifted_q_func,
        )
    end

    return nothing
end
