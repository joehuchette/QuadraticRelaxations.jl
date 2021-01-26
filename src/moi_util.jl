function _map_to_interval_if_needed(model, x::VI, y::VI, l::Float64, u::Float64)
    if l == 0 && u == 1
        return x, y
    else
        Δ = u - l
        @assert Δ > 0
        x̃_v, x̃_c = MOI.add_constrained_variable(model, MOI.Interval(0.0, 1.0))
        ỹ_v, ỹ_c = MOI.add_constrained_variable(model, MOI.Interval(0.0, 1.0))
        MOIU.normalize_and_add_constraint(
            model,
            SV(x) - (l + Δ * SV(x̃_v)),
            MOI.EqualTo(0.0),
        )
        MOIU.normalize_and_add_constraint(
            model,
            SV(y) - (l^2 + 2l * Δ * SV(x̃_v) + Δ^2 * SV(ỹ_v)),
            MOI.EqualTo(0.0),
        )
        return x̃_v, ỹ_v
    end
end

vectorize_quadratics!(model::MOIU.CachingOptimizer) = model.model_cache.model

function vectorize_quadratics!(cached_model::MOIU.Model{Float64})
    scalar_quad_cons = cached_model.moi_scalarquadraticfunction
    @assert isempty(scalar_quad_cons.moi_equalto)
    @assert isempty(scalar_quad_cons.moi_interval)
    @assert isempty(scalar_quad_cons.moi_semicontinuous)
    @assert isempty(scalar_quad_cons.moi_semiinteger)
    @assert isempty(scalar_quad_cons.moi_zeroone)

    num_gt = length(scalar_quad_cons.moi_greaterthan)
    num_lt = length(scalar_quad_cons.moi_lessthan)

    constants = Array{Float64}(undef, 0)
    affine_terms = Array{VAT}(undef, 0)
    quad_terms = Array{VQT}(undef, 0)

    func_count = 0
    for (q_ci, q_f, q_s) in scalar_quad_cons.moi_greaterthan
        func_count += 1
        push!(constants, -(q_f.constant - q_s.lower))
        for aff in q_f.affine_terms
            push!(
                affine_terms,
                VAT(func_count, SAT(-aff.coefficient, aff.variable_index)),
            )
        end
        for quad in q_f.quadratic_terms
            push!(
                quad_terms,
                VQT(
                    func_count,
                    SQT(
                        -quad.coefficient,
                        quad.variable_index_1,
                        quad.variable_index_2,
                    ),
                ),
            )
        end
    end

    @assert func_count == num_gt

    for (q_ci, q_f, q_s) in scalar_quad_cons.moi_lessthan
        func_count += 1
        push!(constants, q_f.constant - q_s.upper)
        for aff in q_f.affine_terms
            push!(
                affine_terms,
                VAT(func_count, SAT(aff.coefficient, aff.variable_index)),
            )
        end
        for quad in q_f.quadratic_terms
            push!(
                quad_terms,
                VQT(
                    func_count,
                    SQT(
                        quad.coefficient,
                        quad.variable_index_1,
                        quad.variable_index_2,
                    ),
                ),
            )
        end
    end

    @assert func_count == num_gt + num_lt

    vector_quad_cons = cached_model.moi_vectorquadraticfunction.moi_nonpositives
    @assert isempty(vector_quad_cons)
    constraint_index = MOI.ConstraintIndex{VQF,MOI.Nonpositives}(1)
    vector_func = VQF(affine_terms, quad_terms, constants)
    vector_set = MOI.Nonpositives(num_gt + num_lt)
    retval = (constraint_index, vector_func, vector_set)
    push!(vector_quad_cons, retval)
    empty!(scalar_quad_cons.moi_greaterthan)
    empty!(scalar_quad_cons.moi_lessthan)
    return retval
end

function _indices_and_coefficients(
    I::AbstractVector{Int},
    J::AbstractVector{Int},
    V::AbstractVector{Float64},
    indices::AbstractVector{Int},
    coefficients::AbstractVector{Float64},
    f::SQF,
    canonical_index::Dict{VI,Int},
)
    variables_seen = 0
    for (i, term) in enumerate(f.quadratic_terms)
        vi_1 = term.variable_index_1
        vi_2 = term.variable_index_2
        if !haskey(canonical_index, vi_1)
            variables_seen += 1
            canonical_index[vi_1] = variables_seen
        end
        if !haskey(canonical_index, vi_2)
            variables_seen += 1
            canonical_index[vi_2] = variables_seen
        end

        I[i] = canonical_index[term.variable_index_1]
        J[i] = canonical_index[term.variable_index_2]
        V[i] = term.coefficient
        # Gurobi returns a list of terms. MOI requires 0.5 x' Q x. So, to get
        # from
        #   Gurobi -> MOI => multiply diagonals by 2.0
        #   MOI -> Gurobi => multiply diagonals by 0.5
        # Example: 2x^2 + x*y + y^2
        #   |x y| * |a b| * |x| = |ax+by bx+cy| * |x| = 0.5ax^2 + bxy + 0.5cy^2
        #           |b c|   |y|                   |y|
        #   Gurobi needs: (I, J, V) = ([0, 0, 1], [0, 1, 1], [2, 1, 1])
        #   MOI needs:
        #     [SQT(4.0, x, x), SQT(1.0, x, y), SQT(2.0, y, y)]
        if I[i] == J[i]
            V[i] *= 0.5
        end
    end
    for (i, term) in enumerate(f.affine_terms)
        indices[i] = term.variable_index.value
        coefficients[i] = term.coefficient
    end
    return
end

function _indices_and_coefficients(f::SQF, canonical_index::Dict{VI,Int})
    f_canon = MOI.Utilities.canonical(f)
    nnz_quadratic = length(f_canon.quadratic_terms)
    nnz_affine = length(f_canon.affine_terms)
    I = Vector{Int}(undef, nnz_quadratic)
    J = Vector{Int}(undef, nnz_quadratic)
    V = Vector{Float64}(undef, nnz_quadratic)
    indices = Vector{Int}(undef, nnz_affine)
    coefficients = Vector{Float64}(undef, nnz_affine)
    _indices_and_coefficients(
        I,
        J,
        V,
        indices,
        coefficients,
        f_canon,
        canonical_index,
    )
    return indices, coefficients, I, J, V
end

function _get_Q_matrix(q_func::SQF, canonical_index::Dict{VI,Int}, n::Int)
    indices, coefficients, I, J, V =
        _indices_and_coefficients(q_func, canonical_index)
    Q = Matrix{Float64}(SparseArrays.sparse(I, J, V, n, n))
    Q = 1 / 2 * (Q + Q')

    @assert n == size(Q, 1) == size(Q, 2)
    @assert LinearAlgebra.issymmetric(Q)
    return Q
end
