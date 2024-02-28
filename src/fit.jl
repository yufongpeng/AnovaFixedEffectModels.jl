# ==========================================================================================================
# Backend funcion
predictors(model::T) where {T <: FixedEffectModel} = model.formula_schema.rhs.terms

# Variable dispersion
dof_pred(model::FixedEffectModel) = nobs(model) - dof_residual(model)
# define dof on NestedModels?

isfe(::AbstractTerm) = false
isfe(::FunctionTerm{typeof(fe)}) = true
isfe(term::InteractionTerm) = any(isfe, term.terms)

"""
    nestedmodels(modeltype::Type{FixedEffectModel}, f::FormulaTerm, data; null = true, kwargs...)

Generate nested models from modeltype, formula and data. The null model will be an empty model if the keyword argument null is true (default).
"""
function nestedmodels(modeltype::Type{FixedEffectModel}, f::FormulaTerm, data; null = true, kwargs...)
    fullm = lfe(f, data; kwargs...)
    predterms = predictors(fullm)
    hasintercept(predterms) || (length(predterms) > 1 ? (predterms = predterms[2:end]) : throw(ArgumentError("Empty model is given!")))
    feterms = filter(isfe, f.rhs)
    subm = ntuple(length(predterms) - 1) do i
        lfe(FormulaTerm(f.lhs, (predterms[1:i]..., feterms...)), data; kwargs...)
    end
    NestedModels(null ? (lfe(FormulaTerm(f.lhs, (ConstantTerm(0), feterms...)), data; kwargs...), subm..., fullm) : (subm..., fullm))
end
