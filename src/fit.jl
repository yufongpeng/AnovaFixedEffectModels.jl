# ==========================================================================================================
# Backend funcion
predictors(model::T) where {T <: FixedEffectModel} = model.formula_schema.rhs.terms

# Variable dispersion
# dof_pred(model::FixedEffectModel) = nobs(model) - dof_residual(model)
dof_aovres(model::FixedEffectModel) = nobs(model) - dof_aov(model)
dof_aov(model::FixedEffectModel) = dof(model) + hasintercept(model.formula_schema) + dof_fes(model)
formula_aov(model::FixedEffectModel) = model.formula

# define dof on NestedModels?

isfe(::AbstractTerm) = false
isfe(::FunctionTerm{typeof(fe)}) = true
isfe(term::InteractionTerm) = any(isfe, term.terms)

"""
    nestedmodels(modeltype::Type{FixedEffectModel}, f::FormulaTerm, data; null = true, kwargs...)

Generate nested models from modeltype, formula and data. The null model will be an empty model if the keyword argument null is true (default).
"""
function nestedmodels(modeltype::Type{FixedEffectModel}, f::FormulaTerm, data, vcov::CovarianceEstimator = Vcov.simple(); null = true, kwargs...)
    fullm = lfe(f, data, vcov; kwargs...)
    predterms = predictors(fullm)
    if !hasintercept(predterms)
        length(predterms) == 1 && throw(ArgumentError("Empty model is given!"))
        null = false
    end
    feterms = filter(isfe, f.rhs)
    subm = ntuple(length(predterms) - 1) do i
        lfe(FormulaTerm(f.lhs, (predterms[1:i]..., feterms...)), data, vcov; kwargs...)
    end
    NestedModels(null ? (lfe(FormulaTerm(f.lhs, (ConstantTerm(0), feterms...)), data, vcov; kwargs...), subm..., fullm) : (subm..., fullm))
end
