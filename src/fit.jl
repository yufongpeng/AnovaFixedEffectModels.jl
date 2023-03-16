# ==========================================================================================================
# Backend funcion
formula(model::T) where {T <: FixedEffectModel} = model.formula
predictors(model::T) where {T <: FixedEffectModel} = model.formula_schema.rhs.terms

# Variable dispersion
dof_pred(model::FixedEffectModel) = nobs(model) - dof_residual(model)
# define dof on NestedModels?