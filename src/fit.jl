# ==========================================================================================================
# Backend funcion

fe_intercept(f::FormulaTerm) = fe_intercept(f.rhs)
fe_intercept(term::StatsModels.TupleTerm) = map(fe_intercept, term)
fe_intercept(term::FunctionTerm) = first(term.exorig.args) == :fe 
fe_intercept(term) = false

width_fe(term::FunctionTerm) = first(term.exorig.args) == :fe ? 0 : 1
width_fe(ts::InteractionTerm) = prod(width_fe(t) for t in ts.terms)
width_fe(term) = width(term)

# Variable dispersion
dof(trm::TableRegressionModel{<: FixedEffectModel}) = trm.model.nobs - trm.model.dof_residual + 1