module AnovaFixedEffectModels


using Statistics, StatsBase, LinearAlgebra, Distributions, Reexport, Printf
@reexport using FixedEffectModels, AnovaBase
import StatsBase: fit!, fit
import StatsModels: TableRegressionModel, vectorize, width, apply_schema, 
                    ModelFrame, ModelMatrix, columntable, asgn

using AnovaBase: select_super_interaction, extract_contrasts, canonicalgoodnessoffit, subformula, dof_asgn, lrt_nested, ftest_nested, _diff, _diffn
import AnovaBase: anova, nestedmodels, anovatable, prednames, predictors, formula
using Tables: columntable

export anova_lfe, lfe

include("anova.jl")
include("fit.jl")
include("io.jl")

end
