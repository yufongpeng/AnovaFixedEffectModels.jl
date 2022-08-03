module AnovaFixedEffectModels


using Statistics, StatsBase, LinearAlgebra, Distributions, Reexport, Printf
@reexport using FixedEffectModels, AnovaBase
import StatsBase: fit!, fit
import StatsModels: TableRegressionModel, vectorize, width, apply_schema, 
                    ModelFrame, ModelMatrix, columntable, asgn
import AnovaBase: ftest_nested, formula, anova, nestedmodels, _diff, _diffn, dof, dof_residual, deviance, nobs, coefnames
using Tables: columntable

export anova_lfe, lfe, to_trm

include("anova.jl")
include("fit.jl")
include("io.jl")

end
