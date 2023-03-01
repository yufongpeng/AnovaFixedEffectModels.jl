# ================================================================================================
# Main API
"""
    anova(<lfemodels>...; test::Type{<: GoodnessOfFit})
    anova(test::Type{<: GoodnessOfFit}, <lfemodels>...;  <keyword arguments>)

Analysis of variance.

Return `AnovaResult{M, test, N}`. See [`AnovaResult`](@ref) for details.

# Arguments
* `lfemodels`: model objects
    1. `FixedEffectModel` fitted by `AnovaFixedEffectModels.lfe` or `FixedEffectModels.reg`.
    If mutiple models are provided, they should be nested and the last one is the most complex.
* `test`: test statistics for goodness of fit. Only `FTest` is available now.

## Other keyword arguments
* When one model is provided:  
    1. `type` specifies type of anova. Default value is 1.
* When multiple models are provided:  
    1. `check`: allows to check if models are nested. Defalut value is true. Some checkers are not implemented now.

!!! note
    For fitting new models and conducting anova at the same time, see [`anova_lfe`](@ref) for `FixedEffectModel`.
"""
anova(::Type{<: GoodnessOfFit}, ::Vararg{FixedEffectModel})

anova(models::Vararg{M}; 
        test::Type{<: GoodnessOfFit} = FTest,
        kwargs...) where {M <: FixedEffectModel} = 
    anova(test, models...; kwargs...)

# ================================================================================================
# ANOVA by F test
anova(::Type{FTest}, 
    model::M; 
    type::Int = 1) where {M <: FixedEffectModel} = anova(FTest, FullModel(model, type, true, true))

function anova(::Type{FTest}, 
            aovm::FullModel{M}) where {M <: FixedEffectModel}

    assign = asgn(predictors(aovm))
    fullasgn = asgn(predictors(aovm.model))
    df = filter(>(0), dof_asgn(assign))
    # May exist some floating point error from dof_residual
    varβ = vcov(aovm.model)
    β = aovm.model.coef
    offset = first(assign) + last(fullasgn) - last(assign) - 1
    if aovm.type == 1
        fs = abs2.(cholesky(Hermitian(inv(varβ))).U * β)
        fstat = ntuple(last(fullasgn) - offset) do fix
            sum(fs[findall(==(fix + offset), fullasgn)]) / df[fix]
        end
    elseif aovm.type == 2
        fstat = ntuple(last(fullasgn) - offset) do fix
            select1 = sort!(collect(select_super_interaction(f.rhs, fix + offset)))
            select2 = setdiff(select1, fix + offset)
            select1 = findall(in(select1), fullasgn)
            select2 = findall(in(select2), fullasgn)
            (β[select1]' * (varβ[select1, select1] \ β[select1]) - β[select2]' * (varβ[select2, select2] \ β[select2])) / df[fix]
        end
    else
        # calculate block by block
        fstat = ntuple(last(fullasgn) - offset) do fix
            select = findall(==(fix + offset), fullasgn)
            β[select]' * (varβ[select, select] \ β[select]) / df[fix]
        end
    end
    dfr = round(Int, dof_residual(aovm.model))
    σ² = rss(aovm.model) / dfr
    devs = @. fstat * σ² * df
    pvalue = @. ccdf(FDist(df, dfr), abs(fstat))
    AnovaResult{FTest}(aovm, df, devs, fstat, pvalue, NamedTuple())
end

# =================================================================================================================
# Nested models 

function anova(::Type{FTest}, 
                models::Vararg{M}; 
                check::Bool = true) where {M <: FixedEffectModel}

    df = dof_pred.(models)
    ord = sortperm(collect(df))
    df = df[ord]
    models = models[ord]
    # May exist some floating point error from dof_residual
    dfr = round.(Int, dof_residual.(models))
    dev = rss.(models)
    # check comparable and nested
    check && @warn "Could not check whether models are nested: results may not be meaningful"
    ftest_nested(NestedModels{M}(models), df, dfr, dev, last(dev) / last(dfr))
end

anova(::Type{FTest}, aovm::NestedModels{M}) where {M <: FixedEffectModel} = 
    lrt_nested(aovm, dof_pred.(aovm.model), rss.(aovm.model), 1)

"""
    lfe(formula::FormulaTerm, df, vcov::CovarianceEstimator = Vcov.simple(); kwargs...)

A `GLM`-styled function to fit a fixed-effect model. 
"""
lfe(formula::FormulaTerm, df, vcov::CovarianceEstimator = Vcov.simple(); kwargs...) = 
    reg(df, formula, vcov; kwargs...)

# =================================================================================================================================
# Fit new models
"""
    anova_lfe(f::FormulaTerm, tbl, vcov::CovarianceEstimator = Vcov.simple(); 
            test::Type{<: GoodnessOfFit} = FTest, <keyword arguments>)
    anova_lfe(test::Type{<: GoodnessOfFit}, f::FormulaTerm, tbl, vcov::CovarianceEstimator = Vcov.simple(); <keyword arguments>)

ANOVA for fixed-effect linear regression.
* `vcov`: estimator of covariance matrix.
* `type`: type of anova (1 , 2 or 3). Default value is 1.

`anova_lfe` generates a `FixedEffectModel`.
"""
anova_lfe(f::FormulaTerm, tbl, vcov::CovarianceEstimator = Vcov.simple(); 
        test::Type{<: GoodnessOfFit} = FTest, 
        kwargs...)= 
    anova(test, FixedEffectModel, f, tbl, vcov; kwargs...)

anova_lfe(test::Type{<: GoodnessOfFit}, f::FormulaTerm, tbl, vcov::CovarianceEstimator = Vcov.simple(); kwargs...) = 
    anova(test, FixedEffectModel, f, tbl, vcov; kwargs...)

function anova(test::Type{<: GoodnessOfFit}, ::Type{FixedEffectModel}, f::FormulaTerm, tbl, vcov::CovarianceEstimator = Vcov.simple(); 
        type::Int = 1, 
        kwargs...)
    model = lfe(f, tbl, vcov; kwargs...)
    anova(test, model; type)
end


