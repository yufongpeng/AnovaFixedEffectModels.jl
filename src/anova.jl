# ================================================================================================
# Main API
@doc """
    anova(<models>...; test::Type{<: GoodnessOfFit})
    anova(test::Type{<: GoodnessOfFit}, <models>...;  <keyword arguments>)

Analysis of variance.

Return `AnovaResult{M, test, N}`. See [`AnovaResult`](@ref) for details.

# Arguments
* `models`: model objects
    1. `TableRegressionModel{<: FixedEffectModel}` fitted by `AnovaFixedEffectModels.lfe`.
    If mutiple models are provided, they should be nested and the last one is the most complex.
* `test`: test statistics for goodness of fit. The default is based on the model type.
    1. `TableRegressionModel{<: FixedEffectModel}`: `FTest`.

## Other keyword arguments
* When one model is provided:  
    1. `type` specifies type of anova (1 or 3). Default value is 1.
* When multiple models are provided:  
    1. `check`: allows to check if models are nested. Defalut value is true. Some checkers are not implemented now.
    2. `isnested`: true when models are checked as nested (manually or automatically). Defalut value is false. 

# Algorithm

The variable `dev` is a vector that the ith element is the sum of [squared deviance residuals (unit deviance)](https://en.wikipedia.org/wiki/Deviance_(statistics)) of the ith model. 
It is equivalent to the residual sum.

The attribute `deviance` of `AnovaResult` is a vector `Δdev` where `Δdevᵢ = devᵢ₋₁ - devᵢ`.

F-statistic is then defined as `Δdev / (dispersion² × degree of freedom)`.

For type I and III ANOVA, F-statistic is computed directly by the variance-covariance matrix (`vcov`) of the most complex model; the deviance is calculated backward.
1. Type I:

    First, calculate `f` as the upper factor of Cholesky factorization of `vcov⁻¹ * β`.

    For a factor that starts from ith row/column of the model matrix with `n` degree of freedom, the f-statistic is `Σᵢⁱ⁺ⁿ⁻¹ fₖ² / n`.
2. Type III: 

    For a factor occupying ith to jth row/column of the model matrix with `n` degree of freedom, f-statistic is `β[i, ..., j]ᵀ * vcov[i, ..., j; i, ..., j]⁻¹ * β[i, ..., j] / n`.

!!! note
    For fitting new models and conducting anova at the same time, see [`anova_lfe`](@ref) for `FixedEffectModel`.
"""
anova(::Val{:AnovaFixedEffectModels})

anova(trms::Vararg{TableRegressionModel{<: FixedEffectModel}}; 
        test::Type{<: GoodnessOfFit} = FTest,
        kwargs...) = 
    anova(test, trms...; kwargs...)

# ================================================================================================
# ANOVA by F test

function anova(::Type{FTest}, 
            trm::TableRegressionModel{<: FixedEffectModel};
            type::Int = 1, kwargs...)

    type == 2           && throw(ArgumentError("Type 2 anova is not implemented"))
    type in [1, 2, 3]   || throw(ArgumentError("Invalid type"))
    assign = trm.mm.assign
    df = dof(assign)
    filter!(>(0), df)
    # May exist some floating point error from dof_residual
    push!(df, round(Int, dof_residual(trm)))
    df = tuple(df...)
    if type in [1, 3] 
        # vcov methods
        varβ = vcov(trm)
        β = trm.model.coef
        if type == 1
            fs = abs2.(cholesky(Hermitian(inv(varβ))).U * β) 
            offset = first(assign) - 1
            fstat = ntuple(last(assign) - offset) do fix
                sum(fs[findall(==(fix + offset), assign)]) / df[fix]
            end
        else
            # calculate block by block
            offset = first(assign) - 1
            fstat = ntuple(last(assign) - offset) do fix
                select = findall(==(fix + offset), assign)
                β[select]' * (varβ[select, select] \ β[select]) / df[fix]
            end
        end
        σ² = rss(trm.model) / last(df)
        devs = (fstat .* σ²..., σ²) .* df
    end
    pvalue = (ccdf.(FDist.(df[1:end - 1], last(df)), abs.(fstat))..., NaN)
    AnovaResult{FTest}(trm, type, df, devs, (fstat..., NaN), pvalue, NamedTuple())
end

# =================================================================================================================
# Nested models 

function anova(::Type{FTest}, 
                trms::Vararg{TableRegressionModel{<: FixedEffectModel}}; 
                check::Bool = true,
                isnested::Bool = false)

    df = dof.(trms)
    ord = sortperm(collect(df))
    df = df[ord]
    trms = trms[ord]
    # May exist some floating point error from dof_residual
    dfr = round.(Int, dof_residual.(trms))
    dev = ntuple(length(trms)) do i 
        trms[i].model.rss
    end

    # check comparable and nested
    check && @warn "Could not check whether models are nested: results may not be meaningful"
    ftest_nested(trms, df, dfr, dev, last(dev) / last(dfr))
end

"""
    lfe(formula::FormulaTerm, df, vcov::CovarianceEstimator = Vcov.simple(); kwargs...)

Fit a `FixedEffectModel` and wrap it into `TableRegressionModel`. 
!!! warn
    This function currently does not perform well. It re-compiles everytime; may be due to `@nonspecialize` for parameters of `reg`.
"""
lfe(formula::FormulaTerm, df, vcov::CovarianceEstimator = Vcov.simple(); kwargs...) = 
    to_trm(reg(df, formula, vcov; kwargs...), df)

"""
    to_trm(model, df)

Wrap fitted `FixedEffectModel` into `TableRegressionModel`.
"""
function to_trm(model::FixedEffectModel, df)
    f = model.formula
    has_fe_intercept = any(fe_intercept(f))
    rhs = vectorize(f.rhs)
    f = isa(first(rhs), ConstantTerm) ? f : FormulaTerm(f.lhs, (ConstantTerm(1), rhs...))
    s = schema(f, df, model.contrasts)
    f = apply_schema(f, s, FixedEffectModel, has_fe_intercept)
    mf = ModelFrame(f, s, columntable(df[!, getproperty.(keys(s), :sym)]), FixedEffectModel)
    # Fake modelmatrix
    assign = mapreduce(((i, t), ) -> i*ones(width_fe(t)),
                        append!,
                        enumerate(vectorize(f.rhs.terms)),
                        init=Int[])
    has_fe_intercept && popfirst!(assign)
    mm = ModelMatrix(ones(Float64, 1, 1), assign)
    TableRegressionModel(model, mf, mm)
end

# =================================================================================================================================
# Fit new models
"""
    anova_lfe(f::FormulaTerm, df, vcov::CovarianceEstimator = Vcov.simple(); 
            test::Type{<: GoodnessOfFit} = FTest, <keyword arguments>)
    anova_lfe(test::Type{<: GoodnessOfFit}, f::FormulaTerm, df, vcov::CovarianceEstimator = Vcov.simple(); <keyword arguments>)

ANOVA for fixed-effect linear regression.
* `vcov`: estimator of covariance matrix.
* `type`: type of anova (1 or 3). Default value is 1.

`anova_lfe` generate a `TableRegressionModel{<: FixedEffectModel}`.
"""
anova_lfe(f::FormulaTerm, df, vcov::CovarianceEstimator = Vcov.simple(); 
        test::Type{<: GoodnessOfFit} = FTest, 
        kwargs...)= 
    anova(test, FixedEffectModel, f, df, vcov; kwargs...)

anova_lfe(test::Type{<: GoodnessOfFit}, f::FormulaTerm, df, vcov::CovarianceEstimator = Vcov.simple(); kwargs...) = 
    anova(test, FixedEffectModel, f, df, vcov; kwargs...)

function anova(test::Type{<: GoodnessOfFit}, ::Type{FixedEffectModel}, f::FormulaTerm, df, vcov::CovarianceEstimator = Vcov.simple(); 
        type::Int = 1, 
        kwargs...)
    trm = to_trm(reg(df, f, vcov; kwargs...), df)
    anova(test, trm; type)
end


