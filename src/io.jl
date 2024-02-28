# ======================================================================================================
# IO
# anovatable api
function anovatable(aov::AnovaResult{<: FullModel{<: T}, FTest}; rownames = push!(prednames(aov), "(Residuals)")) where {T <: FixedEffectModel}
    dfr = round(Int, dof_residual(aov.anovamodel.model))
    σ² = rss(aov.anovamodel.model) / dfr
    AnovaTable([
                [dof(aov)..., dfr], 
                [deviance(aov)..., dfr * σ²], 
                [(deviance(aov) ./ dof(aov))..., σ²], 
                [teststat(aov)..., NaN], 
                [pval(aov)..., NaN]
                ],
              ["DOF", "Exp.SS", "Mean Square", "F value","Pr(>|F|)"],
              rownames, 5, 4)
end 

function anovatable(aov::AnovaResult{<: NestedModels{<: FixedEffectModel, N}, FTest}; rownames = string.(1:N)) where N

    rs = r2.(aov.anovamodel.model)
    rws = ntuple(length(aov.anovamodel.model)) do i 
        aov.anovamodel.model[i].r2_within
    end
    Δrs = _diff(rs)
    Δrws = _diff(rws)
    AnovaTable([
                    dof(aov), 
                    [NaN, _diff(dof(aov))...], 
                    repeat([round(Int, dof_residual(last(aov.anovamodel.model)))], N), 
                    rs,
                    [NaN, Δrs...],
                    rws,
                    [NaN, Δrws...],
                    deviance(aov),                     
                    [NaN, _diffn(deviance(aov))...], 
                    teststat(aov), 
                    pval(aov)
                ],
                ["DOF", "ΔDOF", "Res.DOF", "R²", "ΔR²", "R²_within", "ΔR²_within", "Res.SS", "Exp.SS", "F value", "Pr(>|F|)"],
                rownames, 11, 10)
end 

function show(io::IO, anovamodel::FullModel{<: T}) where {T <: FixedEffectModel}
    println(io, "FullModel for type $(anovamodel.type) test")
    println(io)
    println(io, "Predictors:")
    println(io, join(prednames(anovamodel), ", "))
    println(io)
    println(io, "Formula:")
    println(io, anovamodel.model.formula)
    println(io)
    println(io, "Coefficients:")
    show(io, coeftable(anovamodel.model))
end

function show(io::IO, anovamodel::NestedModels{M, N}) where {M <: FixedEffectModel, N}
    println(io, "NestedModels with $N models")
    println(io)
    println(io, "Formulas:")
    for(id, m) in enumerate(anovamodel.model)
        println(io, "Model $id: ", m.formula)
    end
    println(io)
    println(io, "Coefficients:")
    show(io, coeftable(first(anovamodel.model)))
    println(io)
    N > 2 && print(io, " .\n" ^ 3)
    show(io, coeftable(last(anovamodel.model)))
end

# Show function that delegates to anovatable
function show(io::IO, aov::AnovaResult{<: FullModel{<: FixedEffectModel}, T}) where {T <: GoodnessOfFit}
    at = anovatable(aov)
    println(io, "Analysis of Variance")
    println(io)
    println(io, "Type $(anova_type(aov)) test / $(testname(T))")
    println(io)
    println(io, aov.anovamodel.model.formula)
    println(io)
    println(io, "Table:")
    show(io, at)
end

function show(io::IO, aov::AnovaResult{<: MultiAovModels{<: FixedEffectModel}, T}) where {T <: GoodnessOfFit}
    at = anovatable(aov)
    println(io,"Analysis of Variance")
    println(io)
    println(io, "Type $(anova_type(aov)) test / $(testname(T))")
    println(io)
    for(id, m) in enumerate(aov.anovamodel.model)
        println(io, "Model $id: ", m.formula)
    end
    println(io)
    println(io, "Table:")
    show(io, at)
end