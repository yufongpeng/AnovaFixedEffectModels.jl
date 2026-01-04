# ======================================================================================================
# IO
# anovatable api
function anovatable(aov::AnovaResult{<: FullModel{<: T}, FTest}; rownames = push!(prednames(aov), "(Residuals)")) where {T <: FixedEffectModel}
    dfr = round(Int, dof_aovres(aov.anovamodel.model))
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
                    repeat([round(Int, dof_aovres(last(aov.anovamodel.model)))], N), 
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