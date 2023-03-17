using AnovaFixedEffectModels, CSV, DataFrames, CategoricalArrays, GLM, AnovaGLM
const AFE = AnovaFixedEffectModels
using Test
import Base.isapprox

test_show(x) = show(IOBuffer(), x)
macro test_error(x)
    return quote
        try 
            $x
            false
        catch e
            @error e
            true
        end
    end
end

const anova_datadir = joinpath(dirname(@__FILE__), "..", "data")

"Examples from https://m-clark.github.io/mixed-models-with-R/"
gpa = CSV.read(joinpath(anova_datadir, "gpa.csv"), DataFrame)
transform!(gpa,                                                                                                                   
           7 => x->replace(x, "yes" => true, "no" => false, "NA" => missing),                                                            
           5 => x->replace(x, "male" => 1, "female" => 0),                                                            
           4 => x->replace(x, "1 hour" => 1, "2 hours" => 2, "3 hours" => 3),                                                            
           renamecols = false)
transform!(gpa, [1, 2, 5, 7] .=> categorical, renamecols = false)
# custimized approx
isapprox(x::NTuple{N, Float64}, y::NTuple{N, Float64}, atol::NTuple{N, Float64} = x ./ 1000) where N = 
    all(map((a, b, c)->isapprox(a, b, atol = c > eps(Float64) ? c : eps(Float64)), x, y, atol))

@testset "AnovaFixedEffectModels.jl" begin
    @testset "FixedEffectModel" begin
        @testset "One high dimensional fe on intercept" begin
            global aovf = anova_lfe(@formula(gpa ~ fe(student) + occasion + job), gpa)
            global aovl = anova_lm(@formula(gpa ~ student + occasion + job), gpa)
            @test !(@test_error test_show(aovf))
            @test nobs(aovf) == nobs(aovl)
            @test dof(aovf) == dof(aovl)[3:end]
            @test isapprox(deviance(aovf), deviance(aovl)[3:end])
            @test isapprox(pval(aovf)[1:end - 1], pval(aovl)[3:end - 1])
        end
        @testset "High dimensional fe on slope and intercept" begin
            fem0 = lfe(@formula(gpa ~ fe(student) & occasion), gpa)
            lm0 = lm(@formula(gpa ~ student & occasion), gpa)
            fem1 = lfe(@formula(gpa ~ fe(student) & occasion + job), gpa)
            lm1 = lm(@formula(gpa ~ student & occasion + job), gpa)
            fem2 = lfe(@formula(gpa ~ fe(student) & occasion + 0 + job), gpa)
            lm2 = lm(@formula(gpa ~ student & occasion + 0 + job), gpa)
            global aovf1 = AFE.anova(FullModel(fem1, 1, true, true))
            global aovl1 = AnovaGLM.anova(lm1)
            global aovf2 = AFE.anova(fem2, type = 3)
            global aovl2 = AnovaGLM.anova(lm2, type = 3)
            global aovfs = AFE.anova(NestedModels{FixedEffectModel}(fem0, fem1))
            global aovfs2 = AFE.anova(fem0, fem1)
            global aovls = AnovaGLM.anova(lm0, lm1)
            @test !(@test_error test_show(aovf1))
            @test !(@test_error test_show(aovf2))
            @test !(@test_error test_show(aovfs))
            @test nobs(aovf1) == nobs(aovl1)
            @test last(dof(aovf1)) == dof(aovl1)[end - 1]
            @test isapprox(first(deviance(aovf2)), first(deviance(aovl2)))
            @test isapprox(first(teststat(aovf2)), first(teststat(aovl2)))
            @test isapprox(teststat(aovfs)[2], teststat(aovfs2)[2])
        end
    end
end
