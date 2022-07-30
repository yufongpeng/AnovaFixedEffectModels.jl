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
    4 => x->categorical(x, levels = ["1 hour", "2 hours", "3 hours"], ordered = true),
    renamecols = false)
transform!(gpa, [1, 2, 5, 7] .=> categorical, renamecols = false)

# custimized approx
isapprox(x::NTuple{N, Float64}, y::NTuple{N, Float64}, atol::NTuple{N, Float64} = x ./ 1000) where N = 
    all(map((a, b, c)->isapprox(a, b, atol = c > eps(Float64) ? c : eps(Float64)), x, y, atol))

@testset "AnovaFixedEffectModels.jl" begin
    @testset "FixedEffectModel" begin
        @testset "One high dimensional fe on intercept" begin
            fem1 = lfe(@formula(gpa ~ fe(student) + occasion + job), gpa)
            lm1 = lm(@formula(gpa ~ student + occasion + job), gpa)
            global aovf = AFE.anova(fem1)
            global aovl = AnovaGLM.anova(lm1)
            @test !(@test_error test_show(aovf))
            @test nobs(aovf) == nobs(aovl)
            @test dof(aovf) == dof(aovl)[3:end]
            @test isapprox(deviance(aovf), deviance(aovl)[3:end])
            @test isapprox(pval(aovf)[1:end - 1], pval(aovl)[3:end - 1])
        end
        @testset "High dimensional fe on slope and intercept" begin
            fem0 = lfe(@formula(gpa ~ fe(student) &  occasion), gpa)
            lm0 = lm(@formula(gpa ~ student &  occasion), gpa)
            fem1 = lfe(@formula(gpa ~ fe(student) &  occasion + fe(student) + job), gpa)
            lm1 = lm(@formula(gpa ~ student &  occasion + student + job), gpa)
            fem2 = lfe(@formula(gpa ~ fe(student) &  occasion + 0 + job), gpa)
            lm2 = lm(@formula(gpa ~ student &  occasion + 0 + job), gpa)
            global aovf1 = AFE.anova(fem1)
            global aovl1 = AnovaGLM.anova(lm1)
            global aovf2 = AFE.anova(fem2, type = 3)
            global aovl2 = AnovaGLM.anova(lm2, type = 3)
            global aovfs = AFE.anova(fem0, fem2)
            global aovls = AnovaGLM.anova(lm0, lm2)
            @test !(@test_error test_show(aovf1))
            @test !(@test_error test_show(aovf2))
            @test !(@test_error test_show(aovfs))
            @test nobs(aovf1) == nobs(aovl1)
            @test last(dof(aovf1)) == last(dof(aovl1))
            @test isapprox(last(deviance(aovf1)), last(deviance(aovl1)))
            @test isapprox(first(teststat(aovf2)), first(teststat(aovl2)))
            @test isapprox(last(teststat(aovfs)), last(teststat(aovls)))
        end
    end
end
