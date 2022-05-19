#=
Author: 
    Liam Kruse
Email: 
    lkruse@stanford.edu
plotting.jl:
    Main plotting script for experiments
=#

##
#*******************************************************************************
# PACKAGES
#*******************************************************************************
using CSV
using DataFrames
using Distributions
using Interpolations
using LinearAlgebra
using PGFPlots

##
#*******************************************************************************
# DEFINITIONS
#*******************************************************************************
define_color("vandeusen", [73,92,111])
define_color("vandeusen_light", [128,141,154])
define_color("sky", [160,201,242])

define_color("cordovan", [152,68,71])
define_color("cordovan_light", [183,124,126])
define_color("salmon", [242,109,113])

define_color("middle_green", [91,140,90])
define_color("honey", [232, 174, 104])

define_color("fire_opal", [223,96,71])
define_color("carolina_blue", [85,159,195])
define_color("moss_green", [150,160,83])
define_color("russian_violet", [66,20,71])

##
#*******************************************************************************
# Functions
#*******************************************************************************
function interp_data(file, arr)
    df = CSV.read(file, DataFrame)
    xs = Vector(df[!,"Step"])
    A = Vector(df[!,"Value"])
    interp_linear = LinearInterpolation(xs, A)
    interp_arr = [interp_linear(x) for x in arr]
    return interp_arr
end

# tensorboard --logdir ./hopper
##
################################################################################
# HOPPER
################################################################################

#*******************************************************************************
# Rewards
#*******************************************************************************
# // TODO: Make this more flexible
domain = 200:100:90000
files = readdir("hopper_rew/")
aggregate = []
for f in files
    interp_arr = interp_data("hopper_rew/"*f, domain)
    push!(aggregate, interp_arr)
end

##
μ_arr = []; μ_plus_σ = []; μ_minus_σ = []; interaction = []
for i = 1:length(domain)
    step = [x[i] for x in aggregate]
    μ = mean(step); σ = std(step)
    push!(interaction, domain[i])
    push!(μ_arr, μ)
    push!(μ_plus_σ, μ + σ)
    push!(μ_minus_σ, μ - σ)
end

df = DataFrame(interaction = interaction, mean = μ_arr, upper = μ_plus_σ, lower = μ_minus_σ)
CSV.write("savedata/hopper_rew.csv", df)

##
pushPGFPlotsPreamble("\\usepgfplotslibrary{fillbetween}")
p = Axis([
    Plots.Command("\\addplot [name path = A, vandeusen, opacity = 0.0, forget plot] table [x = {interaction}, y = {lower}, col sep=comma] {savedata/hopper_rew.csv}"),
    Plots.Command("\\addplot [name path = B, vandeusen, opacity = 0.0, forget plot] table [x = {interaction}, y = {upper}, col sep=comma] {savedata/hopper_rew.csv}"),
    Plots.Command("\\addplot [vandeusen, very thick] table [x = {interaction}, y = {mean}, col sep=comma] {savedata/hopper_rew.csv}; \\addlegendentry{Mean reward}"),
    Plots.Command("\\addplot [vandeusen_light,opacity = 0.5] fill between [of = A and B]; \\addlegendentry{\$2\\sigma\$ error bounds}"),
#    Plots.Command("\\addplot [mark=none, cordovan, very thick, domain=0:8.2] {1.0}; \\addlegendentry{\$c\$}"),
],
style="enlarge x limits=false,grid=both",legendPos = "north west", legendStyle = "{nodes={scale=0.75}}",
#ylabel=L"$\hat{c}$", xlabel="Interaction",
ylabel="Mean Reward", xlabel="Interaction",
title="Average Reward per Episode (Hopper)")
save("hopper_rew.pdf", p)

# tensorboard --logdir ./walker2d
##
################################################################################
# WALKER2D
################################################################################

#*******************************************************************************
# Rewards
#*******************************************************************************
# // TODO: Make this more flexible
domain = 200:100:90000
files = readdir("walker2d_rew/")
aggregate = []
for f in files
    interp_arr = interp_data("walker2d_rew/"*f, domain)
    push!(aggregate, interp_arr)
end

##
μ_arr = []; μ_plus_σ = []; μ_minus_σ = []; interaction = []
for i = 1:length(domain)
    step = [x[i] for x in aggregate]
    μ = mean(step); σ = std(step)
    push!(interaction, domain[i])
    push!(μ_arr, μ)
    push!(μ_plus_σ, μ + σ)
    push!(μ_minus_σ, μ - σ)
end

df = DataFrame(interaction = interaction, mean = μ_arr, upper = μ_plus_σ, lower = μ_minus_σ)
CSV.write("savedata/walker2d_rew.csv", df)

##
pushPGFPlotsPreamble("\\usepgfplotslibrary{fillbetween}")
p = Axis([
    Plots.Command("\\addplot [name path = A, cordovan, opacity = 0.0, forget plot] table [x = {interaction}, y = {lower}, col sep=comma] {savedata/walker2d_rew.csv}"),
    Plots.Command("\\addplot [name path = B, cordovan, opacity = 0.0, forget plot] table [x = {interaction}, y = {upper}, col sep=comma] {savedata/walker2d_rew.csv}"),
    Plots.Command("\\addplot [cordovan, very thick] table [x = {interaction}, y = {mean}, col sep=comma] {savedata/walker2d_rew.csv}; \\addlegendentry{Mean reward}"),
    Plots.Command("\\addplot [cordovan_light,opacity = 0.5] fill between [of = A and B]; \\addlegendentry{\$2\\sigma\$ error bounds}"),
#    Plots.Command("\\addplot [mark=none, cordovan, very thick, domain=0:8.2] {1.0}; \\addlegendentry{\$c\$}"),
],
style="enlarge x limits=false,grid=both",legendPos = "north west", legendStyle = "{nodes={scale=0.75}}",
#ylabel=L"$\hat{c}$", xlabel="Interaction",
ylabel="Mean Reward", xlabel="Interaction",
title="Average Reward per Episode (Walker2D)")
save("walker2d_rew.pdf", p)


##
# TOGETHER
pushPGFPlotsPreamble("\\usepgfplotslibrary{fillbetween}")
p = Axis([
    Plots.Command("\\addplot [name path = A, cordovan, opacity = 0.0, forget plot] table [x = {interaction}, y = {lower}, col sep=comma] {savedata/walker2d_rew.csv}"),
    Plots.Command("\\addplot [name path = B, cordovan, opacity = 0.0, forget plot] table [x = {interaction}, y = {upper}, col sep=comma] {savedata/walker2d_rew.csv}"),
    Plots.Command("\\addplot [cordovan, very thick] table [x = {interaction}, y = {mean}, col sep=comma] {savedata/walker2d_rew.csv}; \\addlegendentry{Walker2D mean reward}"),
    Plots.Command("\\addplot [cordovan_light,opacity = 0.5] fill between [of = A and B]; \\addlegendentry{Walker2D \$2\\sigma\$ error bounds}"),
    Plots.Command("\\addplot [name path = A, vandeusen, opacity = 0.0, forget plot] table [x = {interaction}, y = {lower}, col sep=comma] {savedata/hopper_rew.csv}"),
    Plots.Command("\\addplot [name path = B, vandeusen, opacity = 0.0, forget plot] table [x = {interaction}, y = {upper}, col sep=comma] {savedata/hopper_rew.csv}"),
    Plots.Command("\\addplot [vandeusen, very thick] table [x = {interaction}, y = {mean}, col sep=comma] {savedata/hopper_rew.csv}; \\addlegendentry{Hopper mean reward}"),
    Plots.Command("\\addplot [vandeusen_light,opacity = 0.5] fill between [of = A and B]; \\addlegendentry{Hopper \$2\\sigma\$ error bounds}"),
],
style="enlarge x limits=false,grid=both",legendPos = "north west", legendStyle = "{nodes={scale=0.65}}",
#ylabel=L"$\hat{c}$", xlabel="Interaction",
ylabel="Mean Reward", xlabel="Interaction",
title="Average Reward per Episode")
save("together.pdf", p)