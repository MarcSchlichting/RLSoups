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
using Colors
using ColorSchemes
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
define_color("middle_light", [140,175,140])

define_color("star_command", [34,116,165])
define_color("fire_opal", [223,96,71])

N_ROLLOUTS = 10;

blues       = range(colorant"black", stop=colorant"rgb(34,116,165)", length=N_ROLLOUTS)
oranges        = range(colorant"black", stop=colorant"rgb(223,96,71)", length=N_ROLLOUTS)

#blues       = range(colorant"black", stop=colorant"blue", length=N_ROLLOUTS)
#reds        = range(colorant"black", stop=colorant"red", length=N_ROLLOUTS)

hex_blue    = [hex(color) for color in blues]
hex_orange     = [hex(color) for color in oranges]

##
#*******************************************************************************
# FUNCTIONS
#*******************************************************************************
# Function to convert a hex value to an HTML color usable by PGFPlots
function html(hex)
	hex = replace(hex, "#"=>"")
	r,g,b = hex[1:2], hex[3:4], hex[5:6]
	return "{rgb,\"100:red, \"$r; green, \"$g; blue, \"$b}"
end

# Function to plot the time series "tree"
function plot_tree(N_UPDATES, title, df, hex_vals)
    rollouts = 1:N_ROLLOUTS*N_UPDATES
    # Define the axis
    p = Axis(style="enlarge x limits=false, grid=both, no marks", 
        xlabel="step", ylabel="value",title=title)
    sorted = []
    for i in rollouts    
        if i%N_ROLLOUTS == 1
            indices = i:(i+N_ROLLOUTS-1)
            final_values = [df[df.run .== j, :][end, "value"] for j in indices] 
            global sorted = sortperm(final_values)  
        end
        color_ind = findall(x->x == ((i-1)%(N_ROLLOUTS)+1), sorted)
        index = df.run .== i
        steps   = df[index, :][!, "step"]
        values  = df[index, :][!, "value"]
        push!(p, Plots.Linear(steps, values, style = "color=$(html(hex_vals[color_ind[1]])), solid, no marks, forget plot"))
    end
    return p
end

# Function to compute the mean and 4σ bounds for a tree
function compute_mean(N_UPDATES, df)
    rollouts = 1:N_ROLLOUTS*N_UPDATES
    arr = []; add_values = []
    for i in rollouts    
    
        if i%N_ROLLOUTS == 1 && N_UPDATES==5
            arr = (0:100:19900) .+ floor(i/10)*20000
        else
            arr = (0:100:9900) .+ floor(i/10)*10000
        end
        steps = df[df.run .== i, :][:, "step"]
        values = df[df.run .== i, :][:, "value"]
        interp_linear = LinearInterpolation(steps, values, extrapolation_bc = Line()) 
        interp_arr = [interp_linear(x) for x in arr]
        push!(add_values, interp_arr)
    end

    var_arr = []; mean_arr = []; μ_plus_σ = []; μ_minus_σ = [];
    for i = 1:N_ROLLOUTS:(N_ROLLOUTS*N_UPDATES)
        myrange = i:(i+N_ROLLOUTS-1)
        for j = 1:length(add_values[1])
            my_arr = [x[j] for x in add_values[myrange]]
            μ = mean(my_arr); σ = std(my_arr)
            push!(mean_arr, μ); push!(var_arr, σ)
            push!(μ_plus_σ, μ + 2*σ)
            push!(μ_minus_σ, μ - 2*σ)
        end
    end
    return mean_arr, μ_plus_σ, μ_minus_σ
    #return add_values
end


function interp_data(file, arr)
    df = CSV.read(file, DataFrame)
    steps = Vector(df[!,"step"])
    values = Vector(df[!,"value"])
    interp_linear = LinearInterpolation(steps, values, extrapolation_bc = Line()) 
    interp_arr = [interp_linear(x) for x in arr]
    return interp_arr
end

#*******************************************************************************
# PLOTS
#*******************************************************************************

##
################################################################################
# HOPPER WEIGHT SHARING
################################################################################
# Update every 100000 interactions
df = CSV.read("hopper100000/hopper100000.csv", DataFrame)
N_UPDATES = 1;
p = plot_tree(N_UPDATES, "Hopper 100000", df, hex_blue)
save("treeimages/hopper100000.pdf", p)
save("treeimages/hopper100000.tex", p,include_preamble=false)

# Update every 50000 interactions
df = CSV.read("hopper50000/hopper50000.csv", DataFrame)
N_UPDATES = 2;
p = plot_tree(N_UPDATES, "Hopper 50000", df, hex_blue)
save("treeimages/hopper50000.pdf", p)
save("treeimages/hopper50000.tex", p,include_preamble=false)

# Update every 20000 interactions
df = CSV.read("hopper20000/hopper20000.csv", DataFrame)
N_UPDATES = 5;
p = plot_tree(N_UPDATES, "Hopper 20000", df, hex_blue)
save("treeimages/hopper20000.pdf", p)
save("treeimages/hopper20000.tex", p,include_preamble=false)

# Update every 10000 interactions
df = CSV.read("hopper10000/hopper10000.csv", DataFrame)
N_UPDATES = 10;
p = plot_tree(N_UPDATES, "Hopper 10000", df, hex_blue)
save("treeimages/hopper10000.pdf", p)
save("treeimages/hopper10000.tex", p,include_preamble=false)

##
################################################################################
# WALKER WEIGHT SHARING
################################################################################
# Update every 100000 interactions
df = CSV.read("walker100000/walker100000.csv", DataFrame)
N_UPDATES = 1;
p = plot_tree(N_UPDATES, "Walker 100000", df, hex_orange)
save("treeimages/walker100000.pdf", p)
save("treeimages/walker100000.tex", p,include_preamble=false)

# Update every 50000 interactions
df = CSV.read("walker50000/walker50000.csv", DataFrame)
N_UPDATES = 2;
p = plot_tree(N_UPDATES, "Walker 50000", df, hex_orange)
save("treeimages/walker50000.pdf", p)
save("treeimages/walker50000.tex", p,include_preamble=false)

# Update every 20000 interactions
df = CSV.read("walker20000/walker20000.csv", DataFrame)
N_UPDATES = 5;
p = plot_tree(N_UPDATES, "Walker 20000", df, hex_orange)
save("treeimages/walker20000.pdf", p)
save("treeimages/walker20000.tex", p,include_preamble=false)

# Update every 10000 interactions
df = CSV.read("walker10000/walker10000.csv", DataFrame)
N_UPDATES = 10;
p = plot_tree(N_UPDATES, "Walker 10000", df, hex_orange)
save("treeimages/walker10000.pdf", p)
save("treeimages/walker10000.tex", p,include_preamble=false)

##
################################################################################
# HOPPER MEAN AND MEDIAN
################################################################################
# Hopper mean
df = CSV.read("hopper_mean/hopper_mean.csv", DataFrame)
N_UPDATES = 10;
p = plot_tree(N_UPDATES, "Hopper Mean", df, hex_blue)
save("treeimages/hopper_mean.pdf", p)
save("treeimages/hopper_mean.tex", p,include_preamble=false)

# Hopper median
df = CSV.read("hopper_median/hopper_median.csv", DataFrame)
N_UPDATES = 10;
p = plot_tree(N_UPDATES, "Hopper Median", df, hex_blue)
save("treeimages/hopper_median.pdf", p)
save("treeimages/hopper_median.tex", p,include_preamble=false)

##
################################################################################
# WALKER MEAN AND MEDIAN
################################################################################
# Walker mean
df = CSV.read("walker_mean/walker_mean.csv", DataFrame)
N_UPDATES = 10;
p = plot_tree(N_UPDATES, "Walker Mean", df, hex_orange)
save("treeimages/walker_mean.pdf", p)
save("treeimages/walker_mean.tex", p,include_preamble=false)

# Walker median
df = CSV.read("walker_median/walker_median.csv", DataFrame)
N_UPDATES = 10;
p = plot_tree(N_UPDATES, "Walker Median", df, hex_orange)
save("treeimages/walker_median.pdf", p)
save("treeimages/walker_median.tex", p,include_preamble=false)

##
################################################################################
# WALKER PPO
################################################################################
df = CSV.read("walker_ppo/walker_ppo.csv", DataFrame)
N_UPDATES = 5;
p = plot_tree(N_UPDATES, "Walker PPO", df, hex_orange)
save("treeimages/walker_ppo_tree.pdf", p)
save("treeimages/walker_ppo_tree.tex", p, include_preamble=false)

##
################################################################################
# HOPPER SOFTMAX, MEAN, MEDIAN
################################################################################
df = CSV.read("hopper10000/hopper10000.csv", DataFrame)
N_UPDATES = 10;
mean_arr, μ_plus_σ, μ_minus_σ = compute_mean(N_UPDATES, df)
interaction = 1:100:100000
save_df = DataFrame(interaction = interaction, mean = mean_arr, upper = μ_plus_σ, lower = μ_minus_σ)
CSV.write("savedata/hopper10000.csv", save_df)

df = CSV.read("hopper_mean/hopper_mean.csv", DataFrame)
N_UPDATES = 10;
mean_arr, μ_plus_σ, μ_minus_σ = compute_mean(N_UPDATES, df)
interaction = 1:100:100000
save_df = DataFrame(interaction = interaction, mean = mean_arr, upper = μ_plus_σ, lower = μ_minus_σ)
CSV.write("savedata/hopper_mean.csv", save_df)

df = CSV.read("hopper_median/hopper_median.csv", DataFrame)
N_UPDATES = 10;
mean_arr, μ_plus_σ, μ_minus_σ = compute_mean(N_UPDATES, df)
interaction = 1:100:100000
save_df = DataFrame(interaction = interaction, mean = mean_arr, upper = μ_plus_σ, lower = μ_minus_σ)
CSV.write("savedata/hopper_median.csv", save_df)

##
pushPGFPlotsPreamble("\\usepgfplotslibrary{fillbetween}")
p = Axis([
    Plots.Command("\\addplot [name path = A, vandeusen, opacity = 0.0, forget plot] table [x = {interaction}, y = {lower}, col sep=comma] {savedata/hopper10000.csv}"),
    Plots.Command("\\addplot [name path = B, vandeusen, opacity = 0.0, forget plot] table [x = {interaction}, y = {upper}, col sep=comma] {savedata/hopper10000.csv}"),
    Plots.Command("\\addplot [vandeusen, thick] table [x = {interaction}, y = {mean}, col sep=comma] {savedata/hopper10000.csv}; \\addlegendentry{Softmax averaging}"),
    Plots.Command("\\addplot [vandeusen_light,opacity = 0.5] fill between [of = A and B]; \\addlegendentry{Softmax \$4\\sigma\$ bounds}"),

    Plots.Command("\\addplot [name path = A, cordovan, opacity = 0.0, forget plot] table [x = {interaction}, y = {lower}, col sep=comma] {savedata/hopper_mean.csv}"),
    Plots.Command("\\addplot [name path = B, cordovan, opacity = 0.0, forget plot] table [x = {interaction}, y = {upper}, col sep=comma] {savedata/hopper_mean.csv}"),
    Plots.Command("\\addplot [cordovan, thick] table [x = {interaction}, y = {mean}, col sep=comma] {savedata/hopper_mean.csv}; \\addlegendentry{Mean averaging}"),
    Plots.Command("\\addplot [cordovan_light,opacity = 0.5] fill between [of = A and B]; \\addlegendentry{Mean \$4\\sigma\$ bounds}"),

    Plots.Command("\\addplot [name path = A, middle_green, opacity = 0.0, forget plot] table [x = {interaction}, y = {lower}, col sep=comma] {savedata/hopper_median.csv}"),
    Plots.Command("\\addplot [name path = B, middle_green, opacity = 0.0, forget plot] table [x = {interaction}, y = {upper}, col sep=comma] {savedata/hopper_median.csv}"),
    Plots.Command("\\addplot [middle_green, thick] table [x = {interaction}, y = {mean}, col sep=comma] {savedata/hopper_median.csv}; \\addlegendentry{Median averaging}"),
    Plots.Command("\\addplot [middle_light,opacity = 0.5] fill between [of = A and B]; \\addlegendentry{Median \$4\\sigma\$ bounds}"),
],
style="enlarge x limits=false,grid=both",legendPos = "north west", legendStyle = "{nodes={scale=0.75}}",
ylabel="value", xlabel="step",
title="Hopper Softmax, Mean, and Median")
save("hopper_averaging.pdf", p)
save("hopper_averaging.tex", p, include_preamble=false)

##
################################################################################
# WALKER SOFTMAX, MEAN, MEDIAN
################################################################################
df = CSV.read("walker10000/walker10000.csv", DataFrame)
N_UPDATES = 10;
mean_arr, μ_plus_σ, μ_minus_σ = compute_mean(N_UPDATES, df)
interaction = 1:100:100000
save_df = DataFrame(interaction = interaction, mean = mean_arr, upper = μ_plus_σ, lower = μ_minus_σ)
CSV.write("savedata/walker10000.csv", save_df)

df = CSV.read("walker_mean/walker_mean.csv", DataFrame)
N_UPDATES = 10;
mean_arr, μ_plus_σ, μ_minus_σ = compute_mean(N_UPDATES, df)
interaction = 1:100:100000
save_df = DataFrame(interaction = interaction, mean = mean_arr, upper = μ_plus_σ, lower = μ_minus_σ)
CSV.write("savedata/walker_mean.csv", save_df)

df = CSV.read("walker_median/walker_median.csv", DataFrame)
N_UPDATES = 10;
mean_arr, μ_plus_σ, μ_minus_σ = compute_mean(N_UPDATES, df)
interaction = 1:100:100000
save_df = DataFrame(interaction = interaction, mean = mean_arr, upper = μ_plus_σ, lower = μ_minus_σ)
CSV.write("savedata/walker_median.csv", save_df)

##
pushPGFPlotsPreamble("\\usepgfplotslibrary{fillbetween}")
p = Axis([
    Plots.Command("\\addplot [name path = A, vandeusen, opacity = 0.0, forget plot] table [x = {interaction}, y = {lower}, col sep=comma] {savedata/walker10000.csv}"),
    Plots.Command("\\addplot [name path = B, vandeusen, opacity = 0.0, forget plot] table [x = {interaction}, y = {upper}, col sep=comma] {savedata/walker10000.csv}"),
    Plots.Command("\\addplot [vandeusen, thick] table [x = {interaction}, y = {mean}, col sep=comma] {savedata/walker10000.csv}; \\addlegendentry{Softmax averaging}"),
    Plots.Command("\\addplot [vandeusen_light,opacity = 0.5] fill between [of = A and B]; \\addlegendentry{Softmax \$4\\sigma\$ bounds}"),

    Plots.Command("\\addplot [name path = A, cordovan, opacity = 0.0, forget plot] table [x = {interaction}, y = {lower}, col sep=comma] {savedata/walker_mean.csv}"),
    Plots.Command("\\addplot [name path = B, cordovan, opacity = 0.0, forget plot] table [x = {interaction}, y = {upper}, col sep=comma] {savedata/walker_mean.csv}"),
    Plots.Command("\\addplot [cordovan, thick] table [x = {interaction}, y = {mean}, col sep=comma] {savedata/walker_mean.csv}; \\addlegendentry{Mean averaging}"),
    Plots.Command("\\addplot [cordovan_light,opacity = 0.5] fill between [of = A and B]; \\addlegendentry{Mean \$4\\sigma\$ bounds}"),

    Plots.Command("\\addplot [name path = A, middle_green, opacity = 0.0, forget plot] table [x = {interaction}, y = {lower}, col sep=comma] {savedata/walker_median.csv}"),
    Plots.Command("\\addplot [name path = B, middle_green, opacity = 0.0, forget plot] table [x = {interaction}, y = {upper}, col sep=comma] {savedata/walker_median.csv}"),
    Plots.Command("\\addplot [middle_green, thick] table [x = {interaction}, y = {mean}, col sep=comma] {savedata/walker_median.csv}; \\addlegendentry{Median averaging}"),
    Plots.Command("\\addplot [middle_light,opacity = 0.5] fill between [of = A and B]; \\addlegendentry{Median \$4\\sigma\$ bounds}"),
],
style="enlarge x limits=false,grid=both",legendPos = "north west", legendStyle = "{nodes={scale=0.75}}",
ylabel="value", xlabel="step",
title="Walker Softmax, Mean, and Median")
save("walker_averaging.pdf", p)
save("walker_averaging.tex", p, include_preamble=false)

##
################################################################################
# HOPPER ALGOS
################################################################################
df = CSV.read("hopper20000/hopper20000.csv", DataFrame)
N_UPDATES = 5;
mean_arr, μ_plus_σ, μ_minus_σ = compute_mean(N_UPDATES, df)
interaction = 1:1000:100000
save_df = DataFrame(interaction = interaction, mean = mean_arr, upper = μ_plus_σ, lower = μ_minus_σ)
CSV.write("savedata/hopper20000.csv", save_df)

df = CSV.read("hopper_ppo/hopper_ppo.csv", DataFrame)
N_UPDATES = 5;
mean_arr, μ_plus_σ, μ_minus_σ = compute_mean(N_UPDATES, df)
interaction = 1:1000:100000
save_df = DataFrame(interaction = interaction, mean = mean_arr, upper = μ_plus_σ, lower = μ_minus_σ)
CSV.write("savedata/hopper_ppo.csv", save_df)

##
pushPGFPlotsPreamble("\\usepgfplotslibrary{fillbetween}")
p = Axis([
    Plots.Command("\\addplot [name path = A, vandeusen, opacity = 0.0, forget plot] table [x = {interaction}, y = {lower}, col sep=comma] {savedata/hopper20000.csv}"),
    Plots.Command("\\addplot [name path = B, vandeusen, opacity = 0.0, forget plot] table [x = {interaction}, y = {upper}, col sep=comma] {savedata/hopper20000.csv}"),
    Plots.Command("\\addplot [vandeusen, thick] table [x = {interaction}, y = {mean}, col sep=comma] {savedata/hopper20000.csv}; \\addlegendentry{TD3 mean}"),
    Plots.Command("\\addplot [vandeusen_light,opacity = 0.5] fill between [of = A and B]; \\addlegendentry{TD3 \$4\\sigma\$ bounds}"),

    Plots.Command("\\addplot [name path = A, cordovan, opacity = 0.0, forget plot] table [x = {interaction}, y = {lower}, col sep=comma] {savedata/hopper_ppo.csv}"),
    Plots.Command("\\addplot [name path = B, cordovan, opacity = 0.0, forget plot] table [x = {interaction}, y = {upper}, col sep=comma] {savedata/hopper_ppo.csv}"),
    Plots.Command("\\addplot [cordovan, thick] table [x = {interaction}, y = {mean}, col sep=comma] {savedata/hopper_ppo.csv}; \\addlegendentry{PPO mean}"),
    Plots.Command("\\addplot [cordovan_light,opacity = 0.5] fill between [of = A and B]; \\addlegendentry{PPO \$4\\sigma\$ bounds}"),
],
style="enlarge x limits=false,grid=both",legendPos = "north west", legendStyle = "{nodes={scale=0.75}}",
ylabel="value", xlabel="step",
title="PPO vs TD3: Hopper")
save("hopper_algos.pdf", p)
save("hopper_algos.tex", p, include_preamble=false)

##
################################################################################
# WALKER ALGOS
################################################################################
df = CSV.read("walker20000/walker20000.csv", DataFrame)
N_UPDATES = 5;
mean_arr, μ_plus_σ, μ_minus_σ = compute_mean(N_UPDATES, df)
interaction = 1:1000:100000
save_df = DataFrame(interaction = interaction, mean = mean_arr, upper = μ_plus_σ, lower = μ_minus_σ)
CSV.write("savedata/walker20000.csv", save_df)

df = CSV.read("hopper_ppo/hopper_ppo.csv", DataFrame)
N_UPDATES = 5;
mean_arr, μ_plus_σ, μ_minus_σ = compute_mean(N_UPDATES, df)
interaction = 1:1000:100000
save_df = DataFrame(interaction = interaction, mean = mean_arr, upper = μ_plus_σ, lower = μ_minus_σ)
CSV.write("savedata/hopper_ppo.csv", save_df)

pushPGFPlotsPreamble("\\usepgfplotslibrary{fillbetween}")
p = Axis([
    Plots.Command("\\addplot [name path = A, vandeusen, opacity = 0.0, forget plot] table [x = {interaction}, y = {lower}, col sep=comma] {savedata/walker20000.csv}"),
    Plots.Command("\\addplot [name path = B, vandeusen, opacity = 0.0, forget plot] table [x = {interaction}, y = {upper}, col sep=comma] {savedata/walker20000.csv}"),
    Plots.Command("\\addplot [vandeusen, thick] table [x = {interaction}, y = {mean}, col sep=comma] {savedata/walker20000.csv}; \\addlegendentry{TD3 mean}"),
    Plots.Command("\\addplot [vandeusen_light,opacity = 0.5] fill between [of = A and B]; \\addlegendentry{TD3 \$4\\sigma\$ bounds}"),

    Plots.Command("\\addplot [name path = A, cordovan, opacity = 0.0, forget plot] table [x = {interaction}, y = {lower}, col sep=comma] {savedata/walker_ppo.csv}"),
    Plots.Command("\\addplot [name path = B, cordovan, opacity = 0.0, forget plot] table [x = {interaction}, y = {upper}, col sep=comma] {savedata/walker_ppo.csv}"),
    Plots.Command("\\addplot [cordovan, thick] table [x = {interaction}, y = {mean}, col sep=comma] {savedata/walker_ppo.csv}; \\addlegendentry{PPO mean}"),
    Plots.Command("\\addplot [cordovan_light,opacity = 0.5] fill between [of = A and B]; \\addlegendentry{PPO \$4\\sigma\$ bounds}"),
],
style="enlarge x limits=false,grid=both",legendPos = "north west", legendStyle = "{nodes={scale=0.75}}",
ylabel="value", xlabel="step",
title="PPO vs TD3: Walker")
save("walker_algos.pdf", p)
save("walker_algos.tex", p, include_preamble=false)