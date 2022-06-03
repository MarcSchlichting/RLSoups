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
define_color("cordovan", [152,68,71])

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