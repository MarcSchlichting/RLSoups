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

N_ROLLOUTS = 10;

blues       = range(colorant"black", stop=colorant"blue", length=N_ROLLOUTS)
reds        = range(colorant"black", stop=colorant"red", length=N_ROLLOUTS)
hex_blue    = [hex(color) for color in blues]
hex_red     = [hex(color) for color in reds]

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

##
#*******************************************************************************
# PLOTS
#*******************************************************************************
# Update every 100000 interactions
df = CSV.read("hopper100000/hopper100000.csv", DataFrame)
N_UPDATES = 1;
p = plot_tree(N_UPDATES, "Hopper 100000", df, hex_red)
save("treeimages/hopper100000.pdf", p)


# Update every 50000 interactions
df = CSV.read("hopper50000/hopper50000.csv", DataFrame)
N_UPDATES = 2;
p = plot_tree(N_UPDATES, "Hopper 50000", df, hex_red)
save("treeimages/hopper50000.pdf", p)


# Update every 20000 interactions
df = CSV.read("hopper20000/hopper20000.csv", DataFrame)
N_UPDATES = 5;
p = plot_tree(N_UPDATES, "Hopper 20000", df, hex_red)
save("treeimages/hopper20000.pdf", p)


# Update every 10000 interactions
df = CSV.read("hopper10000/hopper10000.csv", DataFrame)
N_UPDATES = 10;
p = plot_tree(N_UPDATES, "Hopper 10000", df, hex_red)
save("treeimages/hopper10000.pdf", p)

# Update every 100000 interactions
df = CSV.read("walker100000/walker100000.csv", DataFrame)
N_UPDATES = 1;
p = plot_tree(N_UPDATES, "Walker 100000", df, hex_red)
save("treeimages/walker100000.pdf", p)


# Update every 50000 interactions
df = CSV.read("walker50000/walker50000.csv", DataFrame)
N_UPDATES = 2;
p = plot_tree(N_UPDATES, "Walker 50000", df, hex_red)
save("treeimages/walker50000.pdf", p)


# Update every 20000 interactions
df = CSV.read("walker20000/walker20000.csv", DataFrame)
N_UPDATES = 5;
p = plot_tree(N_UPDATES, "Walker 20000", df, hex_red)
save("treeimages/walker20000.pdf", p)


# Update every 10000 interactions
df = CSV.read("walker10000/walker10000.csv", DataFrame)
N_UPDATES = 10;
p = plot_tree(N_UPDATES, "Walker 10000", df, hex_red)
save("treeimages/walker10000.pdf", p)