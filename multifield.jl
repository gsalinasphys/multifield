### A Pluto.jl notebook ###
# v0.18.1

using Markdown
using InteractiveUtils

# ╔═╡ 50172480-9ef3-11ec-21a7-f74589ea5026
begin
	using Pkg
	Pkg.activate("Project.toml")
end

# ╔═╡ 367bddc1-674b-4352-90e0-ffb4b0ef4545
begin
	using LinearAlgebra, DifferentialEquations, Plots, PlutoTest, PlutoUI, TensorOperations, LaTeXStrings
	using ForwardDiff: derivative, gradient, jacobian
end

# ╔═╡ Cell order:
# ╠═50172480-9ef3-11ec-21a7-f74589ea5026
# ╠═367bddc1-674b-4352-90e0-ffb4b0ef4545
