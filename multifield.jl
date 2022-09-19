### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 50172480-9ef3-11ec-21a7-f74589ea5026
begin
	using Pkg
	Pkg.activate("Project.toml")
end

# ╔═╡ 367bddc1-674b-4352-90e0-ffb4b0ef4545
begin
	using LinearAlgebra, DifferentialEquations, Plots, PlutoTest, PlutoUI, TensorOperations, LaTeXStrings, Latexify, Parameters, Roots
	using ForwardDiff: derivative, gradient, jacobian
end

# ╔═╡ e2306bb5-b84f-482a-9b06-48af1c79cb89
cd("/home/gsalinas/GitHub/multifield")

# ╔═╡ b363d59b-efbb-4e35-bb26-ef884ceadfda
md"""
Add new package:
"""

# ╔═╡ c88612f4-0fa2-4d3e-8cd4-9714a50715c9
# begin
# 	import Pkg
# 	Pkg.add(["PlutoTest", "PlutoUI", "TensorOperations", "Parameters", "Roots"])
# end

# ╔═╡ 75081cf5-a327-4e4e-8569-8a96f5179479
md"""
Activate the environment:
"""

# ╔═╡ 277fb121-e8eb-48f5-b398-078aaa9ce3a6
md"""
Load packages:
"""

# ╔═╡ 500c52a4-318d-4300-9f23-ceaffa1b83e0
TableOfContents(aside=true)

# ╔═╡ bdb142de-8840-4fc2-9f11-fd4132df64f4
md"""
Some general purpose functions:
"""

# ╔═╡ a8b4458b-fe05-4868-8fd0-3dbb87d7a077
begin
	subscript(i) = join(Char(0x2080 + d) for d in reverse!(digits(i)))
	
	function rand_dir(dim::Int64)
		propose = 2*rand(dim) .- 1
		if sum(propose.^2) < 1
			return propose
		end
		rand_dir(dim)
	end
end;

# ╔═╡ 415291e0-77f8-4e85-89ae-2a0010441f4d
md"""
# Defining the potential
"""

# ╔═╡ 8a710b6a-c1d9-4293-929e-6e5e9cdf5bca
md"""
Some predefined potential types:
"""

# ╔═╡ e31e62df-e592-48c0-9810-b77d40d18a21
begin
	abstract type PotentialType end

#################### Generic potential ####################

	struct GenericPotential <: PotentialType
		params::NamedTuple
	end

	params(P::GenericPotential) = P.params

#################### Double quadratic ####################
	
	struct Quadratic <: PotentialType
		masses::Tuple
	end
	
	V(Q::Quadratic) = function(ϕ::Vector)
		@assert length(Q.masses) == length(ϕ) "Dimention of field vector ϕ does not match the number of mass parameters"
		
		0.5*sum(Q.masses.^2 .* ϕ.^2)
	end

	params(Q::Quadratic) = (; zip(Tuple(Symbol("m"*subscript(i)) for i in 1:length(Q.masses)), Tuple(Q.masses))...)
	
end;

# ╔═╡ 57599d34-6813-4a1d-8710-8af8fd00aa82
md"""
## Formula for a generic potential
"""

# ╔═╡ 46e7357a-7bd8-4c81-9299-1cd419e21da1
md"""
Type below the formula for the potential:
"""

# ╔═╡ 973052be-4ed9-4dfa-bacc-c8c798683b77
V(GP::GenericPotential) = function(ϕ::Vector)
	r, θ = ϕ
	N, gₛ, lₛ, q, a₀, a₁, b₁, p, V₀ = GP.params
	u, μ₅, L4T11 = 50lₛ, 1/((2pi)^5*lₛ^6), 27pi*N*gₛ*lₛ^4/4.
	T₅ = μ₅/gₛ
	γ =  4pi^2*lₛ^2*p*q*T₅*gₛ
	
	𝓗 = L4T11/(3u)^4 * (18u^2/r^2-2log(9u^2/r^2+1))
	𝓕 = 𝓗/9*(r^2+3u^2)^2 + (pi*lₛ^2*q)^2
	φ = 4pi*p*T₅/𝓗*(sqrt(𝓕)-lₛ^2*pi*q*gₛ)
	Φm = 5/72*(9*(r^2/u^2-2)*r^2/u^2 + 162log(r^2/u^2+9) - 9 - 160log(10))
	Φh = a₀*(18u^2/r^2-2log(9u^2/r^2+1)) + 2a₁*(6+9u^2/r^2-2(2+r^2/(3u^2))log(1+9u^2/r^2))cos(θ) + b₁/2*(2+r^2/(3u^2))cos(θ)

	V₀ + φ + γ*(Φm+Φh)
end;

# ╔═╡ 8ccd9f97-1248-4f69-a9a8-d1c18965e9b6
md"""
# Defining the field space metric
"""

# ╔═╡ fe6432a1-94cc-4aa5-821e-c5085e6d54d8
md"""
Some predefined field metric types:
"""

# ╔═╡ 64af0ee4-e86b-445b-aa64-0bb930e458b8
begin
	abstract type FieldSpaceMetricType end

#################### Generic metric ####################
	
	struct GenericMetric <: FieldSpaceMetricType
		params::NamedTuple
	end

#################### Canonical metric ####################
	
	struct Canonical <: FieldSpaceMetricType
	end

	G(C::Canonical) = function(ϕ)
	    Matrix{Float64}(I, length(ϕ), length(ϕ))
	end

#################### Hyperbolic metric in 2D ####################
	
	struct Hyperbolic2D <: FieldSpaceMetricType
	    L::Real
	end
	
	G(H::Hyperbolic2D) = function(ϕ)
		@assert length(ϕ) == 2 "Field has to be two-dimensional"

		ϕ₁, ϕ₂ = ϕ
	    [1 0; 0 H.L^2*sinh(ϕ₁/H.L)^2]
	end

	dot(FS::FieldSpaceMetricType, v₁::Vector, v₂::Vector) = function(ϕ::Vector) v₁'*G(FS)(ϕ)*v₂ end
	norm(FS::FieldSpaceMetricType, v::Vector) = function(ϕ::Vector) sqrt(dot(FS, v, v)(ϕ)) end
	
	perp_to(FS::FieldSpaceMetricType, v::Vector) = function(ϕ::Vector)
		randdir = rand_dir(length(v))
		vn = randdir - dot(FS, randdir, v)(ϕ)*v/(norm(FS, v)(ϕ)^2*norm(FS, randdir)(ϕ)^2)
		vn /= norm(FS, vn)(ϕ)
	end
end;

# ╔═╡ df156a1e-edf2-40d0-9ac0-d3ffb0c4e49a
md"""
And the associated Christoffel symbols:
"""

# ╔═╡ 05d66135-0b1a-455b-b2de-bd2aabfd1bd7
md"""
## Formula for a generic metric
"""

# ╔═╡ 42346539-58cf-45ea-916e-9821c7e80f6a
md"""
Type below the formula for the field space metric:
"""

# ╔═╡ d59fa4db-cd80-402e-a024-974a9ddf2e46
G(GM::GenericMetric) = function(ϕ::Vector)
	r, θ = ϕ
	N, gₛ, lₛ, q, p = GM.params
	u, μ₅, L4T11 = 50lₛ, 1/((2pi)^5*lₛ^6), 27pi*N*gₛ*lₛ^4/4.
	T₅ = μ₅/gₛ

	𝓗 = L4T11/(3u)^4 * (18u^2/r^2-2log(9u^2/r^2+1))
	𝓕 = 𝓗/9*(r^2+3u^2)^2 + (pi*lₛ^2*q)^2
	
	4pi*p*T₅*sqrt(𝓕)*[(r^2+6u^2)/(r^2+9u^2) 0; 0 (r^2+6u^2)/6]
end;

# ╔═╡ 73e7ba6e-7a4e-4051-9c07-711042d1abbf
begin

#################### Generic metric ####################

	Christoffel(FS::FieldSpaceMetricType) = function(ϕ)
		dimension = length(ϕ)
		dG = reshape(jacobian(G(FS), ϕ), (dimension, dimension, dimension))
		Gamma = similar(dG)
		for a in 1:dimension, b in 1:dimension, c in 1:dimension
			Gamma[a, b, c] = 0.5*(dG[a, b, c] + dG[a, c, b] - dG[b, c, a])
		end
		Gamma
	end

#################### Canonical metric ####################
	
	Christoffel(C::Canonical) = function(ϕ)
		dimension = length(ϕ)
		reshape(zeros(dimension^3), (dimension, dimension, dimension))
	end

#################### Hyperbolic metric in 2D ####################
	
	Christoffel(H::Hyperbolic2D) = function(ϕ)
		dimension = length(ϕ)
		Gamma = reshape(zeros(dimension^3), (dimension, dimension, dimension))

		ϕ₁, ϕ₂ = ϕ
		Gamma[1, 2, 2] = - H.L*sinh(ϕ₁/H.L)*cosh(ϕ₁/H.L)
		Gamma[2, 1, 2] = Gamma[2, 2, 1] = H.L*sinh(ϕ₁/H.L)*cosh(ϕ₁/H.L)
		Gamma
	end
end;

# ╔═╡ e63e95a5-a25d-439b-8044-a2d61939be4a
md"""
# Defining a model
"""

# ╔═╡ 9e476ecb-3128-4a91-946e-51159b4364fe
begin
	abstract type ModelType end
	
	@with_kw struct Model <: ModelType
		nfields::Int64
		input_field_names::Tuple = ()

		@assert length(input_field_names) == length(Set(input_field_names)) "Repeated field names!"

		fields = isempty(input_field_names) ? Tuple(Symbol("ϕ"*subscript(i)) for i in 1:nfields) : input_field_names
		
		Potential::PotentialType
		Metric::FieldSpaceMetricType
	end
end;

# ╔═╡ e46818d3-07ba-40b4-889c-78a409e4e2d4
md"""
## Type of potential and its parameters
"""

# ╔═╡ 80594b16-6ce3-40dd-84e7-41ed51bc67fc
# @bind m Slider(1:10, default=5)
# @bind m Slider(0.005:0.001:0.015, default = 0.01)
begin
	N = 1000
	gₛ = 0.01
	lₛ = 501.961
	q = 1
	a₀ = 0.001
	a₁ = 0.0005
	b₁ = 0.001
	p = 5 	# Change p later
	V₀ = -1.1775724676177187e-8
end;

# ╔═╡ 2a449106-a6c4-44b1-9b00-0286601b2ef4
# Vtype = Quadratic((1, m))
Vtype = GenericPotential((N=N, gₛ=gₛ, lₛ=lₛ, q=q, a₀=a₀, a₁=a₁, b₁=b₁, p=p, V₀=V₀))

# ╔═╡ e480cc80-7ea5-49cd-a593-5b91b67874b5
md"""
## Type of field space metric and its parameters
"""

# ╔═╡ 2a9225fa-4d33-4235-96d6-855a42909298
# @bind L Slider(0.03:0.01:0.1, default = 0.05)

# ╔═╡ 64f543ad-db9c-48ae-8971-01a88ec7c18a
# Gtype = Canonical();
# Gtype = Hyperbolic2D(L)
Gtype = GenericMetric((N=N, gₛ=gₛ, lₛ=lₛ, q=q, p=p))

# ╔═╡ 7ae28d3d-1d42-494c-af60-95a2f3c56585
md"""
## Constructing the model
"""

# ╔═╡ 7134e45c-0530-4041-a00f-cfaae7a2a8c2
begin
	nfields = 2
	MyModel = Model(nfields = nfields, input_field_names=(:r, :θ), Potential = Vtype, Metric = Gtype)
end

# ╔═╡ 8e49b081-2da1-45ee-a0c5-1733667fcfd4
md"""
## Plotting the potential
"""

# ╔═╡ dfdba7fe-fd18-4825-b473-c304f4e5fecd
gr()

# ╔═╡ 9cf39b5b-2e12-4e37-9882-665032bcfe8d
function plotV(MyModel::Model, xrange = -10:0.1:10, yrange = -10:0.1:10)
	@assert MyModel.nfields == 2 "Can only plot 2D potential"
	
	pV = surface(xrange, yrange, (ϕ₁, ϕ₂) -> V(MyModel.Potential)([ϕ₁, ϕ₂]), xlabel=latexify(MyModel.fields[1]), ylabel=latexify(MyModel.fields[2]), zlabel=L"$V(\phi)$", dpi=300)

	plnV = surface(xrange, yrange, (ϕ₁, ϕ₂) -> log(V(MyModel.Potential)([ϕ₁, ϕ₂])), xlabel=latexify(MyModel.fields[1]), ylabel=latexify(MyModel.fields[2]), zlabel=L"$ln(V)(\phi)$", dpi=300)

	pV, plnV
end;

# ╔═╡ 4a21987b-a56c-4ad2-a71b-00369b6c8d57
plotV(MyModel, 10:0.1:40, 0:0.1:15)[1]

# ╔═╡ 90006555-5775-421f-b5e8-6a2ddf3d973a
plotV(MyModel)[2]

# ╔═╡ 059157c0-98ab-41e1-bbbc-a5530bb27ecc
md"""
# Background equations
"""

# ╔═╡ ab171ae1-fa68-4ac0-ae7e-33a2487ed695
md"""
We write the background equations as

$\zeta^\prime = f(\zeta)$

where $\zeta \equiv (\phi, \xi)$ and $f(\zeta)$ is given by

$f(\zeta) = \left(\begin{array}{cccc}  \xi^a \\ -\Gamma^a_{bc}\xi^b \xi^c + (\epsilon - 3) (\xi^a + \nabla^a \ln V) \end{array}\right)$

Note that prime denotes derivative with respect to the number of e-folds $N$.
"""

# ╔═╡ 77e12e51-84c1-48e9-827d-867277b7186f
ODE(MyModel::Model) = function f(dζ, ζ, p, N)
	ϕ, ξ = ζ[1:Int(length(ζ)/2)], ζ[Int(length(ζ)/2) + 1:end]
	ϵ = 0.5*norm(MyModel.Metric, ξ)(ϕ)^2
	
	@tensor dξ[a] := - inv(G(MyModel.Metric)(ϕ))[a, d] * Christoffel(MyModel.Metric)(ϕ)[d, b, c] * ξ[b] * ξ[c]
	dξ += (ϵ - 3)*(ξ + inv(G(MyModel.Metric)(ϕ))*gradient(ϕ -> log(V(MyModel.Potential)(ϕ)), ϕ))

	dζ[1] = ξ[1]
	dζ[2] = ξ[2]
	dζ[3] = dξ[1]
	dζ[4] = dξ[2]
end;

# ╔═╡ 8da3562d-9d13-46ce-91ce-0413bf89a530
md"""
## Initial conditions and number of e-folds
"""

# ╔═╡ 75ba08d6-3a55-4147-b239-e4786f8c1803
begin
	abstract type InitialConditions end
	
	struct SlowRoll <: InitialConditions
		ϕ₀::Array
	end
	
	initial(SR::SlowRoll, MyModel::Model) = vcat(SR.ϕ₀, -inv(G(MyModel.Metric)(SR.ϕ₀))*gradient(ϕ -> log(V(MyModel.Potential)(ϕ)), SR.ϕ₀))

	struct HyperInflation2D <: InitialConditions
		ϕ₀::Array
	end

	function initial(HI::HyperInflation2D, MyModel::Model)
		@assert typeof(MyModel.Metric) == Hyperbolic2D "Field space metric has to be Hyperbolic2D"

		L = MyModel.Metric.L
		
		ξ₁₀ = -3L
		ξ₂₀(ϵ) = 1/(L*sinh(HI.ϕ₀[1]/L))*sqrt( (3-ϵ)*gradient(ϕ -> log(V(MyModel.Potential)(ϕ)), HI.ϕ₀)[1] - 9L^2)
		
		to_root(ϵ) = norm(MyModel.Metric, [ξ₁₀, ξ₂₀(ϵ)])(HI.ϕ₀) - 2ϵ
		ϵ₀ = find_zero(to_root, 1e-5)
		
		vcat(HI.ϕ₀, [ξ₁₀, ξ₂₀(ϵ₀)])
	end
end;

# ╔═╡ 17f22706-f6cc-4a79-a733-1d98b160967f
begin
	ϕ₀ = [4., 0.95]
	ζ₀ = initial(SlowRoll(ϕ₀), MyModel)
	# ζ₀ = initial(HyperInflation2D(ϕ₀), MyModel)
	# ζ₀ = [ϕ₀[1], ϕ₀[2], .1, .1]
	Nmin, Nmax = 0., 60.
	Ns = (Nmin, Nmax)
end;

# ╔═╡ f6c9a053-85d9-47ce-b24f-3b27d6a297b0
md"""
## Setting up the ODE problem and solving it
"""

# ╔═╡ eaee4f56-a385-4e03-9bb3-e7bfe113d8f6
md"""
Condition to terminate at the end of inflation:
"""

# ╔═╡ 2db2be07-ddea-40dd-a2a9-cea5e2e694e6
begin
	condition_terminate(MyModel::Model) = function condition_terminate(ζ, N, integrator)		
	    ϕ, ξ = ζ[1:Int(length(ζ)/2)], ζ[Int(length(ζ)/2) + 1:end]
		ϵ = 0.5*norm(MyModel.Metric, ξ)(ϕ)^2
		1 - ϵ
	end
	
	affect!(integrator) = terminate!(integrator)
	end_cb(MyModel::Model) = ContinuousCallback(condition_terminate(MyModel), affect!)
end;

# ╔═╡ fec32909-1d73-4c36-8107-da1cb2f78055
# problem(MyModel::Model, IC::Vector, Ns::Tuple, abstol = 1e-8, reltol = 1e-8) = ODEProblem(ODE(MyModel), IC, Ns, callback = end_cb(MyModel), abstol = abstol, reltol = reltol);
problem(MyModel::Model, IC::Vector, Ns::Tuple, abstol = 1e-8, reltol = 1e-8) = ODEProblem(ODE(MyModel), IC, Ns, abstol = abstol, reltol = reltol);

# ╔═╡ b8fa1f85-c96a-4b52-bb5d-b65209441a4d
solution = solve(problem(MyModel, ζ₀, Ns), Tsit5())

# ╔═╡ a517ba42-291d-40c2-934d-75c30da7eff7
function plotsol!(MyModel::Model, solution::ODESolution, legend=false, label=missing)
	if typeof(MyModel.Metric) == Hyperbolic2D
		
		return plot!(solution, vars = (2, 1), legend=legend, label=label, dpi = 300, proj=:polar, xlabel = latexify(MyModel.fields[1]), ylabel=latexify(MyModel.fields[2]), lims=(0,0.5), yguidefontsize=8, title="Trajectory in the ("*latexify(MyModel.fields[1])*", "*latexify(MyModel.fields[2])*") plane")
	end

	return plot!(solution, vars = (1,2), legend=legend, label=label, arrow=true, dpi = 300, xlabel = latexify(MyModel.fields[1]), ylabel=latexify(MyModel.fields[2]))
end;

# ╔═╡ 7bfc172b-5ede-42bd-a526-42be51e42f2a
begin
	plot()
	plotsol!(MyModel, solution)
end

# ╔═╡ d470672a-0abc-4b1d-93c3-f66fc1d303bd
md"""
Total number of e-folds:
"""

# ╔═╡ c25a6cbc-392f-4cb8-9153-ce77025f6cdc
Nend = solution.t[end]

# ╔═╡ 5e91f713-8708-4aca-be0b-65b226087751
md"""
## Slow-roll parameters
"""

# ╔═╡ a1faeefb-3c5e-41da-9a77-a0d74a2db431
begin
	ϵ(MyModel::Model, sol::ODESolution) = function(N::Real)
		0.5*norm(MyModel.Metric, sol(N)[Int(length(sol(N))/2) + 1:end])(sol(N)[Int(length(sol(N))/2) + 1:end])^2
	end
	v(MyModel::Model, sol::ODESolution) = function(N::Real) sqrt(2ϵ(MyModel, sol)(N)) end
	
	η(MyModel::Model, sol::ODESolution) = function(N::Real)
		ζ, dζ = sol(N), derivative(N -> sol(N), N)
		ϕ, ξ = ζ[1:Int(length(ζ)/2)], ζ[Int(length(ζ)/2) + 1:end]
		@tensor covariant[a] := inv(G(MyModel.Metric)(ϕ))[a, d] * Christoffel(MyModel.Metric)(ϕ)[d, b, c] * ξ[b] * ξ[c]
		dζ[Int(length(dζ)/2) + 1:end] + covariant
	end

	ησ(MyModel::Model, sol::ODESolution) = function(N::Real)
		ζ, dζ = sol(N), derivative(N -> sol(N), N)
		ϕ, ξ = ζ[1:Int(length(ζ)/2)], ζ[Int(length(ζ)/2) + 1:end]
		dot(MyModel.Metric, η(MyModel, sol)(N), ξ)(ϕ)/norm(MyModel.Metric, ξ)(ϕ)
	end

	ηs(MyModel::Model, sol::ODESolution) = function(N::Real)
		ζ, dζ = sol(N), derivative(N -> sol(N), N)
		ϕ, ξ = ζ[1:Int(length(ζ)/2)], ζ[Int(length(ζ)/2) + 1:end]
		eσ = ξ/norm(MyModel.Metric, ξ)(ϕ)
		
		es = (Matrix{Float64}(I, length(ϕ), length(ϕ)) - eσ*eσ')*η(MyModel, sol)(N)
		norm(MyModel.Metric, ξ)(ϕ) == 0 ? perp_to(MyModel.Metric, eσ)(ϕ) : es /= norm(MyModel.Metric, ξ)(ϕ)
	
		dot(MyModel.Metric, η(MyModel, sol)(N), es)(ϕ)
	end
end;

# ╔═╡ f5a46baf-e5e8-405f-9153-ddf9df5c6dbb
begin
	function plotϵ!(MyModel::Model, solution::ODESolution, legend=false, label=missing)
		Nend = solution.t[end]
		plot!(Nmin:0.01:Nend, ϵ(MyModel, solution), legend=legend, label=label, xlabel=latexify("N"), ylabel=latexify("ϵ"))
	end
	
	function plotησ!(MyModel::Model, solution::ODESolution, legend=false, label=missing)
		Nend = solution.t[end]
		pησ = plot!(Nmin:0.1:Nend, N -> ησ(MyModel, solution)(N)/v(MyModel, solution)(N), legend=legend, label=label, xlabel=latexify("N"), ylabel=latexify("η_σ/v"))
	end
	
	function plotηs!(MyModel::Model, solution::ODESolution, legend=false, label=missing)
		Nend = solution.t[end]
		pηs = plot!(Nmin:0.1:Nend, N -> ηs(MyModel, solution)(N)/v(MyModel, solution)(N), legend=legend, label=label, xlabel=latexify("N"), ylabel=latexify("η_s/v"))
	end
end;

# ╔═╡ 724f0557-7e63-4709-8298-f9e8868115a3
begin
	plot()
	pϵ = plotϵ!(MyModel, solution)
	xlabel!("")
	plot()
	pησ = plotησ!(MyModel, solution)
	xlabel!("")
	plot()
	pηs = plotηs!(MyModel, solution)
	plot(pϵ, pησ, pηs, layout=(3,1), size=(600,600), dpi=300)
end

# ╔═╡ c491e06e-1f00-4a32-84b1-fe6a9adcffb2
md"""
## Varying the parameters
"""

# ╔═╡ 1d5d1c0b-2efd-4e9a-b633-37f68fdc9ff5
begin
	modeltag = "HI"
	# pname, prange = :m, 1:1:10
	pname, prange = :L, 0.03:0.01:0.1
	Vt(p::Real) = Quadratic((0.01, 0))
	Gt(p::Real) = Hyperbolic2D(p)
	# Gt(p::Real) = Canonical()

	MyMdl(p::Real) = Model(nfields = nfields, input_field_names = (:ϕ, :χ), Potential = Vt(p), Metric = Gt(p))

	ϕ₀p = [10., 1.]
	# initialpt = SlowRoll(ϕ₀p)
	initialpt = HyperInflation2D(ϕ₀p)
	init(p::Real) = initial(initialpt, MyMdl(p))
end;

# ╔═╡ 32c62918-0cfd-479c-b1b3-e7ca6276f098
begin
	Models, initials, problems, solutions = [], [], [], []
	for p in prange
		push!(Models, MyMdl(p))
	
		push!(initials, initial(initialpt, MyMdl(p)))
	
		push!(problems, problem(MyMdl(p), initial(initialpt, MyMdl(p)), Ns, 1e-300))
		push!(solutions, solve(problem(MyMdl(p), initial(initialpt, MyMdl(p)), Ns, 1e-300), Tsit5()))
	end
end

# ╔═╡ 2eda1cff-3a01-4f45-848c-8f7c18b9436f
begin
	ptrajs = plot(dpi=300)
	for (n, p) in enumerate(prange)
		plotsol!(MyMdl(p), solutions[n], :topleft, latexify(string(pname)*" = $p"))
	end
	plot!()
	xlabel!(latexify(MyModel.fields[1]))
	ylabel!(latexify(MyModel.fields[2]))

	savefig(ptrajs, "output/"*modeltag*"_trajs")
	plot!()
end

# ╔═╡ b068fadb-94eb-4956-b027-9560959a1da0
begin
	pϵp = plot(dpi=300)
	for (n, p) in enumerate(prange)
		plotϵ!(MyMdl(p), solutions[n], :topleft, latexify(string(pname)*" = $p"))
	end
	plot!()

	savefig(pϵp, "output/"*modeltag*"_epsilon")
	plot!()
end

# ╔═╡ a5487e79-bb6e-4d38-ab2f-37e63b32aba4
begin
	pησp = plot(dpi=300)
	for (n, p) in enumerate(prange)
		plotησ!(MyMdl(p), solutions[n], :topleft, latexify(string(pname)*" = $p"))
	end
	ylims!(-1, 3)
	
	savefig(pησp, "output/"*modeltag*"_etapar")
	plot!()
end

# ╔═╡ c95256ff-43a8-45cf-a315-5bc7c7ae3fcb
begin
	pησs = plot(dpi=300)
	for (n, p) in enumerate(prange)
		plotηs!(MyMdl(p), solutions[n], :topleft, latexify(string(pname)*" = $p"))
	end
	plot!()

	savefig(pησs, "output/"*modeltag*"_etaperp")
	plot!()
end

# ╔═╡ Cell order:
# ╟─e2306bb5-b84f-482a-9b06-48af1c79cb89
# ╟─b363d59b-efbb-4e35-bb26-ef884ceadfda
# ╟─c88612f4-0fa2-4d3e-8cd4-9714a50715c9
# ╟─75081cf5-a327-4e4e-8569-8a96f5179479
# ╟─50172480-9ef3-11ec-21a7-f74589ea5026
# ╟─277fb121-e8eb-48f5-b398-078aaa9ce3a6
# ╟─367bddc1-674b-4352-90e0-ffb4b0ef4545
# ╟─500c52a4-318d-4300-9f23-ceaffa1b83e0
# ╟─bdb142de-8840-4fc2-9f11-fd4132df64f4
# ╟─a8b4458b-fe05-4868-8fd0-3dbb87d7a077
# ╟─415291e0-77f8-4e85-89ae-2a0010441f4d
# ╟─8a710b6a-c1d9-4293-929e-6e5e9cdf5bca
# ╟─e31e62df-e592-48c0-9810-b77d40d18a21
# ╟─57599d34-6813-4a1d-8710-8af8fd00aa82
# ╟─46e7357a-7bd8-4c81-9299-1cd419e21da1
# ╠═973052be-4ed9-4dfa-bacc-c8c798683b77
# ╟─8ccd9f97-1248-4f69-a9a8-d1c18965e9b6
# ╟─fe6432a1-94cc-4aa5-821e-c5085e6d54d8
# ╟─64af0ee4-e86b-445b-aa64-0bb930e458b8
# ╟─df156a1e-edf2-40d0-9ac0-d3ffb0c4e49a
# ╟─73e7ba6e-7a4e-4051-9c07-711042d1abbf
# ╟─05d66135-0b1a-455b-b2de-bd2aabfd1bd7
# ╟─42346539-58cf-45ea-916e-9821c7e80f6a
# ╠═d59fa4db-cd80-402e-a024-974a9ddf2e46
# ╟─e63e95a5-a25d-439b-8044-a2d61939be4a
# ╟─9e476ecb-3128-4a91-946e-51159b4364fe
# ╟─e46818d3-07ba-40b4-889c-78a409e4e2d4
# ╠═80594b16-6ce3-40dd-84e7-41ed51bc67fc
# ╠═2a449106-a6c4-44b1-9b00-0286601b2ef4
# ╟─e480cc80-7ea5-49cd-a593-5b91b67874b5
# ╠═2a9225fa-4d33-4235-96d6-855a42909298
# ╠═64f543ad-db9c-48ae-8971-01a88ec7c18a
# ╟─7ae28d3d-1d42-494c-af60-95a2f3c56585
# ╠═7134e45c-0530-4041-a00f-cfaae7a2a8c2
# ╟─8e49b081-2da1-45ee-a0c5-1733667fcfd4
# ╟─dfdba7fe-fd18-4825-b473-c304f4e5fecd
# ╟─9cf39b5b-2e12-4e37-9882-665032bcfe8d
# ╟─4a21987b-a56c-4ad2-a71b-00369b6c8d57
# ╟─90006555-5775-421f-b5e8-6a2ddf3d973a
# ╟─059157c0-98ab-41e1-bbbc-a5530bb27ecc
# ╟─ab171ae1-fa68-4ac0-ae7e-33a2487ed695
# ╟─77e12e51-84c1-48e9-827d-867277b7186f
# ╟─8da3562d-9d13-46ce-91ce-0413bf89a530
# ╠═75ba08d6-3a55-4147-b239-e4786f8c1803
# ╠═17f22706-f6cc-4a79-a733-1d98b160967f
# ╟─f6c9a053-85d9-47ce-b24f-3b27d6a297b0
# ╟─eaee4f56-a385-4e03-9bb3-e7bfe113d8f6
# ╟─2db2be07-ddea-40dd-a2a9-cea5e2e694e6
# ╠═fec32909-1d73-4c36-8107-da1cb2f78055
# ╠═b8fa1f85-c96a-4b52-bb5d-b65209441a4d
# ╟─a517ba42-291d-40c2-934d-75c30da7eff7
# ╟─7bfc172b-5ede-42bd-a526-42be51e42f2a
# ╟─d470672a-0abc-4b1d-93c3-f66fc1d303bd
# ╟─c25a6cbc-392f-4cb8-9153-ce77025f6cdc
# ╟─5e91f713-8708-4aca-be0b-65b226087751
# ╟─a1faeefb-3c5e-41da-9a77-a0d74a2db431
# ╟─f5a46baf-e5e8-405f-9153-ddf9df5c6dbb
# ╟─724f0557-7e63-4709-8298-f9e8868115a3
# ╟─c491e06e-1f00-4a32-84b1-fe6a9adcffb2
# ╠═1d5d1c0b-2efd-4e9a-b633-37f68fdc9ff5
# ╟─32c62918-0cfd-479c-b1b3-e7ca6276f098
# ╟─2eda1cff-3a01-4f45-848c-8f7c18b9436f
# ╟─b068fadb-94eb-4956-b027-9560959a1da0
# ╟─a5487e79-bb6e-4d38-ab2f-37e63b32aba4
# ╟─c95256ff-43a8-45cf-a315-5bc7c7ae3fcb
