# LogisticOptTools.jl

A simple package to benchmark optimization solvers on
logistic regression problems.

Let `X \in \mathbb{R}^{n \times p}` with `n` points and `p` features,
and `y \in \mathbb{R}^n` a vector with values in `\{-1, 1\}` specifying the
binary class of each point in `X`.
We formulate a logistic regression problem as

```math

\min_{\theta \in \mathbb{R}^p} \dfrac{1}{n} \sum_{i=1}^n log(1 + exp(-y_i X_i^\top \theta))  + \lambda || \theta ||

```



## Basic usage

### Step 1: installing the package
LogisticOptTools comes with few dependencies, and the installation
should be straightforward.

```julia
pkg> add https://github.com/frapac/LogisticOptTools.jl
julia> using LogisticOptTools
julia> const LOT = LogisticOptTools

```

### Step 2: loading a dataset
First, download a LIBSVM dataset from this [URL](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html).
`LogisticOptTools` implements a parser for LIBSVM files. To load a dataset
as a dense matrix, the procedure is:

```julia
shell> ls .
covtype.binary.bz2
# Parse as Float64
julia> dataset = LOT.parse_libsvm("covtype.binary.bz2", Float64)
# Load as dense matrix
julia> X = LOT.to_dense(dataset)
julia> y = dataset.labels

```

You could load the dataset as a sparse matrix simply by replacing
`LOT.to_dense` by `LOT.to_sparse`.

### Step 3: computing empirical loss and gradients
Let `theta` be a parameter.
The structure `LogitData` implements the callbacks for the empirical
loss, the gradient, the Hessian and the Hessian-vector product:

```julia
julia> logit = LOT.LogitData(X, y)
julia> p = LOT.dim(data)
# Generate random parameter
julia> theta = randn(p)
# Compute empirical loss
julia> f = LOT.loss(theta, logit)
# Get gradient
julia> grad = zeros(p)
julia> LOT.gradient!(grad, theta, logit)

```

The structure `GeneralizedLinearModel` allows to plug a penalty (either
`L1` or `L2`).

```julia
julia> penalty = LOT.L2Penalty(1.0)
julia> model = GeneralizedLinearModel(logit, penalty)
# Get empirical loss + penalty
julia> f = LOT.loss(theta, model)
# Get gradient of loss + penalty
julia> grad = zeros(p)
julia> LOT.gradient!(grad, theta, model)

```

### Step 4: fitting the logistic regression
Once the dataset loaded, you could fit a logistic regression by
using Optim. You just need to wrap properly the logit callbacks
so that they get a valid signature:

```julia
# Wrap logit callbacks for Optim
julia> eval_f = x -> LOT.loss(x, model)
julia> eval_g(g, x) = LOT.gradient!(g, x, model)

```

and then call any optimization algorithm:

```julia
julia> algo = BFGS()
julia> options = Optim.Options(iterations=250, g_tol=1e-5, show_trace=true)
julia> x0 = zeros(p)
julia> res_joptim = Optim.optimize(eval_f, eval_g, x0, algo, options)

```


## Benchmarks

TODO
