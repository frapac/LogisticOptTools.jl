# LogisticOptTools.jl

| **Build Status** |
|:----------------:|
| [![Build Status][build-img]][build-url] |

A Julia package to benchmark optimization solvers on
logistic regression problems.

* MIT license
* Install using `julia> ] add LogisticOptTools`


## Basic usage

Suppose you import LogisticOptTools in your REPL

```julia
julia> using LogisticOptTools
julia> const LOT = LogisticOptTools

```

Suppose you have available a matrix of features `X` and a vector of observations `y`,
and you want to fit a logistic model onto this data.
You could instantiate a new logistic model simply by typing

```julia
julia> logreg = LOT.LogisticRegressor(X, y,
                                      fit_intercept=true,
                                      penalty=LOT.L2Penalty(1.0))
```

and then fit the logistic regression with `Optim.jl`:

```julia
julia> p = LOT.nfeatures(logreg)
julia> x0 = zeros(p) ; algo = LBFGS()
julia> res = Optim.optimize(logreg, x0, algo)
# Fetch optimal parameters
julia> p_opt = res.minimizer

```


## Benchmarks

LogisticOptTools could use the different algorithms implemented in Optim.jl.
We depict in the following figure a comparison of three algorithms, when
fitting a logistic model on the `covtype` dataset (581,012 data, 54 features).

![benchmark](https://github.com/frapac/LogisticOptTools.jl/examples/iter_g.png)

For an example on how to use other solvers, we have implemented
in `examples/tron.jl` a resolution of a logistic regression problem
with `tron`, a solver implemented [JSOSolvers.jl](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl/).


## Use-cases

### Import LIBSVM datasets

LogisticOptTools supports the `libsvm` format. Once a dataset downloaded
from the [website](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html),
you could load it in the Julia REPL with

```julia
shell> ls .
covtype.binary.bz2
# Parse as Float64
julia> dataset = LOT.parse_libsvm("covtype.binary.bz2", Float64)
# Load as dense matrix
julia> X = LOT.to_dense(dataset)
julia> y = dataset.labels

```

You could load the dataset as a sparse matrix just by replacing
`LOT.to_dense` with `LOT.to_sparse`.


### Advanced usages

You could find in `examples/` a few examples on:

* optimizing the L2 penalty parameter with `Optim.jl`
* fitting a sparse regression (l0-l2 logistic regression) with JuMP and a MILP solver


[build-img]: https://travis-ci.org/frapac/LogisticOptTools.jl.svg?branch=master
[build-url]: https://travis-ci.org/frapac/LogisticOptTools.jl
