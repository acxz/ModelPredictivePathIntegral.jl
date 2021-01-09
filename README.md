# ModelPredictivePathIntegral.jl

A sampling based MPC derived from free energy principles.

## Usage
`mppi(...)` is just a function which outputs the optimized control value.
`mppisim(...)` is also provided as an mpc wrapper for the `mppi` function.

## Examples
In the `examples/` directory some example code is provided that uses
`mppisim(...)`.

To run the example code just run `include('examples/<example>_main.jl')` from
Julia. You can `] activate .` to ensure you have all the correct dependencies
before running the example script.

## References

```
G. Williams, P. Drews, B. Goldfain, J. M. Rehg and E. A. Theodorou, "Information-Theoretic Model Predictive Control: Theory and Applications to Autonomous Driving," in IEEE Transactions on Robotics, vol. 34, no. 6, pp. 1603-1622, Dec. 2018.
```
