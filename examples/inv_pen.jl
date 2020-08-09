# TODO: Use this with openai gym env control theory
#https://gym.openai.com/envs/#classic_control

# TODO: define struct of parameters and functions

import LinearAlgebra: I
import ModelPredictivePathIntegral: mppisim

function apply_ctrl(x, u, dt)
    m = 1;
    l = 1;
    g = 9.8;
    b = 0;

    θ = x[1,:]
    θ′  = x[2,:]

    x′ = similar(x);

    x′[1,:] = θ′;
    @. x′[2,:] = -g / l * sin(θ) - (b * θ′ + u[1,:]) / (m * l ^ 2);

    return x + x′ * dt
end

# TODO can move this into src
function comp_weights(traj_cost)
    λ = 0.01
    val, ind = findmin(traj_cost)
    w = similar(traj_cost)
    @. w = exp(-1 / λ * (traj_cost - val))
    return w / sum(w)
end

function control_update_converged(du, iteration)
    tol = 0.01
    max_iteration = 5;
    if iteration > max_iteration
        return true
    end
    return false
end

function F(x, u, dt)
    m = 1;
    l = 1;
    g = 9.8;
    b = 0;

    θ = x[1,:]
    θ′  = x[2,:]

    x′ = similar(x);

    x′[1,:] = θ′;
    @. x′[2,:] = -g / l * sin(θ) - (b * θ′ + u[1,:]) / (m * l ^ 2);

    return x + x′ * dt
end

function filter_du(du)
    return du
end

function g(u)
    clamped_u = u
end

function gen_next_ctrl(u)
    return randn(1)
end

function is_task_complete(x, t)
    if t > 5
        return true
    end
    return false
end

function run_cost(x)
    Q = [[1., 0] [0, 1]];
    goal_state = [pi, 0];

    dx = similar(x);
    @. dx = x - goal_state;

    return 1/2 * sum(dx .* (Q * dx), dims=1);
end

function state_est(true_x)
    xdim = size(true_x, 1);
    H = I(xdim);
    return H * true_x;
end

function term_cost(x)
    Qf = [[100., 0] [0, 100]];
    goal_state = [pi, 0];

    dx = similar(x);
    @. dx = x - goal_state;

    return 1/2 * sum(dx .* (Qf * dx), dims=1);
end

function state_transform(x)
    return x
end

function control_transform(sample_x, sample_u, dt)
    return sample_u
end

# TODO time,dt makes more sense than time,timesteps
num_samples = 5000; # type stability needs to be Int
time_horizon = 1.;
num_timesteps = 50;
ctrl_dim = 1;
init_ctrl_seq = randn(ctrl_dim, num_timesteps);
init_state = [0., 0.];
ctrl_noise_covar = reshape([5e-1], (1, 1)); # ctrl_dim by ctrl_dim
learning_rate = 0.01;
per_ctrl_based_ctrl_noise = 0.999;
real_traj_cost = true;
plot_traj = true;
print_sim = true;
print_mppi = true;
#print_sim = false;
#print_mppi = false;
save_sampling = false; # Saves 0.8 GB to disk. This will slow the program down
sampling_filename = "inv_pen";

# todo call method
x_hist, u_hist, sample_x_hist, sample_u_hist, rep_traj_cost_hist, time_hist =
    mppisim(is_task_complete, control_update_converged, comp_weights, term_cost,
    run_cost, gen_next_ctrl, state_est, apply_ctrl, g, F,
    state_transform, control_transform, filter_du, num_samples, learning_rate,
    init_state, init_ctrl_seq, ctrl_noise_covar, time_horizon,
    per_ctrl_based_ctrl_noise, real_traj_cost, plot_traj, print_sim, print_mppi,
    save_sampling, sampling_filename);

# TODO Plot
using Plots
plot(time_hist, x_hist[1,:])
