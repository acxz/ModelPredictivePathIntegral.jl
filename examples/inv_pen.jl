import LinearAlgebra: I
import ModelPredictivePathIntegral: mppisim, MppisimParams, MppiParams
import Plots: plotlyjs, plot

function apply_ctrl(x, u, dt)
    m = 1;
    l = 1;
    g = 9.8;
    b = 0;

    θ = x[1]
    θ′= x[2]
    τ = u[1]

    x′= similar(x);

    x′[1] = θ′;
    x′[2] = -g / l * sin(θ) - (b * θ′ + τ) / (m * l ^ 2);

    return x + x′ * dt
end

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

    θ = x[1]
    θ′= x[2]
    τ = u[1]

    x′= similar(x);

    x′[1] = θ′;
    x′[2] = -g / l * sin(θ) - (b * θ′ + τ) / (m * l ^ 2);

    return x + x′ * dt
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

    dx = x - goal_state;

    return 1/2 * dx'* Q * dx;
end

function term_cost(x)
    Qf = [[100., 0] [0, 100]];
    goal_state = [pi, 0];

    dx = x - goal_state;

    return 1/2 * dx' * Qf * dx;
end

function main()
    horizon = 10;
    ctrl_dim = 1;

    mppi_params = MppiParams(
        num_samples = 5000,
        dt = 0.1,
        learning_rate = 0.01,
        init_state = [0, 0],
        init_ctrl_seq = randn(ctrl_dim, horizon),
        ctrl_noise_covar = reshape([5e-1], (1, 1)),
        per_ctrl_based_ctrl_noise = 0.999,
        real_traj_cost = true,
        plot_traj = true,
        print_mppi = true,
        save_sampling = false,
        sampling_filename = "inv_pen",
        func_control_update_converged = control_update_converged,
        func_comp_weights = comp_weights,
        func_term_cost = term_cost,
        func_run_cost = run_cost,
        func_F = F,
    )

    mppisim_params = MppisimParams(
        print_sim = true,
        func_is_task_complete = is_task_complete,
        func_apply_ctrl = apply_ctrl,
        func_gen_next_ctrl = gen_next_ctrl,
        mppi_params = mppi_params
    )

    @time x_hist, u_hist, sample_x_hist, sample_u_hist, rep_traj_cost_hist,
    time_hist = mppisim(mppisim_params);

    plotlyjs()
    x_plot = plot(time_hist, [x_hist[1,:], x_hist[2,:]],
         title = "State",
         xlabel = "Time (s)",
         ylabel = "Value",
         label = ["θ (rad)" "θ′ (rad/s)"]);

    u_plot = plot(time_hist[1:end-1], u_hist[1,:],
         title = "Control",
         xlabel = "Time (s)",
         ylabel = "Value",
         label = "τ (N ⋅ m)");

    display(x_plot)
    display(u_plot)
end

main()
