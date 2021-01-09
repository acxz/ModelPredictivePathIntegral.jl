import LinearAlgebra: Diagonal, I
import ModelPredictivePathIntegral: mppisim, MppisimParams, MppiParams
import Plots: plot, plotly

function apply_ctrl(x, u, dt)
    mc = 1;
    mp = 0.01;
    l = 0.25;
    g = 9.8;

    xpos = x[1];
    θ = x[2];
    xpos′ = x[3];
    θ′= x[4];

    f = u[1];

    x′ = similar(x);

    x′[1] = xpos′;
    x′[2] = θ′;
    x′[3] = (1/(mc + mp * sin(θ)^2)) * (f + mp * sin(θ) * (l * θ′^2 + g * cos(θ)));
    x′[4] = (1/(l * (mc + mp * sin(θ)^2))) * (-f * cos(θ) - mp * l * θ′^2 * cos(θ) * sin(θ) - (mc + mp) * g * sin(θ));

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
    mc = 1;
    mp = 0.01;
    l = 0.25;
    g = 9.8;

    xpos = x[1,:];
    θ = x[2,:];
    xpos′ = x[3,:];
    θ′= x[4,:];

    f = u[1,:];

    x′ = similar(x);

    x′[1,:] = xpos′;
    x′[2,:] = θ′;
    @. x′[3,:] = (1/(mc + mp * sin(θ)^2)) * (f + mp * sin(θ) * (l * θ′^2 + g * cos(θ)));
    @. x′[4,:] = (1/(l * (mc + mp * sin(θ)^2))) * (-f * cos(θ) - mp * l * θ′^2 * cos(θ) * sin(θ) - (mc + mp) * g * sin(θ));

    return x + x′ * dt
end

function gen_next_ctrl(u)
    return u
    # ? randn(1)
end

function is_task_complete(x, t)
    if t > 5
        return true
    end
    return false
end

function run_cost(x)
    Q = Diagonal([1, 1, 1, 1]);
    goal_state = [0, pi, 0, 0];

    dx = similar(x);
    @. dx = x - goal_state;

    return 1/2 * sum(dx .* (Q * dx), dims=1);
end

function term_cost(x)
    Qf = Diagonal([0, 700, 1000, 500])
    goal_state = [0, pi, 0, 0];

    dx = similar(x);
    @. dx = x - goal_state;

    return 1/2 * sum(dx .* (Qf * dx), dims=1);
end

function main()
    horizon = 20;
    ctrl_dim = 1;

    mppi_params = MppiParams(
        num_samples = 5000,
        dt = 0.05,
        learning_rate = 0.01,
        init_state = [0, 0, 0, 0],
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

    plotly()

    x_plot = plot(time_hist, [x_hist[1,:], x_hist[2,:]],
         title = "State",
         xlabel = "Time (s)",
         ylabel = "Value",
         label = ["x (m)" "θ (rad)" "x′ (m/s)" "θ′ (rad/s)"]);

    u_plot = plot(time_hist[1:end-1], u_hist[1,:],
         title = "Control",
         xlabel = "Time (s)",
         ylabel = "Value",
         label = "f (N)");

    display(x_plot)
    display(u_plot)
end

main()
