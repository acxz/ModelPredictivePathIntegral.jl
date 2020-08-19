import Parameters: @with_kw, @unpack

@with_kw struct MppisimParams
    print_sim::Bool = false
    func_is_task_complete::Function
    func_apply_ctrl::Function
    func_gen_next_ctrl::Function = gen_next_ctrl
    func_control_transform::Function = control_transform
    mppi_params::MppiParams
end

function mppisim(mppisim_params)

    # Unpack mppisim_params
    @unpack print_sim, func_is_task_complete, func_apply_ctrl, func_gen_next_ctrl,
    func_control_transform, mppi_params = mppisim_params

    # Unpack mppi_params
    @unpack num_samples, learning_rate, dt, init_state, init_ctrl_seq,
    ctrl_noise_covar, per_ctrl_based_ctrl_noise, real_traj_cost,
    plot_traj, print_mppi, save_sampling,
    sampling_filename, func_control_update_converged,
    func_comp_weights, func_term_cost, func_run_cost,
    func_g, func_F, func_state_transform,
    func_filter_du = mppi_params

    # time stuff
    time = 0.;
    time_hist::Vector{Float32}= []
    time_hist = vcat(time_hist, time)

    # state stuff
    state_dim = size(init_state, 1);
    x_hist::Vector{Vector{Float32}} = []
    x_hist = vcat(x_hist, [init_state])
    curr_x = init_state;

    # sample state stuff
    sample_init_state = func_state_transform(init_state);
    sample_x_hist = [similar(sample_init_state)];

    # control history
    control_dim = size(init_ctrl_seq[1], 1);
    u_hist::Vector{Vector{Float32}} = []
    sample_u_hist::Vector{Vector{Float32}} = []

    # control sequence
    sample_u_traj = init_ctrl_seq;

    # trajectory cost history
    rep_traj_cost_hist::Vector{Float32}= []

    total_timestep_num = 1;
    while(func_is_task_complete(curr_x, time) == false)

        # Use mppi
        mppi_params = MppiParams(
            num_samples = num_samples,
            learning_rate = learning_rate,
            dt = dt,
            init_state = curr_x,
            init_ctrl_seq = sample_u_traj,
            ctrl_noise_covar = ctrl_noise_covar,
            per_ctrl_based_ctrl_noise = per_ctrl_based_ctrl_noise,
            real_traj_cost = real_traj_cost,
            plot_traj = plot_traj,
            print_mppi = print_mppi,
            save_sampling = save_sampling,
            sampling_filename = sampling_filename,
            func_control_update_converged = func_control_update_converged,
            func_comp_weights = func_comp_weights,
            func_term_cost = func_term_cost,
            func_run_cost = func_run_cost,
            func_g = func_g,
            func_F = func_F,
            func_state_transform = func_state_transform,
            func_filter_du = func_filter_du,
        )

        sample_u_traj, rep_traj_cost = mppi(mppi_params);

        # Transform from sample_u to u
        u = func_control_transform(sample_x_hist[total_timestep_num], sample_u_traj[1], dt);

        # Apply control and log data
        curr_x = func_apply_ctrl(x_hist[total_timestep_num], u, dt);

        # Transform from state used in dynamics vs state used in control sampling
        sample_x = func_state_transform(curr_x);

        # Log state data
        x_hist = vcat(x_hist, [curr_x]);
        sample_x_hist = vcat(sample_x_hist, [sample_x]);

        # Log control data
        u_hist = vcat(u_hist, [u]);
        sample_u_hist = vcat(sample_u_hist, [sample_u_traj[1]]);

        # Log trajectory cost data
        rep_traj_cost_hist = vcat(rep_traj_cost_hist, [rep_traj_cost]);

        if(print_sim)
          println("Simtime: $time");
        end

        # Move time forward
        time = time + dt;
        time_hist = vcat(time_hist, [time]);

        # Warmstart next control trajectory using past generated control trajectory
        sample_u_traj[1:end-1] = sample_u_traj[2:end];
        sample_u_traj[end] = func_gen_next_ctrl(sample_u_traj[end]);

        total_timestep_num = total_timestep_num + 1;

        end

    return x_hist, u_hist, sample_x_hist, sample_u_hist, rep_traj_cost_hist, time_hist

end

function gen_next_ctrl(u)
    return u
end

function control_transform(sample_x, sample_u, dt)
    return sample_u
end
