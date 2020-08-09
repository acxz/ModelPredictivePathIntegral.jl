function mppisim(func_is_task_complete, func_control_update_converged,
    func_comp_weights, func_term_cost, func_run_cost,func_gen_next_ctrl,
    func_state_est, func_apply_ctrl, func_g, func_F, func_state_transform,
    func_control_transform, func_filter_du, num_samples, learning_rate,
    init_state, init_ctrl_seq, ctrl_noise_covar, time_horizon,
    per_ctrl_based_ctrl_noise, real_traj_cost, plot_traj, print_sim, print_mppi,
    save_sampling, sampling_filename)

    # time stuff
    num_timesteps = size(init_ctrl_seq, 2);
    dt = time_horizon / num_timesteps;
    time = 0.;
    time_hist = [time];

    # state stuff
    state_dim = size(init_state, 1);
    x_hist = init_state;
    curr_x = init_state;

    # sample state stuff
    sample_init_state = func_state_transform(init_state);
    sample_x_hist = sample_init_state;

    # control history
    control_dim = size(init_ctrl_seq, 1);
    u_hist = Array{typeof(init_ctrl_seq)}(undef, control_dim, 0)
    sample_u_hist = Array{typeof(init_ctrl_seq)}(undef, control_dim, 0)

    # control sequence
    sample_u_traj = init_ctrl_seq;

    # trajectory cost history
    rep_traj_cost_hist = Array{typeof(init_ctrl_seq)}(undef, 1, 0)

    total_timestep_num = 1;
    while(func_is_task_complete(curr_x, time) == false)

        # Use mppi
        sample_u_traj, rep_traj_cost = mppi(func_control_update_converged, 
        func_comp_weights, func_term_cost, func_run_cost, func_g, func_F, 
        func_state_transform, func_filter_du, num_samples, learning_rate, 
        curr_x, sample_u_traj, ctrl_noise_covar, time_horizon, 
        per_ctrl_based_ctrl_noise, real_traj_cost, print_mppi, save_sampling, 
        sampling_filename);

        # Transform from sample_u to u
        u = func_control_transform(sample_x_hist[:,total_timestep_num], sample_u_traj[:,1], dt);

        # Apply control and log data
        true_x = func_apply_ctrl(x_hist[:,total_timestep_num], u, dt);

        # state estimation after applying control
        curr_x = func_state_est(true_x);

        # Transform from state used in dynamics vs state used in control sampling
        sample_x = func_state_transform(curr_x);

        # Log state data
        x_hist = hcat(x_hist, curr_x);
        sample_x_hist = hcat(sample_x_hist, sample_x);

        # Log control data
        u_hist = hcat(u_hist, u);
        sample_u_hist = hcat(sample_u_hist, sample_u_traj[:,1]);

        # Log trajectory cost data
        rep_traj_cost_hist = hcat(rep_traj_cost_hist, rep_traj_cost);

        if(print_sim)
          println("Simtime: $(time)");
        end

        # Move time forward
        time = time + dt;
        push!(time_hist, time);

        # Warmstart next control trajectory using past generated control trajectory
        sample_u_traj[:,1:end-1] = sample_u_traj[:,2:end];
        sample_u_traj[:, end] = func_gen_next_ctrl(sample_u_traj[:, end]);

        total_timestep_num = total_timestep_num + 1;

        end
      
    return x_hist, u_hist, sample_x_hist, sample_u_hist, rep_traj_cost_hist, time_hist
  
end