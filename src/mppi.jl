# TODO Should I broadcast functions or vectorize inside of them
import Statistics: mean

function mppi(func_control_update_converged, func_comp_weights, func_term_cost,
    func_run_cost, func_g, func_F, func_state_transform, func_filter_du,
    num_samples, learning_rate, init_state, init_ctrl_seq, ctrl_noise_covar,
    time_horizon, per_ctrl_based_ctrl_noise, real_traj_cost, print_mppi,
    save_sampling, sampling_filename)

    # time stuff
    num_timesteps = size(init_ctrl_seq, 2);
    dt = time_horizon / num_timesteps;

    # sample state stuff
    sample_init_state = func_state_transform(init_state);
    sample_state_dim = size(sample_init_state,1);

    # state trajectories
    real_x_traj = zeros(sample_state_dim, num_timesteps + 1);
    real_x_traj[:,1] = sample_init_state;
    x_traj = zeros(sample_state_dim, num_samples, num_timesteps + 1);
    x_traj[:,:,1] = repeat(sample_init_state,outer=[1, num_samples]);

    # control stuff
    control_dim = size(init_ctrl_seq, 1);
    du = typemax(Float64) * ones(control_dim, num_timesteps);

    # control sequence
    sample_u_traj = init_ctrl_seq;
    last_sample_u_traj = sample_u_traj;

    # sampled control trajectories
    v_traj = zeros(control_dim, num_samples, num_timesteps);

    # Begin mppi
    iteration = 1;
    while(func_control_update_converged(du, iteration) == false)

        # Noise generation
        flat_distribution = randn(control_dim, num_samples * num_timesteps);
        ctrl_noise_flat = ctrl_noise_covar * flat_distribution;
        ctrl_noise = reshape(ctrl_noise_flat, (control_dim, num_samples, num_timesteps));

        # Compute sampled control trajectories
        ctrl_based_ctrl_noise_samples = floor(Int, per_ctrl_based_ctrl_noise * num_samples);
        if (ctrl_based_ctrl_noise_samples == 0)
            v_traj = ctrl_noise;
        elseif (ctrl_based_ctrl_noise_samples == num_samples)
            v_traj = repeat(reshape(sample_u_traj, (control_dim, 1, num_timesteps)), outer=[1, num_samples, 1]) + ctrl_noise;
        else
            v_traj[:,1:ctrl_based_ctrl_noise_samples,:] = repeat(reshape(sample_u_traj, (control_dim, 1, num_timesteps)), outer=[1, ctrl_based_ctrl_noise_samples, 1]) + ctrl_noise[:,1:ctrl_based_ctrl_noise_samples,:];
            v_traj[:,ctrl_based_ctrl_noise_samples+1:end,:] = ctrl_noise[:,ctrl_based_ctrl_noise_samples+1:end,:];
        end

        traj_cost = zeros(1, num_samples);

        for timestep_num = 1:num_timesteps

            # Forward propagation
            x_traj[:,:,timestep_num+1] = func_F(x_traj[:,:,timestep_num],func_g(v_traj[:,:,timestep_num]),dt);

            traj_cost = traj_cost + func_run_cost(x_traj[:,:,timestep_num]) + learning_rate * sample_u_traj[:,timestep_num]' * inv(ctrl_noise_covar) * (sample_u_traj[:,timestep_num] .- v_traj[:,:,timestep_num]);

            if(print_mppi)
                println("TN: $timestep_num, IN: $iteration, DU: $(mean(sum(abs.(du),dims=1)))" );
            end
        end

        traj_cost = traj_cost + func_term_cost(x_traj[:,:,end]);

        if(save_sampling)
            #save("-append", [sampling_filename '_v_traj.dat'],'v_traj');
            #save("-append", [sampling_filename '_x_traj.dat'],'x_traj');
            #save("-append", [sampling_filename '_traj_cost.dat'], 'traj_cost');
        end

        # Weight and du calculation
        w = func_comp_weights(traj_cost);
        du = reshape(sum(repeat(w, outer=[control_dim, 1, num_timesteps]) .* ctrl_noise, dims=2), (control_dim, num_timesteps));

        # Filter the output from forward propagation
        du = func_filter_du(du);

        sample_u_traj = sample_u_traj + du;
        iteration = iteration + 1;

        last_sample_u_traj = sample_u_traj;

    end

    if (real_traj_cost == true)
        # Loop through the dynamics again to recalcuate traj_cost
        rep_traj_cost = 0.;

        for timestep_num = 1:num_timesteps

          # Forward propagation
          real_x_traj[:,timestep_num+1] = func_F(real_x_traj[:,timestep_num],func_g(sample_u_traj[:,timestep_num]),dt);

          #println("Testing")
          #println("$(rep_traj_cost)")
          #println("$(func_run_cost(real_x_traj[:,timestep_num]))")
          #println("$(rep_traj_cost + func_run_cost(real_x_traj[:,timestep_num]))")
          #println("$(learning_rate)")
          #println("$(rep_traj_cost + func_run_cost(real_x_traj[:,timestep_num]) + learning_rate)")
          #println("$()")
          #println("$()")

          # addition error here means that func_run_cost should be broadcasted
          # for now tmp hack with the one index
          rep_traj_cost = rep_traj_cost + func_run_cost(real_x_traj[:,timestep_num])[1] + learning_rate * sample_u_traj[:,timestep_num]' * inv(ctrl_noise_covar) * (last_sample_u_traj[:,timestep_num] - sample_u_traj[:,timestep_num]);

        end

        rep_traj_cost = rep_traj_cost + func_term_cost(real_x_traj[:,end])[1];
    else
        # normalize weights, in case they are not normalized
        normalized_w = w / sum(w);

        # Compute the representative trajectory cost of what actually happens
        # another way to think about this is weighted average of sample trajectory costs
        rep_traj_cost = sum(normalized_w .* traj_cost);
    end

    return sample_u_traj, rep_traj_cost

end