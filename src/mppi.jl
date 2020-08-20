# TODO: loose end with plotting during running and saving samples

import Parameters: @with_kw, @unpack
import Statistics: mean
import CUDA

@with_kw struct MppiParams
    num_samples::Int64
    learning_rate::Float32
    dt::Float32
    init_state::Array{Float32,1}
    init_ctrl_seq::Array{Float32,2}
    ctrl_noise_covar::Array{Float32,2}
    per_ctrl_based_ctrl_noise::Float32
    real_traj_cost::Bool = false
    plot_traj::Bool = false
    print_mppi::Bool = false
    save_sampling::Bool = false
    sampling_filename::String = "sampling_filename"
    func_control_update_converged::Function = control_update_converged
    func_comp_weights::Function = comp_weights
    func_term_cost::Function
    func_run_cost::Function
    func_g::Function = g
    func_F::Function
    func_state_transform::Function = state_transform
    func_filter_du::Function = filter_du
end

function mppi(mppi_params)

    # Unpack mppi_params
    @unpack num_samples, learning_rate, dt, init_state, init_ctrl_seq,
    ctrl_noise_covar, per_ctrl_based_ctrl_noise, real_traj_cost,
    plot_traj, print_mppi, save_sampling,
    sampling_filename, func_control_update_converged,
    func_comp_weights, func_term_cost, func_run_cost,
    func_g, func_F, func_state_transform,
    func_filter_du = mppi_params

    # time stuff
    horizon = size(init_ctrl_seq, 2);

    # sample state stuff
    sample_init_state = func_state_transform(init_state);
    sample_state_dim = size(sample_init_state,1);

    # state trajectories
    real_x_traj = CUDA.zeros(sample_state_dim, horizon + 1);
    real_x_traj[:,1] = sample_init_state;
    x_traj = CUDA.zeros(sample_state_dim, num_samples, horizon + 1);
    x_traj[:,:,1] = repeat(sample_init_state,outer=[1, num_samples]);

    # control stuff
    control_dim = size(init_ctrl_seq, 1);
    ctrl_noise_covar = CUDA.CuArray(ctrl_noise_covar)
    du = typemax(Float64) * CUDA.ones(control_dim, horizon);
    traj_cost = CUDA.zeros(num_samples);
    w = CUDA.zeros(num_samples);

    # control sequence
    sample_u_traj = CUDA.CuArray(init_ctrl_seq);
    last_sample_u_traj = sample_u_traj;

    # sampled control trajectories
    v_traj = CUDA.zeros(control_dim, num_samples, horizon);

    # Begin mppi
    iteration = 1;
    while(func_control_update_converged(du, iteration) == false)

        # Noise generation
        flat_distribution = CUDA.randn(control_dim, num_samples * horizon);
        ctrl_noise_flat = ctrl_noise_covar * flat_distribution;
        ctrl_noise = reshape(ctrl_noise_flat, (control_dim, num_samples, horizon));

        # Compute sampled control trajectories
        ctrl_based_ctrl_noise_samples = floor(Int, per_ctrl_based_ctrl_noise * num_samples);
        if (ctrl_based_ctrl_noise_samples == 0)
            v_traj = ctrl_noise;
        elseif (ctrl_based_ctrl_noise_samples == num_samples)
            v_traj = repeat(reshape(sample_u_traj, (control_dim, 1, horizon)), outer=[1, num_samples, 1]) + ctrl_noise;
        else
            v_traj[:,1:ctrl_based_ctrl_noise_samples,:] = repeat(reshape(sample_u_traj, (control_dim, 1, horizon)), outer=[1, ctrl_based_ctrl_noise_samples, 1]) + ctrl_noise[:,1:ctrl_based_ctrl_noise_samples,:];
            v_traj[:,ctrl_based_ctrl_noise_samples+1:end,:] = ctrl_noise[:,ctrl_based_ctrl_noise_samples+1:end,:];
        end

        traj_cost = CUDA.zeros(1, num_samples);

        for timestep_num = 1:horizon

            # Forward propagation
            x_traj[:,:,timestep_num+1] = func_F(x_traj[:,:,timestep_num],func_g(v_traj[:,:,timestep_num]),dt);

            traj_cost = traj_cost + func_run_cost(x_traj[:,:,timestep_num]) + learning_rate * sample_u_traj[:,timestep_num]' * inv(ctrl_noise_covar) * (sample_u_traj[:,timestep_num] .- v_traj[:,:,timestep_num]);

            if(print_mppi)
                println("TN: $timestep_num, IN: $iteration, DU: $(mean(sum(abs.(du),dims=1)))" );
            end
        end

        traj_cost = traj_cost + func_term_cost(x_traj[:,:,end]);

        # Weight and du calculation
        w = func_comp_weights(traj_cost);
        du = reshape(sum(repeat(w, outer=[control_dim, 1, horizon]) .* ctrl_noise, dims=2), (control_dim, horizon));

        # Filter the output from forward propagation
        du = func_filter_du(du);

        sample_u_traj = sample_u_traj + du;
        iteration = iteration + 1;

        last_sample_u_traj = sample_u_traj;

    end

    if (real_traj_cost == true)
        # Loop through the dynamics again to recalcuate traj_cost
        rep_traj_cost = CUDA.zeros(1);

        for timestep_num = 1:horizon

          # Forward propagation
          real_x_traj[:,timestep_num+1] = func_F(real_x_traj[:,timestep_num],func_g(sample_u_traj[:,timestep_num]),dt);

          rep_traj_cost = rep_traj_cost + func_run_cost(real_x_traj[:,timestep_num]) .+ learning_rate * sample_u_traj[:,timestep_num]' * inv(ctrl_noise_covar) * (last_sample_u_traj[:,timestep_num] - sample_u_traj[:,timestep_num]);

        end

        rep_traj_cost = rep_traj_cost + func_term_cost(real_x_traj[:,end]);
    else
        # normalize weights, in case they are not normalized
        normalized_w = w / sum(w);

        # Compute the representative trajectory cost of what actually happens
        # another way to think about this is weighted average of sample trajectory costs
        rep_traj_cost = CUDA.CuArray(sum(normalized_w .* traj_cost));
    end

    return Array(sample_u_traj), Array(rep_traj_cost)

end

function comp_weights(traj_cost)
    λ = 0.01
    val, ind = findmin(traj_cost)
    w = similar(traj_cost)
    @. w = exp(-1 / λ * (traj_cost - val))
    return w / sum(w)
end

function control_update_converged(du, iteration)
    max_iteration = 1;
    if iteration > max_iteration
        return true
    end
    return false
end

function filter_du(du)
    return du
end

function g(u)
    clamped_u = u
end

function state_transform(x)
    return x
end
