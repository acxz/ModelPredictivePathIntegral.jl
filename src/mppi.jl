# TODO Array of Array makes more logical sense then a 2 dim array
# since we dealing with vectors and trajectories
# But which allows faster operation, I guess I just gotta benchmark it
# TODO: loose end with plotting during running and saving samples
# TODO: vectorize traj running cost
# TODO: GPU up

import Parameters: @with_kw, @unpack
import Statistics: mean

@with_kw struct MppiParams
    num_samples::Int64
    learning_rate::Float32
    dt::Float32
    init_state::Vector{Float32}
    init_ctrl_seq::Vector{Vector{Float32}}
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
    horizon = size(init_ctrl_seq, 1);

    # sample state stuff
    sample_init_state = func_state_transform(init_state);
    sample_state_dim = size(sample_init_state, 1);

    # state trajectories
    real_x_traj = zeros.(ones(Int64, horizon + 1) * sample_state_dim);
    real_x_traj[1] = sample_init_state;
    x_traj = [zeros.(ones(Int64, horizon + 1) * sample_state_dim) for i=1:num_samples]
    broadcast(x_traj) do traj
        traj[1] = init_state
    end

    # control stuff
    control_dim = size(init_ctrl_seq[1], 1);
    du = [typemax(Float32) * ones(control_dim) for i=1:horizon]
    traj_cost = zeros(num_samples);
    w = zeros(num_samples);

    # control sequence
    sample_u_traj = init_ctrl_seq;
    last_sample_u_traj = sample_u_traj;

    # sampled control trajectories
    v_traj = [zeros.(ones(Int64, horizon) * control_dim) for i=1:num_samples]

    # Begin mppi
    iteration = 1;
    while(func_control_update_converged(du, iteration) == false)

        # Noise generation
        ctrl_noise_distribution = [zeros.(ones(Int64, horizon) * control_dim) for i=1:num_samples]
        ctrl_noise_distribution = broadcast(ctrl_noise_distribution) do ctrl_noise_trajectory
            ctrl_noise_trajectory = broadcast(ctrl_noise_trajectory) do ctrl_noise
                ctrl_noise = ctrl_noise_covar * randn(control_dim)
            end
        end

        # Compute sampled control trajectories
        ctrl_based_ctrl_noise_samples = floor(Int, per_ctrl_based_ctrl_noise * num_samples);
        if (ctrl_based_ctrl_noise_samples == 0)
            v_traj = ctrl_noise_distribution;
        elseif (ctrl_based_ctrl_noise_samples == num_samples)
            v_traj = broadcast(v_traj) do traj
                traj = sample_u_traj
            end
            v_traj = v_traj + ctrl_noise_distribution
        else
            v_traj[1:ctrl_based_ctrl_noise_samples] = broadcast(v_traj[1:ctrl_based_ctrl_noise_samples]) do traj
                traj = sample_u_traj
            end
            v_traj = v_traj + ctrl_noise_distribution
        end

        traj_cost = zeros(num_samples);

        for timestep_num = 1:horizon

            # Forward propagation
            broadcast(x_traj, v_traj) do x_traj_sample, v_traj_sample
                x_traj_sample[timestep_num + 1] = func_F(x_traj_sample[timestep_num], func_g(v_traj_sample[timestep_num]), dt)
            end

            traj_cost = broadcast(traj_cost, x_traj, v_traj) do traj_cost_sample, x_traj_sample, v_traj_sample
                traj_cost_sample = traj_cost_sample + func_run_cost(x_traj_sample[timestep_num]) + learning_rate * sample_u_traj[timestep_num]' * inv(ctrl_noise_covar) * (sample_u_traj[timestep_num] - v_traj_sample[timestep_num]);
            end

            if(print_mppi)
                approx_du = broadcast(du) do du_sample
                    abs.(du_sample)
                end
                println("TN: $timestep_num, IN: $iteration, DU: $(mean(approx_du))" );
            end

        end

        traj_cost = broadcast(traj_cost, x_traj) do traj_cost_sample, x_traj_sample
            traj_cost_sample = traj_cost_sample + func_term_cost(x_traj_sample[end]);
        end

        # Weight and du calculation
        w = func_comp_weights(traj_cost);
        weighted_du = broadcast(w, ctrl_noise_distribution) do w_sample, ctrl_noise_traj
            broadcast(ctrl_noise_traj) do ctrl_noise
                w_sample * ctrl_noise
            end
        end
        du = sum(weighted_du)

        # Filter the output from forward propagation
        du = func_filter_du(du);

        sample_u_traj = sample_u_traj + du;
        iteration = iteration + 1;

        last_sample_u_traj = sample_u_traj;

    end

    if (real_traj_cost == true)
        # Loop through the dynamics again to recalcuate traj_cost
        rep_traj_cost = 0.;

        for timestep_num = 1:horizon

          # Forward propagation
          real_x_traj[timestep_num+1] = func_F(real_x_traj[timestep_num],func_g(sample_u_traj[timestep_num]),dt);

          # addition error here means that func_run_cost should be broadcasted
          # for now tmp hack with the one index
          rep_traj_cost = rep_traj_cost + func_run_cost(real_x_traj[timestep_num]) + learning_rate * sample_u_traj[timestep_num]' * inv(ctrl_noise_covar) * (last_sample_u_traj[timestep_num] - sample_u_traj[timestep_num]);

        end

        rep_traj_cost = rep_traj_cost + func_term_cost(real_x_traj[end]);

    else
        # normalize weights, in case they are not normalized
        normalized_w = w / sum(w);

        # Compute the representative trajectory cost of what actually happens
        # another way to think about this is weighted average of sample trajectory costs
        rep_traj_cost = sum(normalized_w .* traj_cost);
    end

    return sample_u_traj, rep_traj_cost

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
