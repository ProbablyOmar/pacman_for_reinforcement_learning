In this project we build a DQN model to teach the pacman how to eat the pellets and escape the ghosts.
to start training the model run the file called pacman_env.py.

inside this file you will see a line : model_path = "./models/DQN_full_maze_map_baseline_obs_updated_norm_work_station1" this specifies the directory to save the model.
and there is another line:  model.learn(total_timesteps = time_steps , progress_bar=True , reset_num_timesteps = False , tb_log_name = "DQN_full_maze_map_baseline_obs_updated_norm_work_station1") this starts fitting the model
and specifies the tensorboard log folder name. 
