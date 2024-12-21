if __name__ == "__main__":
    env_not_render = gym.make("pacman-v0", max_episode_steps = 10_000)
    env_render = gym.make("pacman-v0", max_episode_steps = 10_000 , render_mode = "human")
    
    model_path = "./models/dqn_baseline_cnn3"   #with max_useless_steps = 1000
    log_path = "./logs/fit"
   
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if not os.path.exists(model_path):  
        print("can't find path")
        os.makedirs(model_path)
        env = env_not_render
        obs , _ = env.reset()

        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=256),
        )

        model = DQN(
            "CnnPolicy" , 
            env , 
            learning_rate  = 0.0001 , 
            learning_starts  = 2000,
            batch_size= 32,   #32
            gamma = 0.97,
            #train_freq = (1, "episode"),
            gradient_steps = 4,
            target_update_interval=100,
            exploration_fraction=0.99,
            exploration_initial_eps=1,
            exploration_final_eps=0.1,

            policy_kwargs = policy_kwargs , 
            verbose = 1 , 
            tensorboard_log = log_path
        )

        time_steps = 400_000
        for i in range (100):
            model.learn(total_timesteps = time_steps , progress_bar=True , reset_num_timesteps = False , tb_log_name = "./cnn/dqn_baseline_cnn_alex_net")
            model.save(f"{model_path}/{(i+1)*time_steps}")

    elif os.path.exists(model_path):
        env = env_render
        obs , _ = env.reset()
        model_final_path = f"{model_path}/1200000.zip"
        model = DQN.load(model_final_path , env = env)
        print(model.policy)

        episodes = 10
        for ep in range(episodes):
            done = False
            while not done:
                action , next_state = model.predict(obs)
                obs, reward, terminated, truncated, info = env.step(int(action))
                print(env.game_score)
                done = terminated
        env.close()