if __name__ == "__main__":
    env = gym.make("pacman-v0", render_mode="human")

    model = DQN_model()
    EPISODES = 20_000

    EPSILON = 1
    EPSILON_DECAY = 0.99975
    MIN_EPSILON = 0.001 

    SHOW = True
    SHOW_EVERY = 50
    RENDER = False
    EPISODES_REWARDS = []
    MIN_REWARD = 10*120 - 50*5  # eaten half of the pellets before being eaten 5 times by ghosts

    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        model.tensorboard.step = episode

        episode_reward = 0

        curr_state , info = env.reset()
        done = False

        while not done :
            if np.random.random() > EPSILON :
                action = model.get_qs(curr_state)
                action = env.get_action(action)
            else:
                action = random.randint(0,4)
                action = env.get_action(action)

            new_state , reward , done = env.step(agent_direction = action , render = RENDER)
            episode_reward += reward

            model.update_replay_memory((curr_state , action , new_state , reward , done))
            model.train(done)

            if SHOW and not episode % SHOW_EVERY:
                render = True
            else:
                render = False

            curr_state = new_state

        EPISODES_REWARDS.append(episode_reward)

        if not episode % SHOW_EVERY:
            avg_reward = sum(EPISODES_REWARDS[-SHOW_EVERY:]) / len(EPISODES_REWARDS[-SHOW_EVERY:])
            min_reward = min(EPISODES_REWARDS[-SHOW_EVERY:])
            max_reward = max(EPISODES_REWARDS[-SHOW_EVERY:])
            model.tensorboard.update_stats(reward_avg=avg_reward, reward_min=min_reward, reward_max=max_reward, epsilon=EPSILON)

        if min_reward >= MIN_REWARD:
            model.model.save(f'models/{model.MODEL_NAME}__{max_reward:_>7.2f}max_{avg_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        if EPSILON > MIN_EPSILON:
            EPSILON *= EPSILON_DECAY
            EPSILON = max(MIN_EPSILON , EPSILON)




    




