from torch.optim import Adam
from modified_tensorboard import TensorboardCallback
import os
from pacman_env import PacmanEnv
from MADDPG import MADDPGAgent
from buffer import MultiAgentReplayBuffer
import numpy as np


if __name__ == "__main__":
    
    env_not_render = PacmanEnv(render_mode=None)
    env_render = PacmanEnv(render_mode="human")


    model_path = "./models/maddpg"
    log_path = "./logs/maddpg"
    chkpt_dir = "./tmp/maddpg/"

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(chkpt_dir, exist_ok=True)

    
    env = env_render
    n_agents = 2  
    actor_dims = [env.observation_space(agent).shape[0] for agent in env.possible_agents]  # Actor input dimensions
    critic_dims = sum(actor_dims)  # Critic input dimensions "joint state"
    n_actions = 5  

    
    model_final_path = os.path.join(model_path, "MADDPG_model.pth")
    if not os.path.exists(model_final_path):
        print("Training new MADDPG model...")

        
        model = MADDPGAgent(
            actor_dims,  # Actor; Takes individual states
            critic_dims,  # Critic; Takes joint states and joint actions 
            n_agents,
            n_actions,
            fc1=64,
            fc2=64,
            alpha=0.01,
            beta=0.01,
            chkpt_dir=chkpt_dir,
            tensorboard_log=log_path,
            device='cuda'
        )

        
        memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, n_agents, n_actions, batch_size=1024)

        total_episodes = 5000
        PRINT_INTERVAL = 100
        total_steps = 0
        best_score = -np.inf
        score_history = []

        for episode in range(total_episodes):
            observations, _ = env.reset()  
            done = [False] * n_agents  
            episode_score = 0

            while not any(done):
                # Chooses actions for each agent based on their individual states
                # ##### Actor
                actions = model.choose_action(observation=observations)  
                actions = {agent: actions[i] for i, agent in enumerate(env.possible_agents)}  
                print(f"Actions: {actions}")
                for agent in env.possible_agents:
                    print(f"{agent} action space: {env.action_space(agent)}")
                obs_, rewards, done, _ = env.step(actions)

                # Prepare joint state and next joint state for the critic
                state = np.concatenate([observations[agent] for agent in env.possible_agents])  # Joint state
                state_ = np.concatenate([obs_[agent] for agent in env.possible_agents])  # Next joint state

                ###store in replay buffer
                memory.store_transition(observations, state, actions, rewards, obs_, state_, done)  

                if total_steps > memory.batch_size:
                    #learning step; Update actor and critic networks
                    actor_loss, critic_loss = model.learn(memory)
                    print(f"Actor Loss: {actor_loss}, Critic Loss: {critic_loss}")

                
                observations = obs_
                episode_score += sum(rewards.values())
                total_steps += 1

            
            score_history.append(episode_score)
            avg_score = np.mean(score_history[-100:])
            
            if avg_score > best_score:
                best_score = avg_score
                model.save_checkpoint(path=model_final_path)

            if episode % PRINT_INTERVAL == 0:
                print(f"Episode {episode}, Average Score: {avg_score:.2f}")

    else:
        print("Loading pre-trained MADDPG model...")
        #load the pre-trained model
        model = MADDPGAgent.load_checkpoint(model_final_path)

        #run for evaluation
        episodes = 10
        for ep in range(episodes):
            observations, _ = env.reset()
            done = [False] * n_agents
            episode_score = 0

            while not any(done):
                
                actions = model.choose_action(observation=observations, explore=False)  
                obs_, rewards, done, _ = env.step(actions)
                episode_score += sum(rewards.values())
                observations = obs_

            print(f"Episode {ep + 1} - Total Score: {episode_score}")

    env.close()
