from .config import EPISODES, TARGET_UPDATE, STEPS_PER_EPISODE
from .ddqn_agent import DDQNAgent
from .spectrum_env import SpectrumEnv


def main():
    env = SpectrumEnv()
    agent = DDQNAgent(state_dim=env.state_dim)

    for episode in range(EPISODES):
        state = env.reset()
        episode_reward = 0.0

        for step in range(STEPS_PER_EPISODE):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, float(done))
            agent.train_step()

            if agent.step_count % TARGET_UPDATE == 0:
                agent.update_target()

            episode_reward += reward
            state = next_state

            if done:
                break

        print(
            f"Episode {episode + 1}/{EPISODES} | Reward: {episode_reward:.2f} "
            f"| epsilon: {agent.epsilon:.3f}"
        )

    agent.update_target()
    agent.online.save("ddqn_spectrum_agent.h5")


if __name__ == "__main__":
    main()
