from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, game , verbose=0):
        super().__init__(verbose)
        self.game = game

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = self.game.num_pellets_last
        self.logger.record("num_pellets_left", value)
        return True