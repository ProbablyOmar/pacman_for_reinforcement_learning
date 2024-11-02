from keras.callbacks import TensorBoard
import tensorflow as tf

class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1

    def update_stats(self, **stats):
        self._write_logs(stats, self.step)
