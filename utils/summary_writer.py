import tensorflow as tf



class SummaryWriter(object):
    def __init__(self, logdir: str) -> None:
        self.writer = tf.summary.create_file_writer(logdir=logdir)

    def add_scalar(self, name: str, data: float, step: int, description=None):
        with self.writer.as_default():
            tf.summary.scalar(name=name, data=data, step=step, description=description)

    def add_text(self, name: str, data: str, step: int, description=None):
        with self.writer.as_default():
            tf.summary.text(name=name, data=data, step=step, description=description)

    def add_image(self, name: str, data, step=None, max_outputs=3, description=None):
        with self.writer.as_default():
            tf.summary.image(name=name, data=data, step=step, description=description)
