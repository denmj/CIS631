class BaseModel(object):
    def __init__(self):
        self.model = None

    def save(self, checkpoint_path):
        if self.model is None:
            raise Exception("Model is not build yet")

    def build_model(self, input_shape, num_classes):
        raise NotImplemented

    def fit(self, x_train, y_train, batch_size, epochs):
        raise NotImplemented

    def evaluate(self, x_test, y_test):
        raise NotImplemented

