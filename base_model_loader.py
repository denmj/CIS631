from tensorflow.keras.models import model_from_json


class BaseModelLoader(object):
    def __init__(self, json_model, model_weights):
        with open(json_model, 'r') as json_file:
            loaded_json_model = json_file.read()
            self.loaded_json_model = model_from_json(loaded_json_model)

        self.loaded_json_model.load_weights(model_weights)
        # Depreciated in tf 2.2 ?
        # self.loaded_json_model._make_predict_function()

    def predict(self, image):
        raise NotImplemented
