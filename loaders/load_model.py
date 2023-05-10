import pickle
import os

def load_model(model_class, folder):
    
    with open(os.path.join(folder, 'model_parameters.pkl'), 'rb') as f:
        params = pickle.load(f)

    model = model_class(*params)

    model.load_weights(os.path.join(folder, 'weights'))

    return model