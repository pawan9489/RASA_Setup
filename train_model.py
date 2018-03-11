from rasa_nlu.converters import load_data
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer

training_data = load_data('data/iHCM_intents/navigation')
trainer = Trainer(RasaNLUConfig("configs/config_spacy.json"))
trainer.train(training_data)
model_directory = trainer.persist('projects/')  # Returns the directory the model is stored in
