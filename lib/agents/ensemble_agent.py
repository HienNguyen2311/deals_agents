from pathlib import Path
import pandas as pd
import joblib
from lib.agents.agent import Agent
from lib.agents.specialist_agent import SpecialistAgent
from lib.agents.frontier_agent import FrontierAgent

cwd = Path.cwd()
model_path = Path(cwd)/"output"/'models'/'ensemble_model.pkl'

class EnsembleAgent(Agent):

    name = "Ensemble Agent"
    color = Agent.YELLOW

    def __init__(self, collection):
        """
        Create an instance of Ensemble, by creating each of the models
        And loading the weights of the Ensemble
        """
        self.log("Initializing Ensemble Agent")
        self.specialist = SpecialistAgent()
        self.frontier = FrontierAgent(collection)
        self.model = joblib.load(str(model_path))
        self.log("Ensemble Agent is ready")

    def price(self, description: str) -> float:
        """
        Run this ensemble model
        Ask each of the models to price the product
        Then use the Linear Regression model to return the weighted price
        :param description: the description of a product
        :return: an estimate of its price
        """
        self.log("Running Ensemble Agent - collaborating with specialist and frontier agents")
        specialist_gpt = self.specialist.price(description, "gpt")
        specialist_llama = self.specialist.price(description, "llama")
        frontier = self.frontier.price(description)
        X = pd.DataFrame({
            'GPT Specialist': [specialist_gpt],
            'Llama Specialist': [specialist_llama],
            'Frontier': [frontier],
            'Min': [min(specialist_gpt, specialist_llama, frontier)],
            'Max': [max(specialist_gpt, specialist_llama, frontier)],
        })
        y = max(0, self.model.predict(X)[0])
        self.log(f"Ensemble Agent complete - returning ${y:.2f}")
        return y