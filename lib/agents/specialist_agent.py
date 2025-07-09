import modal
from lib.agents.agent import Agent

class SpecialistAgent(Agent):
    """
    An Agent that runs two fine-tuned LLMs remotely on Modal for price prediction.
    """

    name = "Specialist Agent"
    color = Agent.RED

    def __init__(self):
        """
        Set up this Agent by creating an instance of the modal class
        """
        self.log("Specialist Agent is initializing - connecting to modal")
        self.llamapricer = modal.Function.from_name("pricer-service", "price_llama")
        self.gptpricer = modal.Function.from_name("pricer-service", "price_gpt")
        self.log("Specialist Agent is ready")

    def price(self, description: str, model: str = "gpt") -> float:
        """
        Make a remote call to return the price estimate using the selected model.
        model: "gpt" or "llama"
        """
        if model == "gpt":
            self.log("Specialist Agent is calling Model GPT")
            result = self.gptpricer.remote(description)
        elif model == "llama":
            self.log("Specialist Agent is calling Model llama")
            result = self.llamapricer.remote(description)
        else:
            raise ValueError("Unknown model type. Use 'gpt' or 'llama'.")
        self.log(f"Specialist Agent completed - predicting ${result:.2f}")
        return result
