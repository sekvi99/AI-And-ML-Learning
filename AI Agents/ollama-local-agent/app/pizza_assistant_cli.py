from typing import List, Any
from pydantic import BaseModel
from .pizza_assistant import PizzaAssistant

class PizzaAssistantCLI(BaseModel):
    """
    Command-line interface for interacting with the PizzaAssistant.
    """
    assistant: PizzaAssistant

    def run(self) -> None:
        """
        Runs the CLI loop, accepting user questions until 'exit' is entered.
        """
        while True:
            print("\n\n--------------")
            question = input("Enter your question about the pizza restaurant (or 'exit' to quit): ")
            if question.lower() == 'exit':
                print("Exiting the program.")
                break

            # In a real application, you would fetch relevant reviews here
            reviews: List[str] = []

            result = self.assistant.answer_question(question, reviews)
            print(result)