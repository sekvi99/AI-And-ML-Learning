from app.pizza_assistant import PizzaAssistant
from app.pizza_assistant_cli import PizzaAssistantCLI
from app.vector import VectorStoreManager

def main() -> None:
    """
    Entry point for the program.
    """
    assistant = PizzaAssistant()
    vector_manager = VectorStoreManager()
    cli = PizzaAssistantCLI(assistant=assistant, vector_manager=vector_manager)
    cli.run()

if __name__ == "__main__":
    main()
