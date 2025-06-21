from app.pizza_assistant import PizzaAssistant
from app.pizza_assistant_cli import PizzaAssistantCLI

def main() -> None:
    """
    Entry point for the program.
    """
    assistant = PizzaAssistant()
    cli = PizzaAssistantCLI(assistant=assistant)
    cli.run()

if __name__ == "__main__":
    main()
