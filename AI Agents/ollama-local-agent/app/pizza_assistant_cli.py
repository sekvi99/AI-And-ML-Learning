from typing import List, Any
from pydantic import BaseModel
from .pizza_assistant import PizzaAssistant
from .vector import VectorStoreManager

class PizzaAssistantCLI(BaseModel):
    """
    Command-line interface for interacting with the PizzaAssistant.
    """
    assistant: PizzaAssistant
    vector_manager: VectorStoreManager

    def run(self) -> None:
        """
        Runs the CLI loop, accepting user questions until 'exit' is entered.
        """
        retriever = self.vector_manager.get_retriever()
        while True:
            print("\n\n--------------")
            question = input("Enter your question about the pizza restaurant (or 'exit' to quit): ")
            if question.lower() == 'exit':
                print("Exiting the program.")
                break

            # Retrieve relevant reviews using the vector store retriever
            docs = retriever.get_relevant_documents(question)
            reviews: List[str] = [doc.page_content for doc in docs]

            result = self.assistant.answer_question(question, reviews)
            print(result)