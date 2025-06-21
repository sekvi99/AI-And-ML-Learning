from typing import List, Any
from pydantic import BaseModel
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

class PizzaAssistant(BaseModel):
    """
    Handles the language model and prompt chain for answering pizza restaurant questions.
    """
    model_name: str = "llama3.2"
    temperature: float = 0.1

    def setup_chain(self) -> Any:
        """
        Initializes the language model and prompt chain.
        """
        model = OllamaLLM(model=self.model_name, temperature=self.temperature)
        template = (
            "You are a helpful assistant in answering questions about a pizza restaurant. \n\n"
            "Here are some relevant reviews: {reviews}\n"
            "Answer the following question: {question}\n"
        )
        prompt = ChatPromptTemplate.from_template(template)
        return prompt | model

    def answer_question(self, question: str, reviews: List[str]) -> Any:
        """
        Answers a question using the model chain.
        """
        chain = self.setup_chain()
        return chain.invoke({
            "reviews": reviews,
            "question": question
        })