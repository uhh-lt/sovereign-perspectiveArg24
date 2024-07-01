# !!this class is entirely copied from the project Irina sent me
from hugchat import hugchat
from hugchat.login import Login
# from hugchat import hugchat
# from hugchat import login
import pandas as pd
import random


class HuggingChat:
    """
    API for accessing HuggingChat
    List of models: https://huggingface.co/chat
    Requires email and password from https://huggingface.co to use
    """

    def __init__(
        self,
        email: str,
        password: str,
        system_prompt: str = "",
        cookie_path_dir: str = "./cookies_snapshot",
        model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
    ):
        self.sign = Login(email, password)
        print('created login')
        cookies = self.sign.login()
        self.sign.saveCookiesToDir(cookie_path_dir)
        self.chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
        # List of models might change, should correspond to list on https://huggingface.co/chat/settings
        self.models = {
            "CohereForAI/c4ai-command-r-plus": 0,
            "meta-llama/Meta-Llama-3-70B-Instruct": 1,
            "google/gemma-2-27b-it": 2,
            "mistralai/Mixtral-8x7B-Instruct-v0.1": 3,
            "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": 4,
            "01-ai/Yi-1.5-34B-Chat": 5,
            "mistralai/Mistral-7B-Instruct-v0.2": 6,
            "microsoft/Phi-3-mini-4k-instruct": 7
        }
        self.system_prompt = system_prompt
        self.model = self.models[model]
        self.chatbot = hugchat.ChatBot(cookies=cookies.get_dict(), system_prompt=self.system_prompt)
        self.chatbot.switch_llm(self.model)
        self.chatbot.new_conversation(switch_to=True, system_prompt=self.system_prompt)

    def prompt(self, prompt: str) -> str:
        return str(self.chatbot.query(text=prompt))

    def delete_conversations(self) -> None:
        """
        Deletes all conversations in a user's profile
        """
        self.chatbot.delete_all_conversations()
        self.chatbot.new_conversation(switch_to=True,system_prompt=self.system_prompt)

    def switch_model(self, model: str) -> None:
        self.model = self.models[model]
        self.chatbot.switch_llm(self.model)
        self.chatbot.new_conversation(switch_to=True,system_prompt=self.system_prompt)

    def switch_system_prompt(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt
    
    def get_remote_llms(self) -> str:
        return self.chatbot.get_remote_llms()
    
    def get_active_model(self) -> str:
        return self.chatbot.active_model