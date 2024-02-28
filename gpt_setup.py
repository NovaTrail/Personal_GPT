import os
import warnings

from transformers import pipeline
import ipywidgets as widgets
from IPython.display import display

warnings.filterwarnings(
    "ignore", category=UserWarning
) 

# USER PARAMETERS
HF_MODEL_NAME = (
    "HuggingFaceH4/zephyr-7b-beta"  # Your chosen converstation-pipeline model
)
SYSTEM_PROMPT = """You are a friendly chatbot that adheres to word limits."""  

class ChatBotGPT:
    """
    A  conversational agent using a Hugging Face transformer model designed to run on google colab.

    Attributes:
        save_path (str): Path to save or load the model.
        device (str): Device configuration (e.g., 'auto', 'cpu', 'gpu').
        conv_history (str): Stores the conversation history.
        sys_prompt (str): Initial system prompt for conversations.
        chat_count (int): Count of chat interactions.
    """

    def __init__(self, save_path, device="auto"):
        """
        Initializes the ChatBotGPT instance.

        Args:
            save_path (str): Path where the model is saved or to be saved.
            device (str): Device setting for running the model ('auto', 'cpu', 'gpu').
        """
        self.save_path = save_path
        self.device = device
        self.conv_history = ""
        self.sys_prompt = SYSTEM_PROMPT
        self.chat_count = 0

        if os.path.exists(self.save_path):
            print("Using the saved model.")
            self.load_model()
        else:
            print("Downloading the model from Hugging Face.")
            self.download_and_save_model()

    def load_model(self):
        """
        Loads the conversation model from the specified save path.
        """
        try:
            self.pipe = pipeline(
                "conversational",
                model=self.save_path,
                device_map=self.device,
            )
        except Exception as e:
            print(f"Failed to load model from {self.save_path}. Error: {e}")

    def reset(self):
        """
        Resets the conversation history and count.
        """
        self.conv_history = ""
        self.chat_count = 0

    def load_prompt(self):
        if os.path.exists("prompt.txt"):
            # print("found the text file")
            with open("prompt.txt", "r") as file:
                # Read the contents of the file
                file_contents = file.read()
            self.sys_prompt += str(file_contents)
        else:
            # print("text file not found")
            pass

    def download_and_save_model(self):
        """
        Downloads and saves the model specified by HF_MODEL_NAME.
        """
        try:
            presave_pipe = pipeline(
                "conversational",
                model=HF_MODEL_NAME,
                device_map=self.device,
                model_kwargs={"load_in_4bit": True},
            )
            presave_pipe.save_pretrained(self.save_path)
            self.load_model()  # Load the model after downloading
        except Exception as e:
            print(f"Failed to download the model from Hugging Face. Error: {e}")

    def chat(self, prompt, max_words=50):
        """
        Generates a response to the given prompt within the specified word limit.

        Args:
            prompt (str): The user input prompt to respond to.
            max_words (int): The maximum number of words allowed in the response.

        Returns:
            str: The generated response from the chatbot.
        """
        if self.chat_count == 0:
            messages = [
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": self.sys_prompt
                    + " The conversation so far: "
                    + self.conv_history,
                },
                {
                    "role": "user",
                    "content": prompt
                    + f" Limit your answer to maximum {max_words} words.",
                },
            ]

        output = self.pipe(
            messages,
            do_sample=True,
            max_new_tokens=max_words,
            pad_token_id=self.pipe.tokenizer.eos_token_id,
        )
        self.conv_history += str(output)
        output = output[2]["content"]

        self.chat_count += 1
        return response

    def chat_box(self):
    
            self.reset()
    
            # Create text area input widget
            text_area_input = widgets.Textarea(
                value="",
                placeholder="Write text here...",
                description="Prompt Input:",
                disabled=False,
                layout=widgets.Layout(width="500px", height="auto"),
            )
    
            # Create button widget
            button = widgets.Button(
                description="Go",
                button_style="primary",  # or 'success', 'info'
                tooltip="Click me",
            )
    
            # Create a slider for setting the word limit
            # Create radio buttons for setting the word limit
            word_limit = widgets.RadioButtons(
                options=[100, 200, 500],
                value=50,  # Default value
                description="Word Limit:",
                disabled=False,
                layout=widgets.Layout(flex_flow="row"),  # Set layout as horizontal
            )
    
            # Create an output area for responses
            response_display = widgets.Output(
                layout={"border": "1px solid black", "width": "600px"}
            )
    
            # Handler function for button click event
            def handle_click(event):
                with response_display:
    
                    user_input = text_area_input.value
                    if not user_input:
                        return
                    user_input = user_input.strip()
                    print(f"**You said:**\n {user_input}")
                    _word_limit = int(word_limit.value)
                    print("\n")
                    # Simulate a response, replace with actual response logic as needed
                    print("**BotGPT🤖 says:**")
                    response = self.chat(user_input, max_words=_word_limit)
                    print(f" {response}\n")
    
            # Attach the event handler to the button
            button.on_click(handle_click)
    
            # Display widgets
            display(response_display, text_area_input, word_limit, button)
            response_display.clear_output()

