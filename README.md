# Personal GPT Setup

Streamline the creation of your personalized GPT conversational agent with this project, running on Google Colab and utilizing the Hugging Face Transformers library for human-like responses.

Requires a Google account with Google Colab access.

Quick start:
<a href="https://colab.research.google.com/github/NovaTrail/Personal_GPT/blob/master/convoGPT.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

  
## Features

- **Custom GPT Model Usage:** Choose any Hugging Face GPT model for your agent.
- **Google Colab Integration:** Benefit from Colab's GPU for quicker responses.
- **Interactive Chat Interface:** Engage in real-time via an IPython widget chat.
- **Conversation History:** Keeps track of the chat for context in responses.

## Instructions

1. **Open in Google Colab:** Open this repo's notebook in Colab:
- https://colab.research.google.com/github/NovaTrail/Personal_GPT/blob/master/convoGPT.ipynb
2. **Mount Google Drive:** Follow the notebook's instructions to mount your drive.
3. **Run Setup Cells:** Runtime -> run all, this will clone the repo and install dependencies.
4. **Initialize ChatBot:** interact with BotGPT using the chat box.

### Customization
- **Model Selection:** Change `HF_MODEL_NAME` to switch models.
- **Prompt.txt:** Edit for different bot contexts.

## Usage
With setup complete, use the chat interface in the notebook for conversations. Tweak the system prompt and model for your style.


## License
Open-sourced under MIT license. Refer to the LICENSE file.

## Acknowledgments

- Zephyr-7B-β a fine-tuned version of mistralai/Mistral-7B-v0.1. -- *Lewis Tunstall Et al.* Oct 2023 (https://arxiv.org/abs/2310.16944).
- Hugging Face for the Transformers library. https://huggingface.co/
- Google Colab for openly available GPU resources.
