# LLaMA Server

LLaMA Server combines the power of [LLaMA C++](https://github.com/ggerganov/llama.cpp) with the beauty of [Chatbot UI](https://github.com/mckaywrigley/chatbot-ui).

🦙LLaMA C++ ➕ 🤖Chatbot UI ➕ 🔗LLaMA Server 🟰 😊

**UPDATE**: Now supports Windows through [pyllamacpp](https://github.com/nomic-ai/pyllamacpp)!

**UPDATE**: Now supports streaming!

## Demo
- Streaming

https://user-images.githubusercontent.com/10931178/229980159-61546fa6-2985-4cdc-8230-5dcb6a69c559.mov

- Non-streaming

https://user-images.githubusercontent.com/10931178/229408428-5b6ef72d-28d0-427f-ae83-e23972e2dcff.mov


## Setup

- Get your favorite LLaMA models by
  - Download from [🤗Hugging Face](https://huggingface.co/models?sort=downloads&search=ggml);
  - Or follow instructions at [LLaMA C++](https://github.com/ggerganov/llama.cpp);
  - Make sure models are converted and quantized;
  - Copy models to the folders with correct name, e.g., `models/7B/ggml-model-q4_0.bin`;

- Set up python environment:
```bash
conda create -n llama python=3.9
conda activate llama
python -m pip install -r requirements.txt
```

- Start LLaMA Server:
```bash
export LLAMA_SERVER_HOME=$(git rev-parse --show-toplevel)  # path to this repo
python -m llama_server
```

- Check out [my fork](https://github.com/nuance1979/chatbot-ui) of Chatbot UI and start the app;
```bash
git clone https://github.com/nuance1979/chatbot-ui
cd chatbot-ui
git checkout llama
npm i
npm run dev
```
- Open the link http://localhost:3000 in your browser;
  - Click "OpenAI API Key" at the bottom left corner and enter your [OpenAI API Key](https://platform.openai.com/account/api-keys);
  - Or follow instructions at [Chatbot UI](https://github.com/mckaywrigley/chatbot-ui) to put your key into a `.env.local` file and restart;
  ```bash
  cp .env.local.example .env.local
  <edit .env.local to add your OPENAI_API_KEY>
  ```
- Enjoy!

## More

- Try a larger model if you have it:
```bash
export LLAMA_MODEL_ID=llama-13b  # llama-7b/llama-33b/llama-65b
python -m llama_server
```

- Try streaming mode by restarting Chatbot UI:
```bash
export LLAMA_STREAM_MODE=1  # 0 to disable streaming
npm run dev
```

## Limitations

- "Regenerate response" is currently not working;
- IMHO, the prompt/reverse-prompt machanism of LLaMA C++'s interactive mode needs an overhaul. I tried very hard to dance around it but the whole thing is still a hack.

## Fun facts

I am not fluent in JavaScript at all but I was able to make the changes in Chatbot UI by chatting with [ChatGPT](https://chat.openai.com); no more StackOverflow.
