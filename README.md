# LLaMA Server

[![PyPI version](https://img.shields.io/pypi/v/llama-server)](https://pypi.org/project/llama-server/) [![Unit test](https://github.com/nuance1979/llama-server/actions/workflows/test.yml/badge.svg?branch=main&&event=push)](https://github.com/nuance1979/llama-server/actions) [![GitHub stars](https://img.shields.io/github/stars/nuance1979/llama-server)](https://star-history.com/#nuance1979/llama-server&Date) [![GitHub license](https://img.shields.io/github/license/nuance1979/llama-server)](https://github.com/nuance1979/llama-server/blob/master/LICENSE)

LLaMA Server combines the power of [LLaMA C++](https://github.com/ggerganov/llama.cpp) (via [PyLLaMACpp](https://github.com/abdeladim-s/pyllamacpp)) with the beauty of [Chatbot UI](https://github.com/mckaywrigley/chatbot-ui).

🦙LLaMA C++ (via 🐍PyLLaMACpp) ➕ 🤖Chatbot UI ➕ 🔗LLaMA Server 🟰 😊

**UPDATE**: Greatly simplified implementation thanks to the [awesome Pythonic APIs](https://github.com/abdeladim-s/pyllamacpp#different-persona) of PyLLaMACpp 2.0.0!

**UPDATE**: Now supports better streaming through [PyLLaMACpp](https://github.com/abdeladim-s/pyllamacpp)!

**UPDATE**: Now supports streaming!

## Demo
- Better Streaming

https://user-images.githubusercontent.com/10931178/231539194-052f7c5f-c7a3-42b7-9f8b-142422e42a67.mov

- Streaming

https://user-images.githubusercontent.com/10931178/229980159-61546fa6-2985-4cdc-8230-5dcb6a69c559.mov

- Non-streaming

https://user-images.githubusercontent.com/10931178/229408428-5b6ef72d-28d0-427f-ae83-e23972e2dcff.mov


## Setup

- Get your favorite LLaMA models by
  - Download from [🤗Hugging Face](https://huggingface.co/models?sort=downloads&search=ggml);
  - Or follow instructions at [LLaMA C++](https://github.com/ggerganov/llama.cpp);
  - Make sure models are converted and quantized;

- Create a `models.yml` file to provide your `model_home` directory and add your favorite [South American camelids](https://en.wikipedia.org/wiki/Lama_(genus)), e.g.:
```yaml
model_home: /path/to/your/models
models:
  llama-7b:
    name: LLAMA-7B
    path: 7B/ggml-model-q4_0.bin  # relative to `model_home` or an absolute path
```
See [models.yml](https://github.com/nuance1979/llama-server/blob/main/models.yml) for an example.

- Set up python environment:
```bash
conda create -n llama python=3.9
conda activate llama
```

- Install LLaMA Server:
  - From PyPI:
  ```bash
  python -m pip install llama-server
  ```
  - Or from source:
  ```bash
  python -m pip install git+https://github.com/nuance1979/llama-server.git
  ```

- Start LLaMA Server with your `models.yml` file:
```bash
llama-server --models-yml models.yml --model-id llama-7b
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
llama-server --models-yml models.yml --model-id llama-13b  # or any `model_id` defined in `models.yml`
```

- Try non-streaming mode by restarting Chatbot UI:
```bash
export LLAMA_STREAM_MODE=0  # 1 to enable streaming
npm run dev
```

## Fun facts

I am not fluent in JavaScript at all but I was able to make the changes in Chatbot UI by chatting with [ChatGPT](https://chat.openai.com); no more StackOverflow.
