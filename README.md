# OnPrem.LLM


<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

> A tool for running large language models on-premises using non-public
> data

**[OnPrem.LLM](https://github.com/amaiya/onprem)** is a simple Python
package that makes it easier to run large language models (LLMs) on your
own machines using non-public data (possibly behind corporate
firewalls). Inspired largely by the
[privateGPT](https://github.com/imartinez/privateGPT) GitHub repo,
**OnPrem.LLM** is intended to help integrate local LLMs into practical
applications.

The full documentation is [here](https://amaiya.github.io/onprem/).

A Google Colab demo of installing and using **OnPrem.LLM** is
[here](https://colab.research.google.com/drive/1LVeacsQ9dmE1BVzwR3eTLukpeRIMmUqi?usp=sharing).

## Install

Once you have [installed
PyTorch](https://pytorch.org/get-started/locally/) and [installed
llama-cpp-python](https://python.langchain.com/docs/integrations/llms/llamacpp#installation),
you can install **OnPrem.LLM** with:

``` sh
pip install onprem
```

For fast GPU-accelerated inference, see [additional instructions
below](https://amaiya.github.io/onprem/#speeding-up-inference-using-a-gpu).
See [the FAQ](https://amaiya.github.io/onprem/#faq), if you experience
issues with
[llama-cpp-python](https://pypi.org/project/llama-cpp-python/)
installation.

**Note:** The `pip install onprem` command will install PyTorch and
llama-cpp-python automatically if not already installed, but we
recommend visting the links above to install these packages in a way
that is optimized for your system (e.g., with GPU support).

## How to Use

### Setup

``` python
from onprem import LLM

llm = LLM()
```

By default, a 7B-parameter model is downloaded and used. If
`use_larger=True`, a 13B-parameter is used. You can also supply the URL
to an LLM of your choosing to
[`LLM`](https://amaiya.github.io/onprem/core.html#llm) (see the [code
generation section
below](https://amaiya.github.io/onprem/#text-to-code-generation) for an
example). Any extra parameters supplied to
[`LLM`](https://amaiya.github.io/onprem/core.html#llm) are forwarded
directly to `llama-cpp-python`. As of v0.0.20, **OnPrem.LLM** supports
the newer GGUF format.

### Send Prompts to the LLM to Solve Problems

This is an example of few-shot prompting, where we provide an example of
what we want the LLM to do.

``` python
prompt = """Extract the names of people in the supplied sentences. Here is an example:
Sentence: James Gandolfini and Paul Newman were great actors.
People:
James Gandolfini, Paul Newman
Sentence:
I like Cillian Murphy's acting. Florence Pugh is great, too.
People:"""

saved_output = llm.prompt(prompt)
```


    Cillian Murphy, Florence Pugh

Additional prompt examples are [shown
here](https://amaiya.github.io/onprem/examples.html).

### Talk to Your Documents

Answers are generated from the content of your documents (i.e.,
[retrieval augmented generation](https://arxiv.org/abs/2005.11401) or
RAG). Here, we will supply `use_larger=True` to use the larger default
model better suited to this use case in addition to using [GPU
offloading](https://amaiya.github.io/onprem/#speeding-up-inference-using-a-gpu)
to speed up answer generation. However, the smaller Zephyr-7B model may
perform even better, responds faster, and is used in our [example
notebook](https://amaiya.github.io/onprem/examples_rag.html).

``` python
from onprem import LLM

llm = LLM(use_larger=True, n_gpu_layers=35)
```

#### Step 1: Ingest the Documents into a Vector Database

``` python
llm.ingest("./sample_data")
```

    Creating new vectorstore at /home/amaiya/onprem_data/vectordb
    Loading documents from ./sample_data
    Loaded 12 new documents from ./sample_data
    Split into 153 chunks of text (max. 500 chars each)
    Creating embeddings. May take some minutes...
    Ingestion complete! You can now query your documents using the LLM.ask method

    Loading new documents: 100%|██████████████████████| 3/3 [00:00<00:00, 25.52it/s]

#### Step 2: Answer Questions About the Documents

``` python
question = """What is  ktrain?"""
result = llm.ask(question)
```

     ktrain is a low-code platform designed to facilitate the full machine learning workflow, from preprocessing inputs to training, tuning, troubleshooting, and applying models. It focuses on automating other aspects of the ML workflow in order to augment and complement human engineers rather than replacing them. Inspired by fastai and ludwig, ktrain is intended to democratize machine learning for beginners and domain experts with minimal programming or data science experience.

The sources used by the model to generate the answer are stored in
`result['source_documents']`:

``` python
print("\nSources:\n")
for i, document in enumerate(result["source_documents"]):
    print(f"\n{i+1}.> " + document.metadata["source"] + ":")
    print(document.page_content)
```


    Sources:


    1.> ./sample_data/ktrain_paper.pdf:
    lection (He et al., 2019). By contrast, ktrain places less emphasis on this aspect of au-
    tomation and instead focuses on either partially or fully automating other aspects of the
    machine learning (ML) workﬂow. For these reasons, ktrain is less of a traditional Au-
    2

    2.> ./sample_data/ktrain_paper.pdf:
    possible, ktrain automates (either algorithmically or through setting well-performing de-
    faults), but also allows users to make choices that best ﬁt their unique application require-
    ments. In this way, ktrain uses automation to augment and complement human engineers
    rather than attempting to entirely replace them. In doing so, the strengths of both are
    better exploited. Following inspiration from a blog post1 by Rachel Thomas of fast.ai

    3.> ./sample_data/ktrain_paper.pdf:
    with custom models and data formats, as well.
    Inspired by other low-code (and no-
    code) open-source ML libraries such as fastai (Howard and Gugger, 2020) and ludwig
    (Molino et al., 2019), ktrain is intended to help further democratize machine learning by
    enabling beginners and domain experts with minimal programming or data science experi-
    4. http://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups
    6

    4.> ./sample_data/ktrain_paper.pdf:
    ktrain: A Low-Code Library for Augmented Machine Learning
    toML platform and more of what might be called a “low-code” ML platform. Through
    automation or semi-automation, ktrain facilitates the full machine learning workﬂow from
    curating and preprocessing inputs (i.e., ground-truth-labeled training data) to training,
    tuning, troubleshooting, and applying models. In this way, ktrain is well-suited for domain
    experts who may have less experience with machine learning and software coding. Where

### Guided Prompts

You can use **OnPrem.LLM** with the
[Guidance](https://github.com/guidance-ai/guidance) package to guide the
LLM to generate outputs based on your conditions and constraints. We’ll
show a couple of examples here, but see [our documentation on guided
prompts](https://amaiya.github.io/onprem/examples_guided_prompts.html)
for more information.

#### Structured Outputs with [`onprem.guider.Guider`](https://amaiya.github.io/onprem/guider.html#guider)

Here, we’ll use a
[`Guider`](https://amaiya.github.io/onprem/guider.html#guider)instance
to generate fictional D&D-type characters that conform to the precise
structure we want (i.e., JSON):

``` python
from onprem.guider import Guider
guider = Guider(llm)
```

``` python
# create the Guider instance
from onprem.guider import Guider
from guidance import gen, select
guider = Guider(llm)

# this is a function that generates a Guidance prompt that will be fed to Guider
sample_weapons = ["sword", "axe", "mace", "spear", "bow", "crossbow"]
sample_armour = ["leather", "chainmail", "plate"]
def generate_character_prompt(
    character_one_liner,
    weapons: list[str] = sample_weapons,
    armour: list[str] = sample_armour,
    n_items: int = 3
):
    prompt = ''
    prompt += "{"
    prompt += f'"description" : "{character_one_liner}",'
    prompt += '"name" : "' + gen(name="character_name", stop='"') + '",'
    prompt += '"age" : ' + gen(name="age", regex="[0-9]+") + ','
    prompt += '"armour" : "' + select(armour, name="armour") + '",'
    prompt += '"weapon" : "' + select(weapons, name="weapon") + '",'
    prompt += '"class" : "' + gen(name="character_class", stop='"') + '",'
    prompt += '"mantra" : "' + gen(name="mantra", stop='"') + '",'
    prompt += '"strength" : ' + gen(name="age", regex="[0-9]+") + ','
    prompt += '"quest_items" : [ '
    for i in range(n_items):
        prompt += '"' + gen(name="items", list_append=True, stop='"') + '"'  
        if i < n_items - 1:
            prompt += ','
    prompt += "]"
    prompt += "}"
    return prompt
```

``` python
# feed prompt to Guider and extract JSON
import json
d = guider.prompt(generate_character_prompt("A quick and nimble fighter"), echo=False)
print('Generated JSON:')
print(json.dumps(d, indent=4))
```

    Generated JSON:
    {
        "items": [
            "Quest Item 3",
            "Quest Item 2",
            "Quest Item 1"
        ],
        "age": "10",
        "mantra": "I am the blade of justice.",
        "character_class": "fighter",
        "weapon": "sword",
        "armour": "leather",
        "character_name": "Katana"
    }

#### Using Regular Expressions to Control LLM Generation

``` python
prompt = f"""Question: Luke has ten balls. He gives three to his brother. How many balls does he have left?
Answer: """ + gen(name='answer', regex='\d+')

guider.prompt(prompt, echo=False)
```

    {'answer': '7'}

``` python
prompt = '19, 18,' + gen(name='output', max_tokens=50, stop_regex='[^\d]7[^\d]')
guider.prompt(prompt)
```

<pre style='margin: 0px; padding: 0px; padding-left: 8px; margin-left: -8px; border-radius: 0px; border-left: 1px solid rgba(127, 127, 127, 0.2); white-space: pre-wrap; font-family: ColfaxAI, Arial; font-size: 15px; line-height: 23px;'>19, 18<span style='background-color: rgba(0, 165, 0, 0.15); border-radius: 3px;' title='0.0'>,</span><span style='background-color: rgba(0, 165, 0, 0.15); border-radius: 3px;' title='0.0'> 1</span><span style='background-color: rgba(0, 165, 0, 0.15); border-radius: 3px;' title='0.0'>7</span><span style='background-color: rgba(0, 165, 0, 0.15); border-radius: 3px;' title='0.0'>,</span><span style='background-color: rgba(0, 165, 0, 0.15); border-radius: 3px;' title='0.0'> 1</span><span style='background-color: rgba(0, 165, 0, 0.15); border-radius: 3px;' title='0.0'>6</span><span style='background-color: rgba(0, 165, 0, 0.15); border-radius: 3px;' title='0.0'>,</span><span style='background-color: rgba(0, 165, 0, 0.15); border-radius: 3px;' title='0.0'> 1</span><span style='background-color: rgba(0, 165, 0, 0.15); border-radius: 3px;' title='0.0'>5</span><span style='background-color: rgba(0, 165, 0, 0.15); border-radius: 3px;' title='0.0'>,</span><span style='background-color: rgba(0, 165, 0, 0.15); border-radius: 3px;' title='0.0'> 1</span><span style='background-color: rgba(0, 165, 0, 0.15); border-radius: 3px;' title='0.0'>4</span><span style='background-color: rgba(0, 165, 0, 0.15); border-radius: 3px;' title='0.0'>,</span><span style='background-color: rgba(0, 165, 0, 0.15); border-radius: 3px;' title='0.0'> 1</span><span style='background-color: rgba(0, 165, 0, 0.15); border-radius: 3px;' title='0.0'>3</span><span style='background-color: rgba(0, 165, 0, 0.15); border-radius: 3px;' title='0.0'>,</span><span style='background-color: rgba(0, 165, 0, 0.15); border-radius: 3px;' title='0.0'> 1</span><span style='background-color: rgba(0, 165, 0, 0.15); border-radius: 3px;' title='0.0'>2</span><span style='background-color: rgba(0, 165, 0, 0.15); border-radius: 3px;' title='0.0'>,</span><span style='background-color: rgba(0, 165, 0, 0.15); border-radius: 3px;' title='0.0'> 1</span><span style='background-color: rgba(0, 165, 0, 0.15); border-radius: 3px;' title='0.0'>1</span><span style='background-color: rgba(0, 165, 0, 0.15); border-radius: 3px;' title='0.0'>,</span><span style='background-color: rgba(0, 165, 0, 0.15); border-radius: 3px;' title='0.0'> 1</span><span style='background-color: rgba(0, 165, 0, 0.15); border-radius: 3px;' title='0.0'>0</span><span style='background-color: rgba(0, 165, 0, 0.15); border-radius: 3px;' title='0.0'>,</span><span style='background-color: rgba(0, 165, 0, 0.15); border-radius: 3px;' title='0.0'> 9</span><span style='background-color: rgba(0, 165, 0, 0.15); border-radius: 3px;' title='0.0'>,</span><span style='background-color: rgba(0, 165, 0, 0.15); border-radius: 3px;' title='0.0'> 8</span><span style='background-color: rgba(0, 165, 0, 0.15); border-radius: 3px;' title='0.0'>,</span></pre>

    {'output': ' 17, 16, 15, 14, 13, 12, 11, 10, 9, 8,'}

See [the
documentation](https://amaiya.github.io/onprem/examples_guided_prompts.html)
for more examples of how to use
[Guidance](https://github.com/guidance-ai/guidance) with **OnPrem.LLM**.

### Summarization Pipeline

Summarize your raw documents (e.g., PDFs, MS Word) with an LLM.

``` python
from onprem import LLM
llm = LLM(n_gpu_layers=35, verbose=False, mute_stream=True) # disabling viewing of intermediate summarization prompts/inferences
```

``` python
from onprem.pipelines import Summarizer
summ = Summarizer(llm)

text = summ.summarize('sample_data/1/ktrain_paper.pdf', max_chunks_to_use=5) # omit max_chunks_to_use parameter to consider entire document
print(text)
```

     The KTrain library provides an easy-to-use framework for building and training machine learning models using low-code techniques for various data types (text, image, graph, tabular) and tasks (classification, regression). It can be used to fine-tune pretrained models in text classification and image classification tasks respectively. Additionally, it reduces cognitive load by providing a unified interface to various and disparate machine learning tasks, allowing users to focus on more important tasks that may require domain expertise or are less amenable to automation.

### Text to Code Generation

We’ll use the CodeUp LLM by supplying the URL and employing the
particular prompt format this model expects.

``` python
from onprem import LLM

url = "https://huggingface.co/TheBloke/CodeUp-Llama-2-13B-Chat-HF-GGUF/resolve/main/codeup-llama-2-13b-chat-hf.Q4_K_M.gguf"
llm = LLM(url, n_gpu_layers=43)  # see below for GPU information
```

Setup the prompt based on what [this model
expects](https://huggingface.co/TheBloke/CodeUp-Llama-2-13B-Chat-HF-GGUF#prompt-template-alpaca)
(this is important):

``` python
template = """
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:"""
```

``` python
answer = llm.prompt(
    "Write Python code to validate an email address.", prompt_template=template
)
```


    Here is an example of Python code that can be used to validate an email address:
    ```
    import re

    def validate_email(email):
        # Use a regular expression to check if the email address is in the correct format
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(pattern, email):
            return True
        else:
            return False

    # Test the validate_email function with different inputs
    print("Email address is valid:", validate_email("example@example.com"))  # Should print "True"
    print("Email address is invalid:", validate_email("example@"))  # Should print "False"
    print("Email address is invalid:", validate_email("example.com"))  # Should print "False"
    ```
    The code defines a function `validate_email` that takes an email address as input and uses a regular expression to check if the email address is in the correct format. The regular expression checks for an email address that consists of one or more letters, numbers, periods, hyphens, or underscores followed by the `@` symbol, followed by one or more letters, periods, hyphens, or underscores followed by a `.` and two to three letters.
    The function returns `True` if the email address is valid, and `False` otherwise. The code also includes some test examples to demonstrate how to use the function.

Let’s try out the code generated above.

``` python
import re


def validate_email(email):
    # Use a regular expression to check if the email address is in the correct format
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if re.match(pattern, email):
        return True
    else:
        return False


print(validate_email("sam@@openai.com"))  # bad email address
print(validate_email("sam@openai"))  # bad email address
print(validate_email("sam@openai.com"))  # good email address
```

    False
    False
    True

The generated code may sometimes need editing, but this one worked
out-of-the-box.

### Using OnPrem.LLM with REST APIs like vLLM

**OnPrem.LLM** can be used with LLMs being served through any
OpenAI-compatible REST API. This means you can use **OnPrem.LLM** with
tools like [vLLM](https://github.com/vllm-project/vllm),
[OpenLLM](https://github.com/bentoml/OpenLLM),
[Ollama](https://ollama.com/blog/openai-compatibility), and the
[llama.cpp
server](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md).

For instance, using [vLLM](https://github.com/vllm-project/vllm), you
can serve a LLaMA 3 model as follows:

``` sh
python -m vllm.entrypoints.openai.api_server --model NousResearch/Meta-Llama-3-8B-Instruct --dtype auto --api-key token-abc123
```

You can then connect OnPrem.LLM to the LLM by supplying the URL of the
server you just started:

``` python
from onprem import LLM
llm = LLM(model_url='http://localhost:8000/v1')
```

You can now easily access the served LLM to solve problems with
**OnPrem.LLM** as you normally would (e.g., RAG question-answering,
summarization, few-shot prompting, code generation, etc.).

### Using OpenAI Models with OnPrem.LLM

Even when using on-premises language models, it can sometimes be useful
to have easy access to non-local, cloud-based models (e.g., OpenAI) for
testing, producing baselines for comparison, and generating synthetic
examples for fine-tuning. For these reasons, in spite of the name,
**OnPrem.LLM** now includes support for OpenAI chat models:

``` python
from onprem import LLM
llm = LLM(model_url='openai://gpt-3.5-turbo', temperature=0) # ChatGPT
```

    /home/amaiya/projects/ghub/onprem/onprem/core.py:138: UserWarning: The model you supplied is gpt-3.5-turbo, an external service (i.e., not on-premises). Use with caution, as your data and prompts will be sent externally.
      warnings.warn(f'The model you supplied is {self.model_name}, an external service (i.e., not on-premises). '+\

``` python
saved_result = llm.prompt('List three cute  names for a cat and explain why each is cute.')
```

    1. Whiskers: Whiskers is a cute name for a cat because it perfectly describes one of the most adorable features of a feline - their long, delicate whiskers. It's a playful and endearing name that captures the essence of a cat's charm.

    2. Pudding: Pudding is an incredibly cute name for a cat because it evokes a sense of softness and sweetness. Just like a bowl of creamy pudding, this name brings to mind a cat's cuddly and lovable nature. It's a name that instantly makes you want to snuggle up with your furry friend.

    3. Muffin: Muffin is an adorable name for a cat because it conjures up images of something small, round, and irresistibly cute - just like a cat! This name is playful and charming, and it perfectly captures the delightful and lovable nature of our feline companions.

## Built-In Web App

**OnPrem.LLM** includes a built-in Web app to access the LLM. To start
it, run the following command after installation:

``` shell
onprem --port 8000
```

Then, enter `localhost:8000` (or `<domain_name>:8000` if running on
remote server) in a Web browser to access the application:

<img src="https://raw.githubusercontent.com/amaiya/onprem/master/images/onprem_screenshot.png" border="1" alt="screenshot" width="775"/>

For more information, [see the corresponding
documentation](https://amaiya.github.io/onprem/webapp.html).

## Speeding Up Inference Using a GPU

The above example employed the use of a CPU. If you have a GPU (even an
older one with less VRAM), you can speed up responses. See [the
LangChain docs on
LLama.cpp](https://python.langchain.com/docs/integrations/llms/llamacpp)
for installing `llama-cpp-python` with GPU support for your system.

The steps below describe installing and using `llama-cpp-python` with
`cuBLAS` support and can be employed for GPU acceleration on systems
with NVIDIA GPUs (e.g., Linux, WSL2, Google Colab).

#### Step 1: Install `llama-cpp-python` with cuBLAS support

``` shell
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

# For Mac users replace above with:
# CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```

#### Step 2: Use the `n_gpu_layers` argument with [`LLM`](https://amaiya.github.io/onprem/core.html#llm)

``` python
llm = LLM(n_gpu_layers=35)
```

The value for `n_gpu_layers` depends on your GPU memory and the model
you’re using (e.g., max of 35 for default 7B model). You can reduce the
value if you get an error (e.g., `CUDA error: out-of-memory`). For
instance, using two old NVDIDIA TITAN V GPUs each with 12GB of VRAM, 59
out 83 layers in a [quantized Llama-2 70B
model](https://huggingface.co/TheBloke/Llama-2-70B-chat-GGUF/resolve/main/llama-2-70b-chat.Q3_K_S.gguf)
can be offloaded to the GPUs (i.e., 60 layers or more results in a “CUDA
out of memory” error).

With the steps above, calls to methods like `llm.prompt` will offload
computation to your GPU and speed up responses from the LLM.

The above assumes that NVIDIA drivers and the CUDA toolkit are already
installed. On Ubuntu Linux systems, this can be accomplished [with a
single
command](https://lambdalabs.com/lambda-stack-deep-learning-software).

## FAQ

1.  **How do I use other models with OnPrem.LLM?**

    > You can supply the URL to other models to the `LLM` constructor,
    > as we did above in the code generation example.

    > As of v0.0.20, we support models in GGUF format, which supersedes
    > the older GGML format. You can find llama.cpp-supported models
    > with `GGUF` in the file name on
    > [huggingface.co](https://huggingface.co/models?sort=trending&search=gguf).

    > Make sure you are pointing to the URL of the actual GGUF model
    > file, which is the “download” link on the model’s page. An example
    > for **Mistral-7B** is shown below:

    > <img src="https://raw.githubusercontent.com/amaiya/onprem/master/images/model_download_link.png" border="1" alt="screenshot" width="775"/>

    > Note that some models have specific prompt formats. For instance,
    > the prompt template required for **Zephyr-7B**, as described on
    > the [model’s
    > page](https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF), is:
    >
    > `<|system|>\n</s>\n<|user|>\n{prompt}</s>\n<|assistant|>`
    >
    > So, to use the **Zephyr-7B** model, you must supply the
    > `prompt_template` argument to the `LLM` constructor (or specify it
    > in the `webapp.yml` configuration for the Web app).
    >
    > ``` python
    > # how to use Zephyr-7B with OnPrem.LLM
    > llm = LLM(model_url='https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_K_M.gguf',
    >           prompt_template = "<|system|>\n</s>\n<|user|>\n{prompt}</s>\n<|assistant|>",
    >           n_gpu_layers=33)
    > llm.prompt("List three cute names for a cat.")
    > ```

2.  **I’m behind a corporate firewall and am receiving an SSL error when
    trying to download the model?**

    > Try this:
    >
    > ``` python
    > from onprem import LLM
    > LLM.download_model(url, ssl_verify=False)
    > ```

    > You can download the embedding model (used by `LLM.ingest` and
    > `LLM.ask`) as follows:
    >
    > ``` sh
    > wget --no-check-certificate https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/all-MiniLM-L6-v2.zip
    > ```

    > Supply the unzipped folder name as the `embedding_model_name`
    > argument to `LLM`.

3.  **How do I use this on a machine with no internet access?**

    > Use the `LLM.download_model` method to download the model files to
    > `<your_home_directory>/onprem_data` and transfer them to the same
    > location on the air-gapped machine.

    > For the `ingest` and `ask` methods, you will need to also download
    > and transfer the embedding model files:
    >
    > ``` python
    > from sentence_transformers import SentenceTransformer
    > model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    > model.save('/some/folder')
    > ```

    > Copy the `some/folder` folder to the air-gapped machine and supply
    > the path to `LLM` via the `embedding_model_name` parameter.

4.  **When installing `onprem`, I’m getting errors related to
    `llama-cpp-python` on Windows/Mac/Linux?**

    > See [this LangChain documentation on
    > LLama.cpp](https://python.langchain.com/docs/integrations/llms/llamacpp)
    > for help on installing the `llama-cpp-python` package for your
    > system. Additional tips for different operating systems are shown
    > below:

    > For **Linux** systems like Ubuntu, try this:
    > `sudo apt-get install build-essential g++ clang`. Other tips are
    > [here](https://github.com/oobabooga/text-generation-webui/issues/1534).

    > For **Windows** systems, either use [Windows Subsystem for Linux
    > (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) or
    > install [Microsoft Visual Studio build
    > tools](https://visualstudio.microsoft.com/vs/older-downloads/) and
    > ensure the selections shown in [this
    > post](https://github.com/imartinez/privateGPT/issues/445#issuecomment-1561343405)
    > are installed. WSL is recommended.

    > For **Macs**, try following [these
    > tips](https://github.com/imartinez/privateGPT/issues/445#issuecomment-1563333950).

    > If you still have problems, there are various other tips for each
    > of the above OSes in [this privateGPT repo
    > thread](https://github.com/imartinez/privateGPT/issues/445). Of
    > course, you can also [easily
    > use](https://colab.research.google.com/drive/1LVeacsQ9dmE1BVzwR3eTLukpeRIMmUqi?usp=sharing)
    > **OnPrem.LLM** on Google Colab.

5.  **`llama-cpp-python` is failing to load my model from the model path
    on Google Colab.**

    > For reasons that are unclear, newer versions of `llama-cpp-python`
    > fail to load models on Google Colab unless you supply
    > `verbose=True` to the `LLM` constructor (which is passed directly
    > to `llama-cpp-python`). If you experience this problem locally,
    > try supplying `verbose=True` to `LLM`.

6.  **I’m getting an `“Illegal instruction (core dumped)` error when
    instantiating a `langchain.llms.Llamacpp` or `onprem.LLM` object?**

    > Your CPU may not support instructions that `cmake` is using for
    > one reason or another (e.g., [due to Hyper-V in VirtualBox
    > settings](https://stackoverflow.com/questions/65780506/how-to-enable-avx-avx2-in-virtualbox-6-1-16-with-ubuntu-20-04-64bit)).
    > You can try turning them off when building and installing
    > `llama-cpp-python`:

    > ``` sh
    > # example
    > CMAKE_ARGS="-DLLAMA_CUBLAS=ON -DLLAMA_AVX2=OFF -DLLAMA_AVX=OFF -DLLAMA_F16C=OFF -DLLAMA_FMA=OFF" FORCE_CMAKE=1 pip install --force-reinstall llama-cpp-python --no-cache-dir
    > ```

7.  **How can I speed up
    [`LLM.ingest`](https://amaiya.github.io/onprem/core.html#llm.ingest)
    using my GPU?**

    > Try using the `embedding_model_kwargs` argument:
    >
    > ``` python
    > from onprem import LLM
    > llm  = LLM(embedding_model_kwargs={'device':'cuda'})
    > ```
