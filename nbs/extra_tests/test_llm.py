"""
This is a simple local test of basic functionality in script form.
The "Exception ignored in: <function Llama.__del__ at 0x7fe2d96d7430>" from llama-cpp-python is
ignored and doesn't affect functionality.
"""

from onprem import LLM, utils as U
import os, tempfile, shutil, argparse


def run(**kwargs):
    # make a temporary directory for our vector database
    vectordb_path = tempfile.mkdtemp()

    # setup
    url = 'https://huggingface.co/TheBloke/orca_mini_3B-GGML/resolve/main/orca-mini-3b.ggmlv3.q4_1.bin'
    url = kwargs['url']
    n_gpu_layers = kwargs['gpu']
    llm = LLM(model_url=url, vectordb_path=vectordb_path, n_gpu_layers=n_gpu_layers)

    # prompt
    saved_output = llm.prompt('What is the capital of France?')
    assert('paris' in saved_output.lower())
    print()
    print()

    # rag

    source_folder = tempfile.mkdtemp()
    U.download('https://raw.githubusercontent.com/amaiya/onprem/master/nbs/sample_data/1/ktrain_paper.pdf', 
              os.path.join(source_folder, 'ktrain.pdf'))
    llm.ingest(source_folder)
    assert(os.path.exists(source_folder))
    print()
    print('LLM.ask test')
    print()
    answer, sources = llm.ask("What is ktrain?")
    assert(len(answer) > 8)
    assert(len(sources)==4)
    print()
    U.download('https://raw.githubusercontent.com/amaiya/onprem/master/nbs/sample_data/3/state_of_the_union.txt', 
              os.path.join(source_folder, 'sotu.txt'))
    llm.ingest(source_folder)
    print()
    answer, sources = llm.ask("Who is Ketanji?")
    assert(len(answer) > 8)
    assert(len(sources)==4)
    print()
    print()
    print('LLM.chat test')
    print()
    result = llm.chat("What is ktrain?")
    assert(len(result['answer']) > 8)
    assert(len(result['source_documents'])==4)
    print()
    result = llm.chat("Does it support image classification?")
    assert(len(result['answer']) > 8)
    assert(len(result['source_documents'])==4)
    print()
    shutil.rmtree(source_folder)
    shutil.rmtree(vectordb_path)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            'Run a simple test of onprem.LLM. (Note that the default model used small, but prone to hallucination.)'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )


    optional_args = parser.add_argument_group("optional arguments")
    optional_args.add_argument(
        "-g",
        "--gpu",
        type=int,
        default=0,
        help=("Number of layers offloaded to GPU"),
    )

    optional_args.add_argument(
        "-u",
        "--url",
        type=str,
        default="https://huggingface.co/TheBloke/orca_mini_3B-GGML/resolve/main/orca-mini-3b.ggmlv3.q4_1.bin",
        help=("URL of model"),
    )

    args = parser.parse_args()

    kwargs = {}
    kwargs["gpu"] = args.gpu
    kwargs["url"] = args.url
    run(**kwargs)















































