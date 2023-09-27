"""
This is a simple local test of basic functionality in script form.
The "Exception ignored in: <function Llama.__del__ at 0x7fe2d96d7430>" from llama-cpp-python is
ignored and doesn't affect functionality.
"""

from onprem import LLM, utils as U
import os, tempfile, shutil, argparse


def test_prompt(llm, **kwargs):
    prompt = """Extract the names of people in the supplied sentences. Here is an example:
Sentence: James Gandolfini and Paul Newman were great actors.
People:
James Gandolfini, Paul Newman
Sentence:
I like Cillian Murphy's acting. Florence Pugh is great, too.
People:"""
    saved_output = llm.prompt(prompt)
    assert saved_output.strip() == 'Cillian Murphy, Florence Pugh', "bad response"
    print()
    print()
    return

def test_rag(llm, **kwargs):
    llm.vectordb_path = tempfile.mkdtemp()

    source_folder = tempfile.mkdtemp()
    U.download('https://raw.githubusercontent.com/amaiya/onprem/master/nbs/sample_data/1/ktrain_paper.pdf', 
              os.path.join(source_folder, 'ktrain.pdf'))
    llm.ingest(source_folder)
    assert(os.path.exists(source_folder))
    print()
    print('LLM.ask test')
    print()
    result = llm.ask("What is ktrain?")
    assert(len(result['answer']) > 8)
    assert(len(result['source_documents'])==4)
    assert('question' in result)
    print()
    U.download('https://raw.githubusercontent.com/amaiya/onprem/master/nbs/sample_data/3/state_of_the_union.txt', 
              os.path.join(source_folder, 'sotu.txt'))
    llm.ingest(source_folder)
    print()
    result = llm.ask("Who is Ketanji? Brown Jackson")
    assert(len(result['answer']) > 8)
    assert('question' in result)
    assert('source_documents' in result)
    print()
    print()
    print('LLM.chat test')
    print()
    result = llm.chat("What is ktrain?")
    assert(len(result['answer']) > 8)
    assert('question' in result)
    assert('source_documents' in result)
    assert('chat_history' in result)
    print()
    print()
    result = llm.chat("Does it support image classification?")
    assert(len(result['answer']) > 8)
    print()
    print()
    shutil.rmtree(source_folder)
    shutil.rmtree(llm.vectordb_path)
    return


def test_semantic(**kwargs):
    vectordb_path = tempfile.mkdtemp()
    url = kwargs['url']
    llm = LLM(
              url=url,
              embedding_model_name='sentence-transformers/nli-mpnet-base-v2', 
              embedding_encode_kwargs={'normalize_embeddings': True},
              vectordb_path=vectordb_path)
    data = [ # from txtai
      "US tops 5 million confirmed virus cases",
      "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
      "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
      "The National Park Service warns against sacrificing slower friends in a bear attack",
      "Maine man wins $1M from $25 lottery ticket",
      "Make huge profits without work, earn up to $100,000 a day"
    ]
    source_folder = tempfile.mkdtemp()
    for i, d in enumerate(data):
        filename = os.path.join(source_folder, f'doc{i}.txt')
        with open(filename, 'w') as f:
            f.write(d)
    llm.ingest(source_folder, chunk_size=500, chunk_overlap=0)
    db = llm.load_ingester().get_db()
    matches = { 'feel good story': data[4],
                'climate change': data[1],
                'public health story' : data[0],
                'war' : data[2],
                'wildlife' : data[3],
                'asia' : data[2],
                'lucky' : data[4],
                'dishonest junk' : data[5] }
    for query in ("feel good story", "climate change", "public health story", "war", "wildlife", "asia", "lucky", "dishonest junk"):
        docs = db.similarity_search(query)
        print(f'{query} : {docs[0].page_content}')
        assert(docs[0].page_content == matches[query])
    shutil.rmtree(source_folder)
    shutil.rmtree(vectordb_path)
    return



def run(**kwargs):



    # setup
    url = kwargs['url']
    n_gpu_layers = kwargs['gpu']
    llm = LLM(model_url=url, n_gpu_layers=n_gpu_layers)

    # prompt test
    test_prompt(llm, **kwargs)

    # rag test
    test_rag(llm, **kwargs)


    # semantic simlarity test
    test_semantic(**kwargs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            'Run a simple test of onprem.LLM. '
            '(Note that the small models may be prone to hallucination, but this will not affect the tests in this script.)'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )


    optional_args = parser.add_argument_group("optional arguments")
    optional_args.add_argument(
        "-g",
        "--gpu",
        type=int,
        default=0,
        help=("Number of layers offloaded to GPU. Default is 0 (CPU is used)."),
    )

    optional_args.add_argument(
        "-u",
        "--url",
        type=str,
        default="https://huggingface.co/juanjgit/orca_mini_3B-GGUF/resolve/main/orca-mini-3b.q4_0.gguf",
        help=("URL of model. Default is a URL to orca-mini-3b.q4_0.gguf."),
    )

    args = parser.parse_args()

    kwargs = {}
    kwargs["gpu"] = args.gpu
    kwargs["url"] = args.url
    run(**kwargs)

