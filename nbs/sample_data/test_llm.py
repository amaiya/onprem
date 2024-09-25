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
    assert saved_output.strip().startswith("Cillian Murphy, Florence Pugh"), "bad response"
    print()
    print()
    return


def test_rag(llm, **kwargs):
    llm.vectordb_path = tempfile.mkdtemp()

    # make source folder
    source_folder = tempfile.mkdtemp()

    # download ktrain paper
    U.download(
        "https://raw.githubusercontent.com/amaiya/onprem/master/nbs/sample_data/1/ktrain_paper.pdf",
        os.path.join(source_folder, "ktrain.pdf"),
    )

    # ingest ktrain paper
    llm.ingest(source_folder)
    assert os.path.exists(source_folder)

    # QA on ktrain paper
    print()
    print("LLM.ask test")
    print()
    result = llm.ask("What is ktrain?")
    assert len(result["answer"]) > 8
    assert len(result["source_documents"]) == 4
    assert "question" in result
    print()


    # download SOTU
    U.download(
        "https://raw.githubusercontent.com/amaiya/onprem/master/nbs/sample_data/3/state_of_the_union.txt",
        os.path.join(source_folder, "sotu.txt"),
    )

    # ingest SOTU
    llm.ingest(source_folder)

    # QA on SOTU
    print()
    result = llm.ask("Who is Ketanji? Brown Jackson")
    assert len(result["answer"]) > 8
    assert "question" in result
    assert "source_documents" in result
    print()

    # QA chat test
    print()
    print("LLM.chat test")
    print()
    result = llm.chat("What is ktrain?")
    assert len(result["answer"]) > 8
    assert "question" in result
    assert "source_documents" in result
    assert "chat_history" in result
    print()
    print()
    result = llm.chat("Does it support image classification?")
    assert len(result["answer"]) > 8
    print()
    print()

    # download MS financial statement
    U.download(
        "https://raw.githubusercontent.com/amaiya/onprem/master/nbs/sample_data/2/ms-financial-statement.pdf",
        os.path.join(source_folder, "ms-financial-statement.pdf"),
    )

    # ingest but ignore MS financial statement
    llm.ingest(source_folder, ignore_fn=lambda x: os.path.basename(x) == 'ms-financial-statement.pdf')
    ingested_files = llm.ingester.get_ingested_files()
    print([os.path.basename(x) for x in ingested_files])
    assert(len(ingested_files) == 2)
    assert('ms-inancial-statement.pdf' not in [os.path.basename(x) for x in ingested_files])


    # cleanup
    shutil.rmtree(source_folder)
    shutil.rmtree(llm.vectordb_path)
    return

def test_guider(llm, **kwargs):
    if llm.is_openai_model(): return

    from onprem.guider import Guider
    guider = Guider(llm)

    from guidance import select, gen

    prompt = f"""Question: Luke has ten balls. He gives three to his brother. How many balls does he have left?  Answer: """ + gen('answer', regex='\d+')
    result = guider.prompt(prompt)
    print(result)
    assert(int(result['answer']) == 7)


def test_summarization(llm, **kwargs):

    from onprem.pipelines import Summarizer
    summ = Summarizer(llm)
    from langchain_community.document_loaders import WebBaseLoader

    loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    docs = loader.load()
    with open('/tmp/blog.txt', 'w') as f:
        f.write(docs[0].page_content)
    text = summ.summarize('/tmp/blog.txt', max_chunks_to_use=1) 
    assert('output_text' in text)
    print(text['output_text'])
    assert(len(text['output_text']) > 0)



def test_extraction(llm, **kwargs):
    from onprem.pipelines import Extractor
    extractor = Extractor(llm)
    prompt = """The following text is delimited by ### and is from the first page of a research paper.  Extract the author of the research paper.
    If the author is listed, begin the response with "Author:" and then output the name and nothing else.
    If there is no author mentioned in the text, output NA.
    ###{text}###
    """
    fpath = os.path.join( os.path.dirname(os.path.realpath(__file__)), '1/ktrain_paper.pdf')
    df = extractor.apply(prompt, fpath=fpath, pdf_pages=[1], stop=[])
    #print(df.loc[df['Extractions'] != 'NA'].Extractions[0])
    author = df.loc[~df['Extractions'].isin(['NA', 'Author: NA'])].Extractions.values.tolist()[0]
    print(author)
    assert 'Arun S. Maiya' in author


def test_classifier(**kwargs):
    from onprem.pipelines import FewShotClassifier
    clf = FewShotClassifier(use_smaller=True)


    from sklearn.datasets import fetch_20newsgroups
    import pandas as pd
    import numpy as np


    # Fetching data
    classes = ["soc.religion.christian", "sci.space"]
    newsgroups = fetch_20newsgroups(subset="all", categories=classes)
    corpus = np.array(newsgroups.data)
    group_labels = np.array(newsgroups.target_names)[newsgroups.target]

    # Wrangling data into a dataframe and selecting training examples
    data = pd.DataFrame({"text": corpus, "label": group_labels})
    train_df = data.groupby("label").sample(5)
    test_df = data.drop(index=train_df.index)

    # small sample of entire dataset set (and much smaller than the test set)
    X_sample = train_df['text'].values
    y_sample = train_df['label'].values

    # test set
    X_test = test_df['text'].values
    y_test = test_df['label'].values

    #clf.train(X_sample,  y_sample, num_epochs=1, batch_size=16, num_iterations=20)
    clf.train(X_sample,  y_sample, max_steps=20)
    
    acc =  clf.evaluate(X_test, y_test, labels=clf.model.labels)['accuracy'] 
    print(acc)
    assert acc > 0.9

    assert clf.predict(['Elon Musk likes launching satellites.']).tolist()[0] == 'sci.space'



def test_semantic(**kwargs):
    vectordb_path = tempfile.mkdtemp()
    url = kwargs["url"]
    llm = LLM(
        model_url=url,
        embedding_model_name="sentence-transformers/nli-mpnet-base-v2",
        embedding_encode_kwargs={"normalize_embeddings": True},
        vectordb_path=vectordb_path,
    )
    data = [  # from txtai
        "US tops 5 million confirmed virus cases",
        "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
        "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
        "The National Park Service warns against sacrificing slower friends in a bear attack",
        "Maine man wins $1M from $25 lottery ticket",
        "Make huge profits without work, earn up to $100,000 a day",
    ]
    source_folder = tempfile.mkdtemp()
    for i, d in enumerate(data):
        filename = os.path.join(source_folder, f"doc{i}.txt")
        with open(filename, "w") as f:
            f.write(d)
    llm.ingest(source_folder, chunk_size=500, chunk_overlap=0)
    db = llm.load_ingester().get_db()
    matches = {
        "feel good story": data[4],
        "climate change": data[1],
        "public health story": data[0],
        "war": data[2],
        "wildlife": data[3],
        "asia": data[2],
        "lucky": data[4],
        "dishonest junk": data[5],
    }
    for query in (
        "feel good story",
        "climate change",
        "public health story",
        "war",
        "wildlife",
        "asia",
        "lucky",
        "dishonest junk",
    ):
        docs = db.similarity_search(query)
        print(f"{query} : {docs[0].page_content}")
        assert docs[0].page_content == matches[query]
    shutil.rmtree(source_folder)
    shutil.rmtree(vectordb_path)
    return


def run(**kwargs):
    # setup
    url = kwargs["url"]
    n_gpu_layers = kwargs["gpu"]
    print(url)
    llm = LLM(model_url=url, n_gpu_layers=n_gpu_layers)


    # prompt test
    test_prompt(llm, **kwargs)

    # guided prompt test
    test_guider(llm, **kwargs)

    # rag test
    test_rag(llm, **kwargs)

    # summarization test
    test_summarization(llm, **kwargs)

    # extraction test
    test_extraction(llm, **kwargs)

    # classifier test
    test_classifier(**kwargs)

    # semantic simlarity test
    test_semantic(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run a simple test of onprem.LLM. "
            "(Note that the small models may be prone to hallucination, but this will not affect the tests in this script.)"
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
        default="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        help=("URL of model. Default is a URL to Mistral-7B-Instruct-v0.2."),
    )

    args = parser.parse_args()

    kwargs = {}
    kwargs["gpu"] = args.gpu
    kwargs["url"] = args.url
    run(**kwargs)
