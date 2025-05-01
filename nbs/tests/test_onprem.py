"""
This is a simple local test of basic functionality in script form.
The "Exception ignored in: <function Llama.__del__ at 0x7fe2d96d7430>" from llama-cpp-python is
ignored and doesn't affect functionality.
"""

from onprem import LLM, utils as U
import os, tempfile, shutil, argparse



def test_prompt(**kwargs):
    llm = kwargs.get('llm', None)
    if not llm: raise ValueError('llm arg is required')
    #prompt = """Extract the names of people in the supplied sentences. Separate the names with commas.
    #[Sentence]: I like Cillian Murphy's acting. Florence Pugh is great, too.
    #[People]:"""
    prompt = """Extract the names of people in the supplied sentences.
# Example 1:
Sentence: James Gandolfini and Paul Newman were great actors.
People:
James Gandolfini, Paul Newman

# Example 2:
Sentence:
I like Cillian Murphy's acting. Florence Pugh is great, too.
People:"""
    saved_output = llm.prompt(prompt, stop=['\n\n'] )
    assert saved_output.strip().startswith("Cillian Murphy, Florence Pugh"), "bad response"
    print()
    print()
    return

def test_rag_dual(**kwargs):
    llm = kwargs.get('llm', None)
    if not llm: raise ValueError('llm arg is required')
    original_vectordb_path = llm.vectordb_path
    original_store_type = llm.store_type
    llm.vectordb_path = tempfile.mkdtemp()
    llm.set_store_type('dual')
    
    print(llm.vectordb_path)

    # make source folder
    source_folder = tempfile.mkdtemp()

    # download ktrain paper
    U.download(
        "https://raw.githubusercontent.com/amaiya/onprem/master/nbs/tests/sample_data/ktrain_paper/ktrain_paper.pdf",
        os.path.join(source_folder, "ktrain.pdf"),
    )

    # ingest ktrain paper
    llm.ingest(source_folder)
    assert os.path.exists(source_folder)
    
    # Verify both stores have documents
    assert os.path.exists(os.path.join(llm.vectordb_path, 'dense'))
    assert os.path.exists(os.path.join(llm.vectordb_path, 'sparse'))
    
    # Verify we can retrieve documents from both stores
    dense_docs = llm.vectorstore.dense_store.get_all_docs()
    sparse_docs = llm.vectorstore.sparse_store.get_all_docs()
    assert len(list(dense_docs)) > 0
    assert len(list(sparse_docs)) > 0
    
    # QA on ktrain paper (should use dense store by default)
    print()
    print("LLM.ask test (using both stores)")
    print()
    result = llm.ask("What is ktrain?")
    assert len(result["answer"]) > 8
    assert len(result["source_documents"]) > 0
    
    # Test keyword search (should use sparse store)
    keyword_results = llm.vectorstore.keyword_search("ktrain")
    assert len(keyword_results['hits']) > 0
    
    # Test semantic search (should use dense store)
    semantic_results = llm.vectorstore.semantic_search("image classification")
    assert len(semantic_results) > 0
    assert(semantic_results[0].metadata['score'] > 0.45)
    
    # cleanup
    shutil.rmtree(source_folder)
    shutil.rmtree(llm.vectordb_path)
    llm.vectordb_path = original_vectordb_path
    llm.set_store_type(original_store_type)
    
    return

def test_rag_sparse(**kwargs):
    llm = kwargs.get('llm', None)
    if not llm: raise ValueError('llm arg is required')
    original_vectordb_path = llm.vectordb_path
    original_store_type = llm.store_type
    llm.vectordb_path = tempfile.mkdtemp()
    llm.set_store_type('sparse')


    # make source folder
    source_folder = tempfile.mkdtemp()

    # download ktrain paper
    U.download(
        "https://raw.githubusercontent.com/amaiya/onprem/master/nbs/tests/sample_data/ktrain_paper/ktrain_paper.pdf",
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
    print(len(result['source_documents']))
    assert len(result["source_documents"]) == 4
    assert "question" in result
    print()


    # download SOTU
    U.download(
        "https://raw.githubusercontent.com/amaiya/onprem/master/nbs/tests/sample_data/sotu/state_of_the_union.txt",
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

    # test filters
    assert(len(llm.query('climate', where_document='affordable')) == 1)
    assert(len(llm.query('climate', filters={'extension':'txt'})) == 2)
    assert(len(llm.query('climate', filters={'extension':'pdf'})) == 0)

    # test semantic search
    result = llm.query('image classification')
    assert(result[0].metadata['score'] > 0.45)

    # test updates
    store = llm.load_vectorstore()
    doc = list(store.get_all_docs())[0]
    id = doc['id']
    print(id)
    assert(store.get_doc(id)['id'] == id)
    doc['document_title'] = 'TEST TITLE'
    doc['page_content'] = 'XYZ'
    store.update_documents([doc])
    assert(store.get_doc(id)['document_title'] == 'TEST TITLE')
    assert(store.get_doc(id)['page_content'].endswith('XYZ'))
    store.remove_document(id)
    assert(store.get_doc(id) is None)


    # cleanup
    shutil.rmtree(source_folder)
    shutil.rmtree(llm.vectordb_path)
    llm.vectordb_path = original_vectordb_path
    llm.set_store_type(original_store_type)


def test_rag_dense(**kwargs):
    llm = kwargs.get('llm', None)
    if not llm: raise ValueError('llm arg is required')

    original_vectordb_path = llm.vectordb_path
    original_store_type = llm.store_type
    llm.vectordb_path = tempfile.mkdtemp()
    llm.set_store_type('dense')



    # make source folder
    source_folder = tempfile.mkdtemp()

    # download ktrain paper
    U.download(
        "https://raw.githubusercontent.com/amaiya/onprem/master/nbs/tests/sample_data/ktrain_paper/ktrain_paper.pdf",
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
        "https://raw.githubusercontent.com/amaiya/onprem/master/nbs/tests/sample_data/sotu/state_of_the_union.txt",
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

    # download MS financial statement
    U.download(
        "https://raw.githubusercontent.com/amaiya/onprem/master/nbs/tests/sample_data/financial_statement/ms-financial-statement.pdf",
        os.path.join(source_folder, "ms-financial-statement.pdf"),
    )

    # ingest but ignore MS financial statement
    llm.ingest(source_folder, ignore_fn=lambda x: os.path.basename(x) == 'ms-financial-statement.pdf')
    ingested_files = llm.vectorstore.get_all_docs()
    sources = set([os.path.basename(x['source']) for x in ingested_files])
    print(sources)
    assert(len(sources) == 2)
    assert('ms-inancial-statement.pdf' not in sources)

    # test large k
    store = llm.load_vectorstore()
    results = store.semantic_search('What is machine learning?', k=store.get_size())
    print(f'# of results for lage k: {len(results)}')
    assert(len(results) > 100)

    # test updates
    store = llm.load_vectorstore()
    doc = list(store.get_all_docs())[0]
    id = doc['id']
    print(id)
    assert(store.get_doc(id)['id'] == id)
    doc['document_title'] = 'TEST TITLE'
    doc['page_content'] = 'XYZ'
    store.update_documents([doc])
    assert(store.get_doc(id)['document_title'] == 'TEST TITLE')
    assert(store.get_doc(id)['page_content'].endswith('XYZ'))
    store.remove_document(id)
    assert(store.get_doc(id) is None)

    # test vector query
    docs = llm.query('image classification')
    assert(docs[0].metadata['score'] >= 0.45)

    # cleanup
    shutil.rmtree(source_folder)
    shutil.rmtree(llm.vectordb_path)
    llm.vectordb_path = original_vectordb_path
    llm.set_store_type(original_store_type)

    return

def test_guider(**kwargs):
    llm = kwargs.get('llm', None)
    if not llm: raise ValueError('llm arg is required')
 
    if llm.is_openai_model(): return

    from onprem.pipelines.guider import Guider
    guider = Guider(llm)

    from guidance import select, gen

    prompt = f"""Question: Luke has ten balls. He gives three to his brother. How many balls does he have left?  Answer: """ + gen('answer', regex=r'\d+')
    result = guider.prompt(prompt)
    print(result)
    assert(int(result['answer']) == 7)


def test_summarization(**kwargs):
 
    llm = kwargs.get('llm', None)
    if not llm: raise ValueError('llm arg is required')

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

    #text,_ = summ.summarize_by_concept(os.path.join( os.path.dirname(os.path.realpath(__file__)),
                                       #'1/ktrain_paper.pdf'),
                                        #concept_description="automl")
    #print(text)
    #assert(len(text) > 0)


def test_extraction(**kwargs):
    llm = kwargs.get('llm', None)
    if not llm: raise ValueError('llm arg is required')

    from onprem.pipelines import Extractor
    extractor = Extractor(llm)
    prompt = """The following text is delimited by ### and is from the first page of a research paper.  Extract the author of the research paper.
    If the author is listed, begin the response with "Author:" and then output the name and nothing else.
    If there is no author mentioned in the text, output NA.
    ###{text}###
    """
    fpath = os.path.join( os.path.dirname(os.path.realpath(__file__)), 'sample_data/ktrain_paper/ktrain_paper.pdf')
    df = extractor.apply(prompt, fpath=fpath, pdf_pages=[1], stop=[])
    #print(df.loc[df['Extractions'] != 'NA'].Extractions[0])
    author = df.loc[~df['Extractions'].isin(['NA', 'Author: NA'])].Extractions.values.tolist()[0]
    print('-----')
    print(author)
    print('-----')
    assert 'Arun S. Maiya' in author or 'Maiya, Arun S.' in author


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

    #clf.train(X_sample,  y_sample, max_steps=20)
    clf.train(X_sample,  y_sample, max_steps=1)
    
    acc =  clf.evaluate(X_test, y_test, labels=clf.model.labels, print_report=False)['accuracy'] 
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
        store_type='dense',
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
    store = llm.load_vectorstore()
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
        docs = store.query(query)
        print(f"{query} : {docs[0].page_content}")
        assert docs[0].page_content == matches[query]
    shutil.rmtree(source_folder)
    shutil.rmtree(vectordb_path)
    return

def test_loading(**kwargs):
    from onprem.ingest import load_single_document, process_folder
    from onprem.utils import segment

    # test markdown and extra/custom fields
    fpath = os.path.join( os.path.dirname(os.path.realpath(__file__)), 'sample_data/ktrain_paper/ktrain_paper.pdf')
    docs = load_single_document(fpath, pdf_markdown=True,
                                store_md5=True, store_mimetype=True, store_file_dates=True,
                                text_callables={'FIRST_PARAGRAPH':lambda text:text.split("\n\n")[0]},
                                file_callables={'EXT':lambda fname:fname.split(".")[-1]})
    assert len(docs) == 1
    paragraphs = segment(docs[0].page_content, unit='paragraph')
    md_id = 0
    for i, p in enumerate(paragraphs):
        if p.startswith('#'):
            md_id = i
            break

    assert paragraphs[md_id].startswith('#')
    assert(docs[0].metadata['EXT'] == 'pdf')
    assert( len(docs[0].metadata['FIRST_PARAGRAPH']) > 0)
    assert(docs[0].metadata['mimetype'] == 'application/pdf')
    assert(docs[0].metadata['extension'] == 'pdf')
    assert(docs[0].metadata['md5'] == 'c562b02005810b05f6ac4b17732ab4b0')
    assert(docs[0].metadata.get('createdate', None) is not None)

    # test table inference
    docs = load_single_document(fpath, infer_table_structure=True)
    assert len(docs) == 7
    assert docs[-1].page_content.startswith("Table 1")


    # test paragraph chunking on TXT
    sotu_folder = os.path.join( os.path.dirname(os.path.realpath(__file__)), 'sample_data/sotu')
    docs = process_folder(sotu_folder, strict_paragraph_preservation=True)
    assert(len(list(docs)) == 91)


def test_pdftables(**kwargs):
    """
    Not currently run as test_pdf also tests tables
    """
    from onprem.ingest.pdftables import PDFTables
    fpath = os.path.join( os.path.dirname(os.path.realpath(__file__)), 'sample_data/billionaires/The_Worlds_Billionaires.pdf')
    pdftab = PDFTables.from_file(fpath, verbose=True)
    assert len(pdftab.dfs) > 30 
    assert len(pdftab.captions) > 30
    assert len(pdftab.dfs) == len(pdftab.captions)
    assert pdftab.captions[-6] == "Number and combined net worth of billionaires by year [66] See also"

def test_pydantic(**kwargs):
    """
    Test Structured Output
    """
    llm = kwargs.get('llm', None)
    if not llm: raise ValueError('llm arg is required')
    from pydantic import BaseModel, Field

    class Joke(BaseModel):
        setup: str = Field(description="question to set up a joke")
        punchline: str = Field(description="answer to resolve the joke")
    structured_output = llm.pydantic_prompt('Tell me a joke.', pydantic_model=Joke)
    try:
        assert issubclass(type(structured_output), BaseModel)
    except:
        print('Malformed/incomplete output - attepmting fix with OpenAI model...')
        from langchain_openai import ChatOpenAI
        structured_output = llm.pydantic_prompt('Tell me a joke.', pydantic_model=Joke,
                                                attempt_fix=True, fix_llm=ChatOpenAI())
        assert issubclass(type(structured_output), BaseModel)
    print(structured_output.setup)
    print(structured_output.punchline)


def test_transformers(**kwargs):
    #llm = LLM(model_id='gpt2', device_map='cpu')
    llm = LLM(model_id='Qwen/Qwen2.5-0.5B-Instruct', device_map='cpu')
    output = llm.prompt('Repeat the word Paris three times.',
                        stop=['Paris'])
    assert("paris" in output.lower())

def test_tm(**kwargs):
    """
    Test topic modeling
    """
    import numpy as np
    import pandas as pd
    from onprem.sk.tm import get_topic_model
    from sklearn.datasets import fetch_20newsgroups

    remove = ("headers", "footers", "quotes")
    newsgroups_train = fetch_20newsgroups(subset="train", remove=remove)
    newsgroups_test = fetch_20newsgroups(subset="test", remove=remove)
    texts = newsgroups_train.data + newsgroups_test.data
    MIN_WORDS = 30
    texts = [text for text in texts if len(text.split()) > MIN_WORDS]
    tm = get_topic_model(texts, model_type='lda', max_iter=5, n_topics=50)
    tm.build(texts, threshold=0.25)
    rawtext = """Elon Musk leads Space Exploration Technologies (SpaceX), where he oversees
the development and manufacturing of advanced rockets and spacecraft for missions
to and beyond Earth orbit."""
    predicted_topic_id = np.argmax(tm.predict([rawtext]))
    assert("space" in tm.topics[predicted_topic_id])

    # scorer
    tm.train_scorer(topic_ids=[predicted_topic_id])
    other_topics = [i for i in range(tm.n_topics) if i not in [predicted_topic_id]]
    other_texts = [d["text"].strip() for d in tm.get_docs(topic_ids=other_topics) if len(d['text'].strip().split()) > MIN_WORDS]
    other_scores = tm.score(other_texts)
    pd.set_option("display.max_colwidth", None)

    other_preds = [int(score > 0) for score in other_scores]
    data = sorted(
        list(zip(other_preds, other_scores, other_texts[:250])),
        key=lambda item: item[1],
        reverse=True,
    )
    df = pd.DataFrame(data, columns=["Prediction", "Score", "Text"])
    assert('space' in df[df.Score>0].head()['Text'].tolist()[0])

    # recommender
    tm.train_recommender()
    results = tm.recommend(text=rawtext, n=15)
    assert('space' in results[0]['text'])

def test_skclassifier(**kwargs):
    """
    Test scikit-learn classifier
    """

    categories = [
                 "alt.atheism",
                 "soc.religion.christian",
                 "comp.graphics",
                 "sci.med" ]
    from sklearn.datasets import fetch_20newsgroups

    train_b = fetch_20newsgroups(
                subset="train", categories=categories, shuffle=True, random_state=42
    )
    test_b = fetch_20newsgroups(
    subset="test", categories=categories, shuffle=True, random_state=42
    )
    x_train = train_b.data
    y_train = train_b.target
    x_test = test_b.data
    y_test = test_b.target
    classes = train_b.target_names

    from onprem.pipelines import SKClassifier
    clf = SKClassifier()
    clf.train(x_train, y_train)
    test_doc = "god christ jesus mother mary church sunday lord heaven amen"
    acc = clf.evaluate(x_test, y_test, print_report=False)['accuracy']
    assert(3 == clf.predict(test_doc))  
    assert(acc > 0.85) # should 0.89+ for default SGDClassifier and 0.93 for NBSVM
    print(acc)


def test_hfclassifier(**kwargs):
    """
    Test scikit-learn classifier
    """

    categories = [
                 "alt.atheism",
                 "soc.religion.christian",
                 "comp.graphics",
                 "sci.med" ]
    from sklearn.datasets import fetch_20newsgroups

    train_b = fetch_20newsgroups(
                subset="train", categories=categories, shuffle=True, random_state=42
    )
    test_b = fetch_20newsgroups(
    subset="test", categories=categories, shuffle=True, random_state=42
    )
    x_train = train_b.data
    y_train = train_b.target
    x_test = test_b.data
    y_test = test_b.target
    classes = train_b.target_names

    from onprem.pipelines import HFClassifier
    clf = HFClassifier()
    clf.train(x_train, y_train)
    test_doc = "god christ jesus mother mary church sunday lord heaven amen"
    acc = clf.evaluate(x_test, y_test, print_report=False)['accuracy']
    assert(3 == clf.predict(test_doc))  
    print(acc)
    assert(acc > 0.8)
    return


def test_search(**kwargs):
    """
    Test search
    """
    from onprem.ingest.stores import SparseStore
    from onprem.ingest import load_single_document, chunk_documents
    docs = load_single_document('sample_data/ktrain_paper/ktrain_paper.pdf', 
                                store_md5=True)
    docs = chunk_documents(docs)
    se = SparseStore()
    se.add_documents(docs)
    assert(se.get_size() ==  41)
    assert(len(list(se.get_all_docs())) ==  41)
    assert(len(se.query("table")['hits']) == 2)
    assert(len(se.query("table", limit=2, page=1)['hits']) == 2)
    assert(len(se.query("table", limit=1, page=1)['hits']) == 1)
    assert(len(se.query("table", limit=1, page=2)['hits']) == 1)
    assert(se.query('table')['total_hits'] == 2)
    assert(len(se.query("table", limit=1, page=3)['hits']) == 0)
    assert(se.query("table", limit=1)['hits'][0].metadata['md5'] == 'c562b02005810b05f6ac4b17732ab4b0')
    assert(len(se.query('page:5')['hits']) == 6)
    assert(len(se.query('page:6')['hits']) == 0)
    doc_id = se.query('table')['hits'][0].metadata['id']
    assert(len(se.query(f'id:{doc_id}')['hits']) == 1)
    assert(se.query(f'id:{doc_id}')['total_hits'] == 1)
    assert(se.get_doc(doc_id)['id'] == doc_id)
    assert(se.get_doc('XXX') is None)
    se.remove_document(doc_id)
    print(se.get_size())
    assert(se.get_doc(doc_id) is None)
    assert(se.get_size() ==  40)
    assert(len(list(se.get_all_docs())) ==  40)

    # ensure text is normalized (ligature issues are addressed)
    assert(len(se.query('classification')['hits']) == 10)




TESTS = { 'test_prompt' : test_prompt,
          #'test_guider' : test_guider, # Guidance tends to segfault with newer llama_cpp
          'test_rag_dense'    : test_rag_dense,
          'test_rag_sparse'    : test_rag_sparse,
          'test_rag_dual'     : test_rag_dual,
          'test_summarization' : test_summarization,
          'test_extraction' : test_extraction,
          'test_classifier' : test_classifier,
          'test_loading' : test_loading,
          'test_pydantic' : test_pydantic,
          'test_semantic' : test_semantic,
          'test_tm' : test_tm,
          'test_skclassifier' : test_skclassifier,
          'test_hfclassifier' : test_hfclassifier,
          'test_transformers' : test_transformers,
          'test_search' : test_search,}

SHARE_LLM = ['test_prompt', 'test_rag_dense', 'test_rag_sparse', 'test_rag_dual',
             'test_summarization', 'test_extraction', 'test_transformers', 'test_pydantic']

def run(**kwargs):

    prompt_template = kwargs.get('prompt_template', None)
    if prompt_template:
        prompt_template = prompt_template.replace('\\n', '\n')
        print(f'Using prompt_template:{prompt_template}')
        print()
        print()

    if kwargs.get('transformers_only', False):
        test_transformers(**kwargs)
        return

    if kwargs.get('list_tests', False):
        print('List of possible tests:')
        for k in TESTS:
            print(f'\t{k}')
        print()
        print('To run a subset of tests, use --test option. By default, all tests except test_transformers are run.')
        return



    tests = kwargs['test']
    to_run = tests if tests else list(TESTS.keys())
    print(f'Running the following tests: {", ".join(to_run)}')
    print()

    # setup
    url = kwargs["url"]
    n_gpu_layers = kwargs["gpu"]
    print(url)

    if len(set(to_run) & set(SHARE_LLM)) > 0:
        llm = LLM(model_url=url, n_gpu_layers=n_gpu_layers, max_tokens=128, prompt_template=prompt_template)
        kwargs['llm'] = llm

    for test in to_run:
        fn = TESTS.get(test, None)
        if not fn:
            print(f'{test} is invalid - skipping')
            continue
        print('----------------------------------------')
        print(f'Running {test}:')
        print('----------------------------------------')
        fn(**kwargs)


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
        default="https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_K_M.gguf",
        help=("URL of model. Default is a URL to Zephyr-7b-beta."),
    )
    optional_args.add_argument(
        "-p",
        "--prompt-template",
        type=str,
        help=("Prompt template to use. Should have a single variable {prompt}. Not required if default model url is used."),
    )
    optional_args.add_argument(
        "-o",
        "--transformers-only",
        action="store_true",
        help=("Only test transformers."),
    )

    optional_args.add_argument(
        "-l",
        "--list-tests",
        action="store_true",
        help=("List all registered tests."),
    )

    optional_args.add_argument('-t', '--test', nargs='+',
                               help='Names of specific tests to run. Use --list to see all possible tests')

    args = parser.parse_args()

    kwargs = {}
    kwargs["gpu"] = args.gpu
    kwargs["url"] = args.url
    kwargs['prompt_template'] = args.prompt_template
    kwargs['transformers_only'] = args.transformers_only
    kwargs['test'] = args.test
    kwargs['list_tests'] = args.list_tests
    run(**kwargs)
