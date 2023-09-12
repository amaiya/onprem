"""
This is a simple local test of basic functionality in script form.
The "Exception ignored in: <function Llama.__del__ at 0x7fe2d96d7430>" from llama-cpp-python is
ignored and doesn't affect functionality.
"""

from onprem import LLM, utils as U
import os, tempfile, shutil



if __name__ == "__main__":
    # make a temporary directory for our vector database
    vectordb_path = tempfile.mkdtemp()

    # setup
    llm = LLM(vectordb_path=vectordb_path)

    # prompt
    saved_output = llm.prompt('What is the capital of France?')
    print()

    # rag

    source_folder = tempfile.mkdtemp()
    U.download('https://raw.githubusercontent.com/amaiya/onprem/master/nbs/sample_data/1/ktrain_paper.pdf', 
              os.path.join(source_folder, 'ktrain.pdf'))
    llm.ingest(source_folder)
    assert(os.path.exists(source_folder))
    print()
    answer, sources = llm.ask("What is ktrain?")
    assert(len(answer) > 8)
    assert(len(sources)==4)
    print()
    U.download('https://raw.githubusercontent.com/amaiya/onprem/master/nbs/sample_data/3/state_of_the_union.txt', 
              os.path.join(source_folder, 'sotu.txt'))
    llm.ingest(source_folder)
    print()
    answers, sources = llm.ask("Who is Ketanji?")
    assert(len(answer) > 8)
    assert(len(sources)==4)
    print()
    shutil.rmtree(source_folder)
    shutil.rmtree(vectordb_path)

