# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/04_pipelines.summarizer.ipynb.

# %% auto 0
__all__ = ['DEFAULT_MAP_PROMPT', 'DEFAULT_REDUCE_PROMPT', 'DEFAULT_BASE_REFINE_PROMPT', 'DEFAULT_REFINE_PROMPT', 'Summarizer']

# %% ../../nbs/04_pipelines.summarizer.ipynb 3
import os
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, Callable
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

from ..ingest import load_single_document, load_documents


# %% ../../nbs/04_pipelines.summarizer.ipynb 4
DEFAULT_MAP_PROMPT = """The following is a set of documents
{docs}
Based on this list of docs, please write a concise summary. 
CONCISE SUMMARY:"""

DEFAULT_REDUCE_PROMPT = """The following is set of summaries:
{docs}
Take these and distill it into a final, consolidated summary. 
SUMMARY:"""

DEFAULT_BASE_REFINE_PROMPT = """Write a concise summary of the following:
{text}
CONCISE SUMMARY:"""

DEFAULT_REFINE_PROMPT = (
    "Your job is to produce a final summary\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to refine the existing summary"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "Given the new context, refine the original summary."
    "If the context isn't useful, return the original summary."
)

# %% ../../nbs/04_pipelines.summarizer.ipynb 5
class Summarizer:
    def __init__(
        self,
        llm,
        prompt_template: Optional[str] = None,              
        map_prompt: Optional[str] = None,
        reduce_prompt: Optional[str] = None,
        refine_prompt: Optional[str] = None, 
        **kwargs,
    ):
        """
        `Summarizer` summarizes one or more documents

        **Args:**

        - *llm*: An `onprem.LLM` object
        - *prompt_template*: A model specific prompt_template with a single placeholder named "{prompt}".
                             All prompts (e.g., Map-Reduce prompts) are wrapped within this prompt.
                             If supplied, overrides the `prompt_template` supplied to the `LLM` constructor.
        - *map_prompt*: Map prompt for Map-Reduce summarization. If None, default is used.
        - *reduce_prompt*: Reduce prompt for Map-Reduce summarization. If None, default is used.
        - *refine_prompt*: Refine prompt for Refine-based summarization. If None, default is used.

        """
        self.llm = llm
        self.prompt_template = prompt_template if prompt_template is not None else llm.prompt_template
        self.map_prompt = map_prompt if map_prompt else DEFAULT_MAP_PROMPT
        self.reduce_prompt = reduce_prompt if reduce_prompt else DEFAULT_REDUCE_PROMPT
        self.refine_prompt = refine_prompt if refine_prompt else DEFAULT_REFINE_PROMPT


    def summarize(self, 
                  fpath:str, 
                  strategy:str='map_reduce',
                  chunk_size:int=1000, 
                  chunk_overlap:int=0, 
                  token_max:int=2000,
                  max_chunks_to_use: Optional[int] = None,
                 ):
        """
        Summarize one or more documents (e.g., PDFs, MS Word, MS Powerpoint, plain text)
        using either Langchain's Map-Reduce strategy or Refine strategy.

        **Args:**

        - *fpath*: A path to either a folder of documents or a single file.
        - *strategy*: One of {'map_reduce', 'refine'}. 
        - *chunk_size*: Number of characters of each chunk to summarize
        - *chunk_overlap*: Number of characters that overlap between chunks
        - *token_max*: Maximum number of tokens to group documents into
        - *max_chunks_to_use*: Maximum number of chunks (starting from beginning) to use.
                               Useful for documents that have abstracts or informative introductions.
                               If None, all chunks are considered for summarizer.

        **Returns:**

        - str: a summary of your documents
        """
          
        if os.path.isfile(fpath):
            docs = load_single_document(fpath)
        else:
            docs = load_documents(fpath)

        if strategy == 'map_reduce':
            summary = self._map_reduce(docs, 
                                      chunk_size=chunk_size, 
                                      chunk_overlap=chunk_overlap, 
                                      token_max=token_max,
                                      max_chunks_to_use=max_chunks_to_use)
        elif strategy == 'refine':
            summary = self._refine(docs, 
                                   chunk_size=chunk_size, 
                                   chunk_overlap=chunk_overlap, 
                                   token_max=token_max,
                                   max_chunks_to_use=max_chunks_to_use)
            
        else:
            raise ValueError(f'Unknown strategy: {self.strategy}')
        return summary

    
    def _map_reduce(self, docs, chunk_size=1000, chunk_overlap=0, token_max=1000, 
                    max_chunks_to_use = None, **kwargs):
        """ Map-Reduce summarization"""
        langchain_llm = self.llm.llm

        # Map
        # map_template = """The following is a set of documents
        # {docs}
        # Based on this list of docs, please identify the main themes 
        # Helpful Answer:"""
        # map_template = """The following is a set of documents
        # {docs}
        # Based on this list of docs, please write a concise summary.
        # CONCISE SUMMARY:"""
        map_template = self.map_prompt
        if self.prompt_template: 
            map_template = self.prompt_template.format(**{'prompt':map_template})
        map_prompt = PromptTemplate.from_template(map_template)
        map_chain = LLMChain(llm=langchain_llm, prompt=map_prompt)

        # Reduce
        # reduce_template = """The following is set of summaries:
        # {docs}
        # Take these and distill it into a final, consolidated summary. 
        # SUMMARY:"""
        reduce_template = self.reduce_prompt
        if self.prompt_template:
            reduce_template = self.prompt_template.format(**{'prompt':reduce_template})
        reduce_prompt = PromptTemplate.from_template(reduce_template)

        # Run chain
        reduce_chain = LLMChain(llm=langchain_llm, prompt=reduce_prompt)
        
        # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="docs"
        )
        
        # Combines and iteravely reduces the mapped documents
        reduce_documents_chain = ReduceDocumentsChain(
            # This is final chain that is called.
            combine_documents_chain=combine_documents_chain,
            # If documents exceed context for `StuffDocumentsChain`
            collapse_documents_chain=combine_documents_chain,
            # The maximum number of tokens to group documents into.
            token_max=token_max)

        # Combining documents by mapping a chain over them, then combining results
        map_reduce_chain = MapReduceDocumentsChain(
            # Map chain
            llm_chain=map_chain,
            # Reduce chain
            reduce_documents_chain=reduce_documents_chain,
            # The variable name in the llm_chain to put the documents in
            document_variable_name="docs",
            # Return the results of the map steps in the output
            return_intermediate_steps=False,
        )
        
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        split_docs = text_splitter.split_documents(docs)
        split_docs = split_docs[:max_chunks_to_use] if max_chunks_to_use else split_docs

        return map_reduce_chain.invoke(split_docs)

    def _refine(self, docs, chunk_size=1000, chunk_overlap=0, 
                max_chunks_to_use = None, **kwargs):
        """ Refine summarization"""

        # initial_template = """Write a concise summary of the following:
        # {text}
        # CONCISE SUMMARY:"""
        initial_template = self.refine_base_prompt
        if self.prompt_template:
            initial_template = self.prompt_template.format(**{'prompt':initial_template})
        prompt = PromptTemplate.from_template(initial_template)
        
        # refine_template = (
        #     "Your job is to produce a final summary\n"
        #     "We have provided an existing summary up to a certain point: {existing_answer}\n"
        #     "We have the opportunity to refine the existing summary"
        #     "(only if needed) with some more context below.\n"
        #     "------------\n"
        #     "{text}\n"
        #     "------------\n"
        #     "Given the new context, refine the original summary."
        #     "If the context isn't useful, return the original summary."
        # )
        refine_template = self.refine_prompt
        if self.prompt_template:
            refine_template = self.prompt_template.format(**{'prompt':refine_template})
        refine_prompt = PromptTemplate.from_template(refine_template)
        chain = load_summarize_chain(
            llm=self.llm.llm,
            chain_type="refine",
            question_prompt=prompt,
            refine_prompt=refine_prompt,
            return_intermediate_steps=True,
            input_key="input_documents",
            output_key="output_text",
        )
        
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        split_docs = text_splitter.split_documents(docs)
        split_docs = split_docs[:max_chunks_to_use] if max_chunks_to_use else split_docs
        result = chain({"input_documents": split_docs}, return_only_outputs=True)
        print(result)
        return result['output_text']
        