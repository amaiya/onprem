# Changes

Most recent releases are shown at the top. Each release shows:

- **New**: New classes, methods, functions, etc
- **Changed**: Additional parameters, changes to inputs or outputs, etc
- **Fixed**: Bug fixes that don't change documented behaviour


## 0.13.0 (TBD)

### new:
- Support for loading htm, json, and xlsx extensions (#161)
- Speed up `remove_document` functionality in vector stores #163

### changed
- avoid multiprocessing if `n_proc=1` due to Windows (#162)

### fixed:
- Fixed and organized upload UI
- fix ChromaDB HNSW "ef or M is too small" error (#162)


## 0.12.8 (2025-04-04)

### new:
- N/A

### changed
- N/A

### fixed:
- Normalize path separators for Windows support of folder filtering (#160)


## 0.12.7 (2025-04-04)

### new:
- N/A

### changed
- Support `folder` argument in `LLM.ask` and `LLM.query` (#158)
- folder filter for RAG in Web UI (#159)

### fixed:
- retain RAG results on screen in Web UI (#157)
- pin to `pymupdf4llm==0.0.17` becaue later versions produce
  bad results


## 0.12.6 (2025-04-03)

### new:
- N/A

### changed
- N/A

### fixed:
- disable parallelization in web interface when on Windows (#155)
- fix folder filter on Windows systems (#156)


## 0.12.5 (2025-04-03)

### new:
- N/A

### changed
- N/A

### fixed:
- workaround torch issue described [here](https://github.com/VikParuchuri/marker/issues/442)
- fix folder filtering for semantic search (#154)


## 0.12.4 (2025-04-02)

### new:
- N/A

### changed
- N/A

### fixed:
- Remove mimetype warning for Windows
- Web UI Search page fixes/improvements


## 0.12.3 (2025-04-02)

### new:
- N/A

### changed
- N/A

### fixed:
- Fix generation placement in Web app.


## 0.12.2 (2025-04-02)

### new:
- N/A

### changed
- N/A

### fixed:
- Set llm model name properly in model cards in web app (#152)
- Use `kwargs["model"]` as `LLM.model_name` if present (#152)
- Remove display of `n_gpu_layers` (#152)


## 0.12.1 (2025-04-01)

### new:
- N/A

### changed
- N/A

### fixed:
- remove None in spinner during interactive chat (#151)


## 0.12.0 (2025-04-01)

### new:
- Re-vamped Web UI
- Support for dual vectorstore that stores documents
  in both dense vectorstore and sparse vectorstore

### changed
- convert webapp to multipage (#150)
- models now stored in `models` subfolder within
  `onprem_data` by default.
- `streamlit` added as dependency

### fixed:
- N/A


## 0.11.1 (2025-03-28)

### new:
- N/A

### changed
- support folder creation in `utils.download` (#149)

### fixed:
- pin to `sentence_transformers<4` due to SetFit bug


## 0.11.0 (2025-03-26)

### new:
- add `LLM.set_store_type` method (#147)

### changed
- Default model changed to Zephyr-7b-beta (#148)
- remove `LLM.ask_with_memory` (#146)

### fixed:
- source is empty during OCR (#144)


## 0.10.1 (2025-03-26)

### new:
- N/A

### changed
- N/A

### fixed:
- ensure chat returns string response (#142)
- revert to `dense` as default `store_type` (#143)



## 0.10.0 (2025-03-25)

### new:
- support for custom metadata in vectorstore (#126)
- basic full-text indexing (#132)
- support for using sparse vector stores with `LLM.ask` (#136)
- support complex RAG filtering (#137)

### changed
- **Breaking Changes**: Use sparse vector stores as default (#141)
- **Breaking Changes**:`LLM.chat` renamed to `LLM.ask_with_memory`.
  `LLM.chat` is now a simple conversational chatbot (no RAG) (#138)
- **Breaking Changes**: refactor vectorstore (#133, 1e84f46)
- **Breaking Changes**: Vector stores are stored within a subfolder
  of `LLMvectordb_path` (either `dense` or `sparse`) (#140)
- use os.walk instead of glob for `extract_files` and remove dot
  from extensions (#127)
- Add `batch_size` parameter to `LLM.ingest` (#128)
- use generators in `load_documents` (#129)
- Changed `split_list` to `batch_list`
- explicitly define available metadata types (#131)
- use GPU for embeddings by default, if available (#135)

### fixed:
- Use `load_vectordb` to load vector database in `LLM.query` (#130)
- disable progress bar for `pdf_markdown` (due to notebook issue) (#134)
- fix bugs and add tests for vector store updates (#139)


## 0.9.0 (2025-02-26)

### new:
- Support for using self-ask prompt strategy with RAG (#120)
- Improved table understanding when invoking `LLm.ask`. (#124)
- helpers for document metadata (#121)

### changed
- Added `k` and `score_threshold` arguments to `LLM.ask` (#122)
- Added `n_proc` paramter to control the number of CPUs used 
  by `LLM.ingest` (ee09807)
- Upgrade version of `chromadb` (#125)

### fixed:
- Ensure table-processing is sequential and not parallelized (#123)
- Fixes to support newer version of `langchain_community`. (#125)


## 0.8.0 (2025-02-13)

### new:
- Added `HFClassifier` to `pipelines.classifier` module (#119)
- Added `SKClassifier` to `pipelines.classifier` module (#118)
- `sk` "helper" module to fit simple scikit-learn text models (#117)

### changed
- Added `process_documents` function (#117)

### fixed:
- Pass `autodetect_encoding` argument to `TextLoader` (#116)


## 0.7.1 (2024-12-18)

### new:
- N/A

### changed
- N/A

### fixed:
- Fix for HF chat template issue (#113/#114)


## 0.7.0 (2024-12-16)

### new:
- Support for structured outputs (#110)
- Support for table extraction (#106, #107)
- Facilitate identifying tables extracted as HTML (#112)

### changed
- Remove dependnency on deprecated RetrievalQA (#108)
- Refactored code base (#109)
- Use new JSON-safe formatting of prompt templates (#109)

### fixed:
- Added `utils.format_string` function to help format template strings
  with embedded JSON (#105)
- support stop strings with transformers (#111)


## 0.6.1 (2024-12-04)

### new:
- N/A

### changed
- Changed `pdf_use_unstructured` to `pdf_unstructured` and
  `pdf2md` to `pdf_markdown` (#102)

### fixed:
- N/A

## 0.6.0 (2024-12-03)

### new:
- Improved PDF text extraction including optional markdown
  conversion, table inference, and OCR (#100)

### changed
- N/A

### fixed:
- Add support for HF training (#98)
- Default to localhost in Web app (#99)


## 0.5.2 (2024-011-25)

### new:
- N/A

### changed
- N/A

### fixed:
- Allow all Hugging Face pipeline/model arguments to be supplied (#96)


## 0.5.1 (2024-11-22)

### new:
- N/A

### changed
- Refactored Hugging Face transformers backend (#95)

### fixed:
- Suppress swig deprecation warning (#93)
- Raise error if summarizers encounter bad document (#94)


## 0.5.0 (2024-11-20)

### new:
- Support for Hugging Face transformers as LLM engine instead of Llama.cpp

### changed
- `LLM.prompt` now accepts OpenAI-style messages in form of list of dictionaries

### fixed:
- Remove unused imports (#92)


## 0.4.0 (2024-11-13)

### new:
- Added `default_model` parameter to `LLM` to more easily use `Llama-3.1-8B-Instruct`.

### changed
- N/A

### fixed:
- N/A


## 0.3.2 (2024-11-08)

### new:
- N/A

### changed
- Added key-value pair, `ocr:True`, to `Document.metadata` when PDF is OCR'ed (#91)

### fixed:
- removed dead code in `pipelines.summarizer` (#88)


## 0.3.1 (2024-10-18)

### new:
- N/A

### changed
- Removed `include_surrounding` parameter from `summarize_by_concept`

### fixed:
- N/A


## 0.3.0 (2024-10-11)

### new:
- Support for concept-focused summarizations (#87)

### changed
- Replace `use_larger` parameter with `use_zephyr`

### fixed:
- Replace deprecated `CallbackManager` (#86)


## 0.2.4 (2024-09-30)

### new:
- N/A

### changed
- N/A

### fixed:
- Check if `docs` is None (#85)


## 0.2.3 (2024-09-27)

### new:
- N/A

### changed
- N/A

### fixed:
- Fixed error when raising Exceptions in `Ingester` (#84)


## 0.2.2 (2024-09-26)

### new:
- N/A

### changed
- N/A

### fixed:
- Resolve issues with PDFs that mix OCR/not-OCR (#83)


## 0.2.1 (2024-09-26)

### new:
- N/A

### changed
- Auto set some unstructured settings based on input (#81)

### fixed:
- Ensure any supplied `unstructured` `kwargs` do not persist (#81)



## 0.2.0 (2024-09-25)

### new:
- Better PDF OCR support and table-handling (#75, #80)

### changed
- add `pdf_use_unstructured` argument to `LLM.ingest` for
  PDF OCR and better table-handling (#79)
- Allow configuration of unstructured for PDFs from `LLM.ingest` (#80)

### fixed:
- N/A


## 0.1.4 (2024-09-25)

### new:
- OCR support (#75)

### changed
- Added `Ingester.store_documents` method (#36,#77)

### fixed:
- switch to `langchain_huggingface` and `langchain_chroma` (#78)


## 0.1.3 (2024-08-16)

### new:
- N/A

### changed
- N/A

### fixed:
Added `preproc_fn` to `Extractor.apply` (#74)


## 0.1.2 (2024-06-05)

### new:
- N/A

### changed
- N/A

### fixed:
- Segment needs to accept arguments in extractor pipeline (#70)


## 0.1.1 (2024-06-03)

### new:
- N/A

### changed
- Add clean function to `Extractor.apply` (#69)

### fixed:
- Remove BOS token from default prompt (#67)
- Remove call to `db.persist` (#68)


## 0.1.0 (2024-06-01)

### new:
- Use OnPrem.LLM with OpenAI-compatible REST APIs (#61)
- information extraction pipeline (#64)
- experimental support for Azure OpenAI (#63)
- Docker support
- Few-Shot classification pipeline (#66)

### changed
- change default model to Mistral (#65)
- allow installation of onprem without llama-cpp-python for easier use with LLMs served through 
  REST APIs (#62)
- Added `ignore_fn` argument to `LLM.ingest` to allow more control over ignoring certain files (#58)
- Added `Ingester.get_ingested_files` to show files ingested into vector database (#59)

### fixed:
- If encountering a loading error when processing a file, skip and continue instead of halting (#60)
- Add check for partially download files (#49)


## 0.0.36 (2024-01-16)

### new:
- Support for OpenAI models (#55) 

### changed
- `LLM.prompt`, 'LLM.ask`, and `LLM.chat` now accept extra `**kwargs` that are sent diretly to model (#54)

### fixed:
- N/A


## 0.0.35 (2024-01-15)

### new:
- N/A

### changed
- Updates for `langchain>=0.1.0` (which is now minimum version)

### fixed:
- N/A


## 0.0.34 (2024-01-13)

### new:
- Uses [Zephyr-7B](https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF) as default model in `webapp.yml`. (#52)

### changed
- Added `stop` paramter to `LLM.prompt` (overrides `stop` paramter supplied to constructor) (#53)

### fixed:
- N/A


## 0.0.33 (2024-01-08)

### new:
- N/A

### changed
- Added `prompt_template` parameter to `LLM` constructor (#51)
- Added `update_max_tokens` and `update_stop` methods to `LLM` for dynamic adjustments during prompt experiments

### fixed:
- Explicitly set `offload_kqv` to ensure GPUs are fully utilized (#50)


## 0.0.32 (2023-12-10)

### new:
- Summarization pipeline (#35)

### changed
- Upgrades to all dependencies, but pin `chromadb==0.4.15` to retain compatibilitiy with older langchain
- Default `n_ctx` (context window) changed to 3900

### fixed:
- N/A


## 0.0.31 (2023-12-09)

### new:
- The `guider` module, a simplistic interface to [Guidance](https://github.com/guidance-ai/guidance) (#34)

### changed
- N/A

### fixed:
- N/A


## 0.0.30 (2023-12-07)

### new:
- N/A

### changed
- progress bar for embeddings creation (#46)
- Support model-specific prompt templates in `LLM.ask` method (#47)

### fixed:
- Added `python-docx` as dependency (#43)
- Added `python-pptx` as dependency (#44)
- Pass `prompt_template` to `ask` method in Web app (#47)
- Skip file beginning with '~$' in `LLM.ingest` (#45)


## 0.0.29 (2023-10-27)

### new:
- N/A

### changed
- N/A

### fixed:
- Added warning if URL is not pointing to GGUF model file. (#40)


## 0.0.28 (2023-10-06)

### new:
- N/A

### changed
- N/A

### fixed:
- Changed default value for `verbose` in `LLM` from False to True due to `llama-cpp-python` bug (#37)


## 0.0.27 (2023-09-30)

### new:
- N/A

### changed
- Remove pin for `llama-cpp-python` so latest is always used (#33)

### fixed:
- N/A


## 0.0.26 (2023-09-27)

### new:
- N/A

### changed
- Include `prompt_template` variable in YAML (#32)

### fixed:
- N/A



## 0.0.25 (2023-09-27)

### new:
- N/A

### changed
- **Breaking Change**: The `LLM.ask` method now returns a dictionary with keys: `answer`, `source_documents`, and `question` (#31)

### fixed:
- N/A


## 0.0.24 (2023-09-26)

### new:
- N/A

### changed
- Added `rag_text_path` and `verbose` to default `webapp.yml`.

### fixed:
- Moving `load_llm` to constructor seems to prevent model loading issues in `Llamacpp` (#30)


## 0.0.23 (2023-09-25)

### new:
- N/A

### changed
- round scores in web app to 3 decimal places (#29)

### fixed:
- N/A


## 0.0.22 (2023-09-24)

### new:
- attempt to auto-create symlinks for serving source documents

### changed
- N/A

### fixed:
- N/A


## 0.0.21 (2023-09-22)

### new:
- Support for hyperlinks to sources in RAG screen of  Web app (#28)

### changed
- N/A

### fixed:
- `LLM.ingest` converts relative paths to absolute paths during ingestion


## 0.0.20 (2023-09-22)

### new:
- Support for `GGUF` format as the default LLM format. (#1)

### changed
- All default models have been changed to `GGUF` models.
- updated pin for `llama-cpp-python` to support GGUF format.

### fixed:
- Misc adjustments and bug fixes for built-in Web app


## 0.0.19 (2023-09-21)

### new:
- Built-in Web app for both RAG and general prompting

### changed
- **Possible Breaking Change**: Support for `score_threshold` in `LLM.ask` and `LLM.chat` (#26)
- Use `CallbackManager` (#24)

### fixed:
- N/A



## 0.0.18 (2023-09-18)

### new:
- N/A

### changed
- `LLM.chat` now includes `source_documents` in output (#23)

### fixed:
- N/A


## 0.0.17 (2023-09-17)

### new:
- The `LLM.chat` method supports question-answering with conversational memory. (#20)

### changed
- `LLM` now accepts a `callbacks` parameter for custom callbacks. (#21)
- added additional examples

### fixed:
- N/A


## 0.0.16 (2023-09-12)

### new:
- Support for prompt templates in `ask` (#17)

### changed
- Added `LLM.load_qa` method

### fixed:
- batchify input to `Chroma` (#18)


## 0.0.15 (2023-09-11)

### new:
- N/A

### changed
- N/A

### fixed:
- pass `embedding_model_kwargs` and `embedding_encode_kwargs` to `HuggingFaceEmbeddings` (#16)


## 0.0.14 (2023-09-11)

### new:
- N/A

### changed
- Added `Ingester.get_embeddings` method to access instance of `HuggingFaceEmbeddings`
- Added `chunk_size` and `chunk_overlap` parameters to `Ingester.ingest` and `LLM.ingest` (#13)

### fixed:
- Check to ensure `source_directory` is a folder in `LLM.ingest` (#15)


## 0.0.13 (2023-09-10)

### new:
- N/A

### changed
- Accept extra `kwargs` and supply them to `langchain.llms.Llamacpp` (#12)
- Add optional argument to specify custom path to vector DB (#11)

### fixed:
- N/A


## 0.0.12 (2023-09-09)

### new:
- N/A

### changed
- Add optional argument to specify custom path to download LLM (#5), thanks to @rabilrbl

### fixed:
- Fixed capitalization in download confirmation (#9), thanks to @rabilrbl
- Insert [dummy replacement](https://stackoverflow.com/questions/74918614/error-importing-seaborn-module-attributeerror/76760670#76760670) of decorator into numpy



## 0.0.11 (2023-09-07)

### new:
- N/A

### changed
- Print `persist_directory` when creating new vector store
- Revert `numpy` pin

### fixed:
- N/A



## 0.0.10 (2023-09-07)

### new:
- N/A

### changed
- Pin to `numpy==1.23.3` due to `_no_nep50` error in some environments

### fixed:
- N/A


## 0.0.9 (2023-09-06)

- Last release without CHANGELOG updates

