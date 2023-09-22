# Changes

Most recent releases are shown at the top. Each release shows:

- **New**: New classes, methods, functions, etc
- **Changed**: Additional parameters, changes to inputs or outputs, etc
- **Fixed**: Bug fixes that don't change documented behaviour

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
- Suport for a Built-in Web app

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

