# Autogenerated by nbdev

d = { 'settings': { 'branch': 'master',
                'doc_baseurl': '/onprem',
                'doc_host': 'https://amaiya.github.io',
                'git_url': 'https://github.com/amaiya/onprem',
                'lib_path': 'onprem'},
  'syms': { 'onprem.console': {},
            'onprem.core': { 'onprem.core.AnswerConversationBufferMemory': ('core.html#answerconversationbuffermemory', 'onprem/core.py'),
                             'onprem.core.AnswerConversationBufferMemory.save_context': ( 'core.html#answerconversationbuffermemory.save_context',
                                                                                          'onprem/core.py'),
                             'onprem.core.LLM': ('core.html#llm', 'onprem/core.py'),
                             'onprem.core.LLM.__init__': ('core.html#llm.__init__', 'onprem/core.py'),
                             'onprem.core.LLM.ask': ('core.html#llm.ask', 'onprem/core.py'),
                             'onprem.core.LLM.chat': ('core.html#llm.chat', 'onprem/core.py'),
                             'onprem.core.LLM.check_model': ('core.html#llm.check_model', 'onprem/core.py'),
                             'onprem.core.LLM.download_model': ('core.html#llm.download_model', 'onprem/core.py'),
                             'onprem.core.LLM.ingest': ('core.html#llm.ingest', 'onprem/core.py'),
                             'onprem.core.LLM.is_azure': ('core.html#llm.is_azure', 'onprem/core.py'),
                             'onprem.core.LLM.is_local': ('core.html#llm.is_local', 'onprem/core.py'),
                             'onprem.core.LLM.is_local_api': ('core.html#llm.is_local_api', 'onprem/core.py'),
                             'onprem.core.LLM.is_openai_model': ('core.html#llm.is_openai_model', 'onprem/core.py'),
                             'onprem.core.LLM.load_chatqa': ('core.html#llm.load_chatqa', 'onprem/core.py'),
                             'onprem.core.LLM.load_ingester': ('core.html#llm.load_ingester', 'onprem/core.py'),
                             'onprem.core.LLM.load_llm': ('core.html#llm.load_llm', 'onprem/core.py'),
                             'onprem.core.LLM.load_qa': ('core.html#llm.load_qa', 'onprem/core.py'),
                             'onprem.core.LLM.load_vectordb': ('core.html#llm.load_vectordb', 'onprem/core.py'),
                             'onprem.core.LLM.prompt': ('core.html#llm.prompt', 'onprem/core.py'),
                             'onprem.core.LLM.update_max_tokens': ('core.html#llm.update_max_tokens', 'onprem/core.py'),
                             'onprem.core.LLM.update_stop': ('core.html#llm.update_stop', 'onprem/core.py')},
            'onprem.guider': { 'onprem.guider.Guider': ('guider.html#guider', 'onprem/guider.py'),
                               'onprem.guider.Guider.__init__': ('guider.html#guider.__init__', 'onprem/guider.py'),
                               'onprem.guider.Guider.prompt': ('guider.html#guider.prompt', 'onprem/guider.py')},
            'onprem.ingest': { 'onprem.ingest.Ingester': ('ingest.html#ingester', 'onprem/ingest.py'),
                               'onprem.ingest.Ingester.__init__': ('ingest.html#ingester.__init__', 'onprem/ingest.py'),
                               'onprem.ingest.Ingester.get_db': ('ingest.html#ingester.get_db', 'onprem/ingest.py'),
                               'onprem.ingest.Ingester.get_embedding_model': ( 'ingest.html#ingester.get_embedding_model',
                                                                               'onprem/ingest.py'),
                               'onprem.ingest.Ingester.get_ingested_files': ('ingest.html#ingester.get_ingested_files', 'onprem/ingest.py'),
                               'onprem.ingest.Ingester.ingest': ('ingest.html#ingester.ingest', 'onprem/ingest.py'),
                               'onprem.ingest.MyElmLoader': ('ingest.html#myelmloader', 'onprem/ingest.py'),
                               'onprem.ingest.MyElmLoader.load': ('ingest.html#myelmloader.load', 'onprem/ingest.py'),
                               'onprem.ingest.batchify_chunks': ('ingest.html#batchify_chunks', 'onprem/ingest.py'),
                               'onprem.ingest.does_vectorstore_exist': ('ingest.html#does_vectorstore_exist', 'onprem/ingest.py'),
                               'onprem.ingest.load_documents': ('ingest.html#load_documents', 'onprem/ingest.py'),
                               'onprem.ingest.load_single_document': ('ingest.html#load_single_document', 'onprem/ingest.py'),
                               'onprem.ingest.process_documents': ('ingest.html#process_documents', 'onprem/ingest.py')},
            'onprem.pipelines.classifier': { 'onprem.pipelines.classifier.ClassifierBase': ( 'pipelines.classifier.html#classifierbase',
                                                                                             'onprem/pipelines/classifier.py'),
                                             'onprem.pipelines.classifier.ClassifierBase.arrays2dataset': ( 'pipelines.classifier.html#classifierbase.arrays2dataset',
                                                                                                            'onprem/pipelines/classifier.py'),
                                             'onprem.pipelines.classifier.ClassifierBase.dataset2arrays': ( 'pipelines.classifier.html#classifierbase.dataset2arrays',
                                                                                                            'onprem/pipelines/classifier.py'),
                                             'onprem.pipelines.classifier.ClassifierBase.evaluate': ( 'pipelines.classifier.html#classifierbase.evaluate',
                                                                                                      'onprem/pipelines/classifier.py'),
                                             'onprem.pipelines.classifier.ClassifierBase.explain': ( 'pipelines.classifier.html#classifierbase.explain',
                                                                                                     'onprem/pipelines/classifier.py'),
                                             'onprem.pipelines.classifier.ClassifierBase.get_labels': ( 'pipelines.classifier.html#classifierbase.get_labels',
                                                                                                        'onprem/pipelines/classifier.py'),
                                             'onprem.pipelines.classifier.ClassifierBase.get_trainer': ( 'pipelines.classifier.html#classifierbase.get_trainer',
                                                                                                         'onprem/pipelines/classifier.py'),
                                             'onprem.pipelines.classifier.ClassifierBase.sample_examples': ( 'pipelines.classifier.html#classifierbase.sample_examples',
                                                                                                             'onprem/pipelines/classifier.py'),
                                             'onprem.pipelines.classifier.ClassifierBase.save': ( 'pipelines.classifier.html#classifierbase.save',
                                                                                                  'onprem/pipelines/classifier.py'),
                                             'onprem.pipelines.classifier.ClassifierBase.train': ( 'pipelines.classifier.html#classifierbase.train',
                                                                                                   'onprem/pipelines/classifier.py'),
                                             'onprem.pipelines.classifier.FewShotClassifier': ( 'pipelines.classifier.html#fewshotclassifier',
                                                                                                'onprem/pipelines/classifier.py'),
                                             'onprem.pipelines.classifier.FewShotClassifier.__init__': ( 'pipelines.classifier.html#fewshotclassifier.__init__',
                                                                                                         'onprem/pipelines/classifier.py'),
                                             'onprem.pipelines.classifier.FewShotClassifier.save': ( 'pipelines.classifier.html#fewshotclassifier.save',
                                                                                                     'onprem/pipelines/classifier.py'),
                                             'onprem.pipelines.classifier.FewShotClassifier.train': ( 'pipelines.classifier.html#fewshotclassifier.train',
                                                                                                      'onprem/pipelines/classifier.py')},
            'onprem.pipelines.extractor': { 'onprem.pipelines.extractor.Extractor': ( 'pipelines.extractor.html#extractor',
                                                                                      'onprem/pipelines/extractor.py'),
                                            'onprem.pipelines.extractor.Extractor.__init__': ( 'pipelines.extractor.html#extractor.__init__',
                                                                                               'onprem/pipelines/extractor.py'),
                                            'onprem.pipelines.extractor.Extractor.apply': ( 'pipelines.extractor.html#extractor.apply',
                                                                                            'onprem/pipelines/extractor.py'),
                                            'onprem.pipelines.extractor.Extractor.segment': ( 'pipelines.extractor.html#extractor.segment',
                                                                                              'onprem/pipelines/extractor.py')},
            'onprem.pipelines.summarizer': { 'onprem.pipelines.summarizer.Summarizer': ( 'pipelines.summarizer.html#summarizer',
                                                                                         'onprem/pipelines/summarizer.py'),
                                             'onprem.pipelines.summarizer.Summarizer.__init__': ( 'pipelines.summarizer.html#summarizer.__init__',
                                                                                                  'onprem/pipelines/summarizer.py'),
                                             'onprem.pipelines.summarizer.Summarizer._map_reduce': ( 'pipelines.summarizer.html#summarizer._map_reduce',
                                                                                                     'onprem/pipelines/summarizer.py'),
                                             'onprem.pipelines.summarizer.Summarizer._refine': ( 'pipelines.summarizer.html#summarizer._refine',
                                                                                                 'onprem/pipelines/summarizer.py'),
                                             'onprem.pipelines.summarizer.Summarizer.summarize': ( 'pipelines.summarizer.html#summarizer.summarize',
                                                                                                   'onprem/pipelines/summarizer.py')},
            'onprem.utils': { 'onprem.utils.download': ('utils.html#download', 'onprem/utils.py'),
                              'onprem.utils.get_datadir': ('utils.html#get_datadir', 'onprem/utils.py'),
                              'onprem.utils.split_list': ('utils.html#split_list', 'onprem/utils.py')},
            'onprem.webapp': {}}}
