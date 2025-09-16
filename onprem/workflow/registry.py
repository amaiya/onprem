"""
Node registry for the workflow engine.
"""

# Import all node implementations
from .loaders import LoadFromFolderNode, LoadSingleDocumentNode, LoadWebDocumentNode
from .textsplitters import SplitByCharacterCountNode, SplitByParagraphNode, KeepFullDocumentNode
from .storage import ChromaStoreNode, WhooshStoreNode, ElasticsearchStoreNode
from .query import QueryWhooshStoreNode, QueryChromaStoreNode, QueryElasticsearchStoreNode

# Document Transformer implementations
from .document_transformers import (AddMetadataNode, ContentPrefixNode, ContentSuffixNode, 
                                   DocumentFilterNode, PythonDocumentTransformerNode)

# Processor and Exporter implementations
from .processors import (PromptProcessorNode, ResponseCleanerNode, SummaryProcessorNode,
                        PythonDocumentProcessorNode, PythonResultProcessorNode)
from .exporters import CSVExporterNode, ExcelExporterNode, JSONExporterNode


# Node Registry - maps node type names to node classes
NODE_REGISTRY = {
    # Loaders
    "LoadFromFolder": LoadFromFolderNode,
    "LoadSingleDocument": LoadSingleDocumentNode,
    "LoadWebDocument": LoadWebDocumentNode,
    
    # Text Splitters
    "SplitByCharacterCount": SplitByCharacterCountNode,
    "SplitByParagraph": SplitByParagraphNode,
    "KeepFullDocument": KeepFullDocumentNode,
    
    # Document Transformers
    "AddMetadata": AddMetadataNode,
    "ContentPrefix": ContentPrefixNode,
    "ContentSuffix": ContentSuffixNode,
    "DocumentFilter": DocumentFilterNode,
    "PythonDocumentTransformer": PythonDocumentTransformerNode,
    
    # Storage
    "ChromaStore": ChromaStoreNode,
    "WhooshStore": WhooshStoreNode,
    "ElasticsearchStore": ElasticsearchStoreNode,
    
    # Query
    "QueryWhooshStore": QueryWhooshStoreNode,
    "QueryChromaStore": QueryChromaStoreNode,
    "QueryElasticsearchStore": QueryElasticsearchStoreNode,
    
    # Processors
    "PromptProcessor": PromptProcessorNode,
    "ResponseCleaner": ResponseCleanerNode,
    "SummaryProcessor": SummaryProcessorNode,
    "PythonDocumentProcessor": PythonDocumentProcessorNode,
    "PythonResultProcessor": PythonResultProcessorNode,
    
    # Exporters
    "CSVExporter": CSVExporterNode,
    "ExcelExporter": ExcelExporterNode,
    "JSONExporter": JSONExporterNode,
}