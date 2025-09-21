"""
Node registry for the workflow engine.
"""

# Import all node implementations
from .loaders import LoadFromFolderNode, LoadSingleDocumentNode, LoadWebDocumentNode, LoadSpreadsheetNode
from .textsplitters import SplitByCharacterCountNode, SplitByParagraphNode, KeepFullDocumentNode
from .storage import ChromaStoreNode, WhooshStoreNode, ElasticsearchStoreNode
from .query import QueryWhooshStoreNode, QueryChromaStoreNode, QueryElasticsearchStoreNode, QueryDualStoreNode

# Document Transformer implementations
from .document_transformers import (AddMetadataNode, ContentPrefixNode, ContentSuffixNode, 
                                   DocumentFilterNode, PythonDocumentTransformerNode)

# Processor and Exporter implementations
from .processors import (PromptProcessorNode, ResponseCleanerNode, SummaryProcessorNode,
                        PythonDocumentProcessorNode, PythonResultProcessorNode,
                        AggregatorNode, PythonAggregatorNode, DocumentToResultsNode)
from .exporters import CSVExporterNode, ExcelExporterNode, JSONExporterNode, JSONResponseExporterNode


# Node Registry - maps node type names to node classes
NODE_REGISTRY = {
    # Loaders
    "LoadFromFolder": LoadFromFolderNode,
    "LoadSingleDocument": LoadSingleDocumentNode,
    "LoadWebDocument": LoadWebDocumentNode,
    "LoadSpreadsheet": LoadSpreadsheetNode,
    
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
    "QueryDualStore": QueryDualStoreNode,
    
    # Processors
    "PromptProcessor": PromptProcessorNode,
    "ResponseCleaner": ResponseCleanerNode,
    "SummaryProcessor": SummaryProcessorNode,
    "DocumentToResults": DocumentToResultsNode,
    "PythonDocumentProcessor": PythonDocumentProcessorNode,
    "PythonResultProcessor": PythonResultProcessorNode,
    "AggregatorNode": AggregatorNode,
    "PythonAggregatorNode": PythonAggregatorNode,
    
    # Exporters
    "CSVExporter": CSVExporterNode,
    "ExcelExporter": ExcelExporterNode,
    "JSONExporter": JSONExporterNode,
    "JSONResponseExporter": JSONResponseExporterNode,
}