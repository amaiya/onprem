"""
Node registry for the workflow engine.
"""

# Import all node implementations
from .loaders import LoadFromFolderNode, LoadSingleDocumentNode, LoadWebDocumentNode
from .textsplitters import SplitByCharacterCountNode, SplitByParagraphNode, KeepFullDocumentNode
from .storage import ChromaStoreNode, WhooshStoreNode, ElasticsearchStoreNode
from .query import QueryWhooshStoreNode, QueryChromaStoreNode, QueryElasticsearchStoreNode


# Processor and Exporter implementations
from .processors import PromptProcessorNode, ResponseCleanerNode, SummaryProcessorNode
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
    
    # Exporters
    "CSVExporter": CSVExporterNode,
    "ExcelExporter": ExcelExporterNode,
    "JSONExporter": JSONExporterNode,
}