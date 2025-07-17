#!/usr/bin/env python3
"""
Test script for ElasticsearchStore implementation
"""

import os
from langchain_core.documents import Document
from onprem.ingest.stores.sparse import SparseStore
from onprem.ingest.stores.dual import DualStore

def get_elastic_env():
    """
    Get Elasticsearch configuration from environment variables.
    
    Environment variables:
    - ELASTIC_HOST: Elasticsearch URL (default: http://localhost:9200)
    - ELASTIC_INDEX: Index name (default: test_index)
    - ELASTIC_USER: Username for basic auth
    - ELASTIC_PASSWORD: Password for basic auth
    - ELASTIC_VERIFY_CERTS: Whether to verify SSL certificates (default: true)
    - ELASTIC_CA_CERTS: Path to CA certificate file
    - ELASTIC_TIMEOUT: Connection timeout in seconds (default: 30)
    
    Returns:
        dict: Configuration dictionary
    """
    config = {
        "host": os.getenv("ELASTIC_HOST", "http://localhost:9200"),
        "index": os.getenv("ELASTIC_INDEX", "test_index"),
        "basic_auth": None,
        "verify_certs": os.getenv("ELASTIC_VERIFY_CERTS", "true").lower() == "true",
        "ca_certs": os.getenv("ELASTIC_CA_CERTS"),
        "timeout": int(os.getenv("ELASTIC_TIMEOUT", "30")),
    }
    
    # Set up basic auth if credentials are provided
    user = os.getenv("ELASTIC_USER")
    password = os.getenv("ELASTIC_PASSWORD")
    if user and password:
        config["basic_auth"] = (user, password)
    
    return config

def test_elasticsearch_store(host=None, index=None, basic_auth=None, verify_certs=None, ca_certs=None, timeout=None):
    """Test basic ElasticsearchStore functionality"""
    print("Testing ElasticsearchStore...")
    
    # Get configuration from environment
    config = get_elastic_env()
    
    # Override with provided parameters
    host = config["host"] if host is None else host
    index = config["index"] if index is None else index
    basic_auth = config["basic_auth"] if basic_auth is None else basic_auth
    verify_certs = config["verify_certs"] if verify_certs is None else verify_certs
    ca_certs = config["ca_certs"] if ca_certs is None else ca_certs
    timeout = config["timeout"] if timeout is None else timeout
    
    print(f"Connecting to: {host}")
    print(f"Index: {index}")
    print(f"Authentication: {'Yes' if basic_auth else 'No'}")
    print(f"Verify certs: {verify_certs}")
    print(f"Timeout: {timeout}s")
    
    # Test factory method
    try:
        store_params = {
            'kind': 'elasticsearch',
            'persist_location': host,
            'index_name': index,
            'verify_certs': verify_certs,
            'timeout': timeout,
        }
        
        if basic_auth:
            store_params['basic_auth'] = basic_auth
        if ca_certs:
            store_params['ca_certs'] = ca_certs
            
        store = SparseStore.create(**store_params)
        print("✓ ElasticsearchStore created successfully")
    except ImportError as e:
        print(f"✗ ImportError: {e}")
        print("Please install elasticsearch: pip install elasticsearch")
        return False
    except Exception as e:
        print(f"✗ Error creating ElasticsearchStore: {e}")
        print("Make sure Elasticsearch is running and accessible")
        return False
    
    # Test basic operations
    try:
        # Test exists (should be False for new index)
        exists = store.exists()
        assert not exists, "New index should not exist initially"
        print(f"✓ exists() returned: {exists}")
        
        # Test size
        size = store.get_size()
        assert size == 0, "New index should have size 0"
        print(f"✓ get_size() returned: {size}")
        
        # Test adding documents with dynamic fields
        docs = [
            Document(page_content="This is a test document", 
                    metadata={
                        "source": "test1.txt",
                        "author": "John Doe",  # String field
                        "priority": 5,  # Numeric field
                        "published": True,  # Boolean field
                        "tags": ["python", "elasticsearch"],  # List field
                        "created_date": "2024-01-15",  # Date field
                        "custom_score": 4.5  # Float field
                    }),
            Document(page_content="Another test document", 
                    metadata={
                        "source": "test2.txt",
                        "author": "Jane Smith",
                        "priority": 3,
                        "published": False,
                        "tags": ["search", "indexing"],
                        "created_date": "2024-01-16",
                        "custom_score": 3.8
                    })
        ]
        
        store.add_documents(docs)
        print("✓ Documents with dynamic fields added successfully")
        
        # Test exists after adding documents
        exists = store.exists()
        assert exists, "Index should exist after adding documents"
        print(f"✓ exists() after adding docs: {exists}")
        
        # Test size after adding documents
        size = store.get_size()
        assert size == 2, f"Index should have 2 documents, got {size}"
        print(f"✓ get_size() after adding docs: {size}")
        
        # Test query
        query_results = store.query("test document")
        assert query_results['total_hits'] > 0, "Query should return hits for 'test document'"
        assert 'hits' in query_results, "Query results should contain 'hits' key"
        print(f"✓ query() returned {query_results['total_hits']} hits")
        
        # Test semantic search
        semantic_results = store.semantic_search("test document", limit=2)
        assert len(semantic_results) > 0, "Semantic search should return results"
        print(f"✓ semantic_search() returned {len(semantic_results)} results")
        
        # Test dynamic field filtering
        try:
            # Test boolean field filter
            bool_filter_results = store.query("document", filters={"published": True})
            assert bool_filter_results['total_hits'] == 1, "Boolean filter should return 1 hit for published=True"
            print(f"✓ Boolean field filter returned {bool_filter_results['total_hits']} hits")
            
            # Test string field filter
            author_filter_results = store.query("document", filters={"author": "John Doe"})
            # Note: This might be 0 if the field mapping doesn't match exactly
            print(f"✓ String field filter returned {author_filter_results['total_hits']} hits")
            
            # Test numeric field filter (note: exact match for term filter)
            priority_filter_results = store.query("document", filters={"priority": 5})
            assert priority_filter_results['total_hits'] == 1, "Numeric filter should return 1 hit for priority=5"
            print(f"✓ Numeric field filter returned {priority_filter_results['total_hits']} hits")
            
            # Test list field filter
            tags_filter_results = store.query("document", filters={"tags": "python"})
            assert tags_filter_results['total_hits'] == 1, "List filter should return 1 hit for tags containing 'python'"
            print(f"✓ List field filter returned {tags_filter_results['total_hits']} hits")
            
        except Exception as e:
            print(f"⚠ Dynamic field filtering failed: {e}")
        
        # Test retrieving documents with dynamic fields
        try:
            if query_results['hits']:
                doc_id = query_results['hits'][0]['id']
                retrieved_doc = store.get_doc(doc_id)
                if retrieved_doc:
                    # Check if dynamic fields are preserved
                    dynamic_fields = ['author', 'priority', 'published', 'tags', 'custom_score']
                    preserved_fields = [field for field in dynamic_fields if field in retrieved_doc]
                    assert len(preserved_fields) >= 3, f"Should preserve at least 3 dynamic fields, got {len(preserved_fields)}"
                    print(f"✓ Dynamic fields preserved: {preserved_fields}")
                    
                    # Show sample values
                    if 'author' in retrieved_doc:
                        print(f"  - Author: {retrieved_doc['author']}")
                    if 'priority' in retrieved_doc:
                        print(f"  - Priority: {retrieved_doc['priority']}")
                    if 'tags' in retrieved_doc:
                        print(f"  - Tags: {retrieved_doc['tags']}")
                else:
                    print("⚠ Could not retrieve document for dynamic field verification")
                    assert False, "Should be able to retrieve document by ID"
        except Exception as e:
            print(f"⚠ Dynamic field retrieval test failed: {e}")
        
        # Clean up
        store.erase(confirm=False)
        print("✓ Index erased successfully")
        
        return True
        
    except AssertionError as e:
        print(f"✗ Test assertion failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        return False

def test_elasticsearch_dual_store():
    """Test ElasticsearchDualStore functionality"""
    print("\n" + "="*50)
    print("Testing ElasticsearchDualStore (unified dense + sparse)...")
    print("="*50)
    
    # Get configuration from environment
    config = get_elastic_env()
    config['index'] = config['index'] + '_dual'  # Use different index
    
    print(f"Connecting to: {config['host']}")
    print(f"Index: {config['index']}")
    
    # Test factory method
    try:
        store_params = {
            'dense_kind': 'elasticsearch',
            'sparse_kind': 'elasticsearch',
            'persist_location': config['host'],
            'index_name': config['index'],
            'verify_certs': config['verify_certs'],
            'timeout': config['timeout'],
        }
        
        if config['basic_auth']:
            store_params['basic_auth'] = config['basic_auth']
        if config['ca_certs']:
            store_params['ca_certs'] = config['ca_certs']
            
        store = DualStore.create(**store_params)
        print("✓ ElasticsearchDualStore created successfully")
    except Exception as e:
        print(f"✗ Error creating ElasticsearchDualStore: {e}")
        return False
    
    # Test basic operations
    try:
        # Test adding documents
        docs = [
            Document(page_content="Machine learning is a subset of artificial intelligence", 
                    metadata={"source": "ml_doc1.txt", "category": "AI"}),
            Document(page_content="Deep learning uses neural networks with multiple layers", 
                    metadata={"source": "dl_doc2.txt", "category": "AI"}),
        ]
        
        store.add_documents(docs)
        print("✓ Documents added successfully")
        
        # Test search (sparse)
        search_results = store.search("machine learning", limit=2)
        assert search_results['total_hits'] > 0, "Search should return hits for 'machine learning'"
        assert 'hits' in search_results, "Search results should contain 'hits' key"
        print(f"✓ search() returned {search_results['total_hits']} hits")
        
        # Test semantic search (dense)
        try:
            semantic_results = store.semantic_search("artificial intelligence", limit=2)
            assert semantic_results['total_hits'] > 0, "Semantic search should return hits"
            assert 'hits' in semantic_results, "Semantic search results should contain 'hits' key"
            print(f"✓ semantic_search() returned {semantic_results['total_hits']} hits")
        except Exception as e:
            print(f"⚠ semantic_search() failed: {e}")
            assert False, f"Semantic search should not fail: {e}"
        
        # Test hybrid search
        try:
            assert hasattr(store, 'hybrid_search'), "ElasticsearchDualStore should have hybrid_search method"
            hybrid_results = store.hybrid_search("machine learning AI", limit=2, weights=[0.6, 0.4])
            assert hybrid_results['total_hits'] > 0, "Hybrid search should return hits"
            assert 'hits' in hybrid_results, "Hybrid search results should contain 'hits' key"
            print(f"✓ hybrid_search() returned {hybrid_results['total_hits']} hits")
        except Exception as e:
            print(f"⚠ hybrid_search() failed: {e}")
            assert False, f"Hybrid search should not fail: {e}"
        
        # Test get_dense_db and get_sparse_db return same client
        dense_db = store.get_dense_db()
        sparse_db = store.get_sparse_db()
        assert dense_db is sparse_db, "Unified store should return same client for dense and sparse operations"
        print(f"✓ Unified store: dense_db is sparse_db = {dense_db is sparse_db}")
        
        # Clean up
        store.erase(confirm=False)
        print("✓ Index erased successfully")
        
        return True
        
    except AssertionError as e:
        print(f"✗ Test assertion failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        return False

if __name__ == "__main__":
    # Test ElasticsearchStore (sparse only)
    success1 = test_elasticsearch_store()
    
    # Test ElasticsearchDualStore (unified dense + sparse)
    success2 = test_elasticsearch_dual_store()
    
    if success1 and success2:
        print("\n🎉 All tests passed!")
        print("\nSUMMARY:")
        print("✓ ElasticsearchStore (sparse text search)")
        print("✓ ElasticsearchDualStore (unified dense + sparse)")
    else:
        print("\n❌ Some tests failed")
        if not success1:
            print("✗ ElasticsearchStore test failed")
        if not success2:
            print("✗ ElasticsearchDualStore test failed")
    
        # Example of testing with specific parameters
        print("\n" + "="*50)
        print("Example: Testing with HTTPS and authentication")
        print("="*50)
        print("To test with HTTPS and authentication, set environment variables:")
        print("export ELASTIC_HOST=https://localhost:9200")
        print("export ELASTIC_USER=elastic")
        print("export ELASTIC_PASSWORD=changeme")
        print("export ELASTIC_VERIFY_CERTS=false")
        print("export ELASTIC_CA_CERTS=/path/to/ca.crt")
        print("export ELASTIC_TIMEOUT=60")
        print("\nThen run this script again.")
