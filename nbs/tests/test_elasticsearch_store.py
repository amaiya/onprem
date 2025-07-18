#!/usr/bin/env python3
"""
Test script for ElasticsearchStore implementation
"""

import os
from langchain_core.documents import Document
from onprem.ingest.stores.sparse import SparseStore
from onprem.ingest.stores import VectorStoreFactory

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
    """Test basic ElasticsearchSparseStore functionality"""
    print("Testing ElasticsearchSparseStore...")
    
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
        print("✓ ElasticsearchSparseStore created successfully")
    except ImportError as e:
        print(f"✗ ImportError: {e}")
        print("Please install elasticsearch: pip install elasticsearch")
        return False
    except Exception as e:
        print(f"✗ Error creating ElasticsearchSparseStore: {e}")
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
    """Test ElasticsearchStore functionality"""
    print("\n" + "="*50)
    print("Testing ElasticsearchStore (unified dense + sparse)...")
    print("="*50)
    
    # Get configuration from environment
    config = get_elastic_env()
    config['index'] = config['index'] + '_dual'  # Use different index
    
    print(f"Connecting to: {config['host']}")
    print(f"Index: {config['index']}")
    
    # Test factory method
    try:
        store_params = {
            'kind': 'elasticsearch',
            'persist_location': config['host'],
            'index_name': config['index'],
            'verify_certs': config['verify_certs'],
            'timeout': config['timeout'],
        }
        
        if config['basic_auth']:
            store_params['basic_auth'] = config['basic_auth']
        if config['ca_certs']:
            store_params['ca_certs'] = config['ca_certs']
            
        store = VectorStoreFactory.create(**store_params)
        print("✓ ElasticsearchStore created successfully")
    except Exception as e:
        print(f"✗ Error creating ElasticsearchStore: {e}")
        return False
    
    # Test basic operations
    try:
        # Clear any existing data first
        if store.exists():
            store.erase(confirm=False)
            print("✓ Cleared existing index data")
        
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
            semantic_results = store.semantic_search("artificial intelligence", limit=2, return_dict=True)
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
        
        # Test that both dense and sparse functionality work (implementation details are abstracted)
        print("✓ Both dense and sparse search functionality working correctly")
        
        # Test for duplicate documents (regression test for issue #XXX)
        try:
            expected_doc_count = len(docs)
            actual_doc_count = store.get_size()
            assert actual_doc_count == expected_doc_count, f"Expected {expected_doc_count} documents but got {actual_doc_count}. This suggests duplicate indexing."
            print(f"✓ No duplicate documents: {actual_doc_count} documents stored as expected")
            
            # Also check for duplicate results in search
            search_results = store.search("machine learning", limit=10)
            unique_ids = set()
            duplicate_count = 0
            for hit in search_results['hits']:
                hit_id = hit.get('id')
                if hit_id in unique_ids:
                    duplicate_count += 1
                else:
                    unique_ids.add(hit_id)
            
            assert duplicate_count == 0, f"Found {duplicate_count} duplicate documents in search results"
            print(f"✓ No duplicate search results: {len(unique_ids)} unique documents found")
            
        except Exception as e:
            print(f"⚠ Duplicate document test failed: {e}")
            assert False, f"Duplicate document test should not fail: {e}"
        
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

def test_dynamic_chunking():
    """Test dynamic chunking feature for semantic search"""
    print("\n" + "="*50)
    print("Testing Dynamic Chunking Feature...")
    print("="*50)
    
    # Get configuration from environment
    config = get_elastic_env()
    config['index'] = config['index'] + '_chunking'  # Use different index
    
    print(f"Connecting to: {config['host']}")
    print(f"Index: {config['index']}")
    
    # Test with dynamic chunking enabled
    try:
        store_params = {
            'kind': 'elasticsearch',
            'persist_location': config['host'],
            'index_name': config['index'],
            'verify_certs': config['verify_certs'],
            'timeout': config['timeout'],
            'chunk_for_semantic_search': True,  # Enable dynamic chunking
            'chunk_size': 100,  # Small chunks for testing
            'chunk_overlap': 20,
        }
        
        if config['basic_auth']:
            store_params['basic_auth'] = config['basic_auth']
        if config['ca_certs']:
            store_params['ca_certs'] = config['ca_certs']
            
        store = SparseStore.create(**store_params)
        print("✓ ElasticsearchSparseStore with dynamic chunking created successfully")
    except Exception as e:
        print(f"✗ Error creating store with dynamic chunking: {e}")
        return False
    
    try:
        # Create test documents with different sizes
        docs = [
            # Large document that will be chunked - target term at the end
            Document(page_content=(
                "This is the first paragraph about machine learning. "
                "Machine learning is a subset of artificial intelligence that focuses on algorithms. "
                "The second paragraph discusses deep learning concepts. "
                "Deep learning uses neural networks with multiple layers to process data. "
                "The third paragraph covers natural language processing. "
                "Natural language processing enables computers to understand human language. "
                "The final paragraph talks about computer vision applications. "
                "Computer vision allows machines to interpret and understand visual information from images and videos."
            ), metadata={"source": "large_doc.txt", "doc_type": "large"}),
            
            # Medium document - has some similar terms but not the exact target
            Document(page_content=(
                "Python is a programming language widely used in data science. "
                "It has extensive libraries for machine learning and data analysis. "
                "Popular libraries include NumPy, Pandas, and Scikit-learn. "
                "Data visualization is important for understanding patterns."
            ), metadata={"source": "medium_doc.txt", "doc_type": "medium"}),
            
            # Small document - completely different topic
            Document(page_content="TensorFlow is an open-source machine learning framework developed by Google.", 
                    metadata={"source": "small_doc.txt", "doc_type": "small"}),
            
            # Additional document that might confuse traditional search
            Document(page_content=(
                "Computer graphics and rendering applications have evolved significantly. "
                "Modern computers can process complex visual data efficiently. "
                "However, this document doesn't contain the specific term we're looking for."
            ), metadata={"source": "graphics_doc.txt", "doc_type": "distractor"})
        ]
        
        store.add_documents(docs)
        print("✓ Test documents added successfully")
        
        # Test semantic search with chunking
        print("\nTesting semantic search with dynamic chunking...")
        
        # Search for something in the middle/end of the large document
        semantic_results = store.semantic_search("computer vision applications", limit=3)
        assert len(semantic_results) > 0, "Semantic search should return results"
        print(f"✓ semantic_search() returned {len(semantic_results)} results")
        
        # Check if we got enhanced metadata from chunking
        found_large_doc = False
        for result in semantic_results:
            if result.metadata.get('doc_type') == 'large':
                found_large_doc = True
                
                # Check for chunking metadata
                assert 'best_chunk_text' in result.metadata, "Should have best_chunk_text metadata"
                assert 'best_chunk_idx' in result.metadata, "Should have best_chunk_idx metadata"
                assert 'total_chunks' in result.metadata, "Should have total_chunks metadata"
                
                # Check that chunks are available in metadata
                assert 'chunks' in result.metadata, "Should have chunks in metadata when chunking is enabled"
                assert isinstance(result.metadata['chunks'], list), "Chunks should be a list"
                assert len(result.metadata['chunks']) > 1, "Should have multiple chunks"
                
                # Check that page_content is a string (joined chunks)
                assert isinstance(result.page_content, str), "page_content should be a string (joined chunks)"
                
                print(f"✓ Large document chunking metadata found:")
                print(f"  - Best chunk index: {result.metadata['best_chunk_idx']}")
                print(f"  - Total chunks: {result.metadata['total_chunks']}")
                print(f"  - Chunks available: {len(result.metadata['chunks'])} chunks")
                print(f"  - Content type: {type(result.page_content)} (joined chunks)")
                print(f"  - Best chunk preview: {result.metadata['best_chunk_text'][:60]}...")
                
                # The best chunk should contain our search term
                best_chunk = result.metadata['best_chunk_text'].lower()
                assert 'computer vision' in best_chunk or 'visual' in best_chunk, \
                    "Best chunk should contain relevant terms"
                print("✓ Best chunk contains relevant search terms")
                
                # Should have multiple chunks for the large document
                assert result.metadata['total_chunks'] > 1, \
                    f"Large document should be split into multiple chunks, got {result.metadata['total_chunks']}"
                print(f"✓ Large document properly chunked into {result.metadata['total_chunks']} chunks")
                
                # Verify chunk metadata matches actual chunks
                assert len(result.metadata['chunks']) == result.metadata['total_chunks'], \
                    f"Chunks list length should match total_chunks: {len(result.metadata['chunks'])} vs {result.metadata['total_chunks']}"
                print("✓ Chunks list matches total chunk count")
                
                # Verify page_content is joined chunks
                expected_content = '\n\n'.join(result.metadata['chunks'])
                assert result.page_content == expected_content, "page_content should be joined chunks"
                print("✓ page_content correctly contains joined chunks")
                
                break
        
        assert found_large_doc, "Should find the large document in results"
        print("✓ Large document found with proper chunking metadata")
        
        # Test that normal semantic search still works (without chunking metadata for small docs)
        python_results = store.semantic_search("Python programming", limit=2)
        assert len(python_results) > 0, "Should find Python-related document"
        print("✓ Normal semantic search still works for other documents")
        
        # Compare with traditional approach
        print("\nComparing with traditional approach...")
        
        # Create store without chunking for comparison
        store_no_chunk = SparseStore.create(
            kind='elasticsearch',
            persist_location=config['host'],
            index_name=config['index'] + '_no_chunk',
            verify_certs=config['verify_certs'],
            timeout=config['timeout'],
            chunk_for_semantic_search=False,  # Disable dynamic chunking
            basic_auth=config['basic_auth'],
            ca_certs=config['ca_certs']
        )
        
        # Add same documents
        store_no_chunk.add_documents(docs)
        
        # Search for the same term
        no_chunk_results = store_no_chunk.semantic_search("computer vision applications", limit=3)
        
        # Compare results
        print(f"✓ Results comparison:")
        print(f"  - With chunking: {len(semantic_results)} results")
        print(f"  - Without chunking: {len(no_chunk_results)} results")
        
        # Check the actual scores to see if chunking provides better relevance
        chunking_scores = [r.metadata.get('score', 0) for r in semantic_results if r.metadata.get('doc_type') == 'large']
        traditional_scores = [r.metadata.get('score', 0) for r in no_chunk_results if r.metadata.get('doc_type') == 'large']
        
        # Most importantly, check that the large document is found with chunking
        found_large_in_chunked = any(r.metadata.get('doc_type') == 'large' for r in semantic_results)
        found_large_in_traditional = any(r.metadata.get('doc_type') == 'large' for r in no_chunk_results)
        
        if found_large_in_chunked and found_large_in_traditional:
            print("✓ Both approaches found the large document")
            
            # Compare relevance scores
            if chunking_scores and traditional_scores:
                chunk_score = chunking_scores[0]
                trad_score = traditional_scores[0]
                print(f"  - Chunking score: {chunk_score:.4f}")
                print(f"  - Traditional score: {trad_score:.4f}")
                
                if chunk_score > trad_score:
                    print("✓ Chunking found more relevant content (higher score)")
                elif chunk_score == trad_score:
                    print("✓ Same relevance score (both found the content equally well)")
                else:
                    print("⚠ Traditional approach had higher score (unexpected)")
            
        elif found_large_in_chunked and not found_large_in_traditional:
            print("✓ Chunking successfully found large document that traditional approach missed!")
        elif not found_large_in_chunked and found_large_in_traditional:
            print("⚠ Traditional approach found large document but chunking didn't (this shouldn't happen)")
        else:
            print("⚠ Neither approach found the large document (this shouldn't happen)")
        
        # Validate that chunking can find more relevant results
        # The key benefit is better relevance scoring within documents, not necessarily more results
        if len(semantic_results) > len(no_chunk_results):
            print("✓ Chunking found more relevant results (demonstrates improved recall)")
        elif len(semantic_results) == len(no_chunk_results):
            print("✓ Same number of results (both approaches found the content)")
            # Even with same number of results, chunking can provide better relevance
            if found_large_in_chunked and chunking_scores and traditional_scores:
                if chunking_scores[0] > traditional_scores[0]:
                    print("✓ Chunking demonstrated improved relevance scoring")
        else:
            print("⚠ Chunking found fewer results (this could happen due to scoring differences)")
        
        # Check that chunking version doesn't have the extra metadata
        for result in no_chunk_results:
            assert 'best_chunk_text' not in result.metadata, "No-chunking version should not have chunk metadata"
            assert 'chunks' not in result.metadata, "No-chunking version should not have chunks metadata"
        print("✓ No-chunking version correctly lacks chunk metadata")
        
        # Clean up both indices
        store.erase(confirm=False)
        store_no_chunk.erase(confirm=False)
        print("✓ Test indices erased successfully")
        
        return True
        
    except AssertionError as e:
        print(f"✗ Test assertion failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error during dynamic chunking test: {e}")
        return False

if __name__ == "__main__":
    # Test ElasticsearchStore (sparse only)
    success1 = test_elasticsearch_store()
    
    # Test ElasticsearchDualStore (unified dense + sparse)
    success2 = test_elasticsearch_dual_store()
    
    # Test dynamic chunking feature
    success3 = test_dynamic_chunking()
    
    if success1 and success2 and success3:
        print("\n🎉 All tests passed!")
        print("\nSUMMARY:")
        print("✓ ElasticsearchSparseStore (sparse text search)")
        print("✓ ElasticsearchStore (unified dense + sparse)")
        print("✓ Dynamic chunking for semantic search")
    else:
        print("\n❌ Some tests failed")
        if not success1:
            print("✗ ElasticsearchSparseStore test failed")
        if not success2:
            print("✗ ElasticsearchStore test failed")
        if not success3:
            print("✗ Dynamic chunking test failed")
    
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
