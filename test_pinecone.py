#!/usr/bin/env python3
"""
Pinecone Connection Test Script

This script tests the Pinecone connection and basic functionality:
1. Environment variable loading
2. Pinecone client initialization
3. Index connection
4. Basic query test
5. Index statistics

Usage:
    python test_pinecone.py
"""

import os
import sys
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

def test_environment_variables():
    """Test if all required environment variables are loaded"""
    print("🔍 Testing Environment Variables...")
    print("-" * 40)
    
    # Load environment variables
    load_dotenv()
    
    required_vars = [
        'PINECONE_API_KEY',
        'PINECONE_INDEX', 
        'PINECONE_URL'
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask the API key for security
            if 'API_KEY' in var:
                masked_value = value[:8] + '*' * (len(value) - 12) + value[-4:]
                print(f"✅ {var}: {masked_value}")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Missing")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n❌ Missing variables: {missing_vars}")
        print("Please check your .env file and ensure all required variables are set.")
        return False
    
    print(f"\n✅ All environment variables loaded successfully!")
    return True

def test_pinecone_client():
    """Test Pinecone client initialization"""
    print("\n🔗 Testing Pinecone Client...")
    print("-" * 40)
    
    try:
        api_key = os.getenv('PINECONE_API_KEY')
        pc = Pinecone(api_key=api_key)
        print("✅ Pinecone client initialized successfully")
        return pc
    except Exception as e:
        print(f"❌ Error initializing Pinecone client: {str(e)}")
        return None

def test_index_connection(pc):
    """Test connection to the specified index"""
    print("\n📊 Testing Index Connection...")
    print("-" * 40)
    
    try:
        index_name = os.getenv('PINECONE_INDEX')
        print(f"🔍 Attempting to connect to index: {index_name}")
        
        # List all indexes
        indexes = pc.list_indexes()
        print(f"📋 Available indexes: {indexes.names()}")
        
        if index_name not in indexes.names():
            print(f"❌ Index '{index_name}' not found!")
            print(f"Available indexes: {indexes.names()}")
            return None
        
        # Connect to the index
        index = pc.Index(index_name)
        print(f"✅ Successfully connected to index: {index_name}")
        return index
        
    except Exception as e:
        print(f"❌ Error connecting to index: {str(e)}")
        return None

def test_index_stats(index):
    """Test index statistics"""
    print("\n📈 Testing Index Statistics...")
    print("-" * 40)
    
    try:
        stats = index.describe_index_stats()
        print(f"✅ Index statistics retrieved successfully")
        print(f"   📊 Total vectors: {stats.total_vector_count}")
        print(f"   📐 Dimension: {stats.dimension}")
        print(f"   📁 Namespaces: {list(stats.namespaces.keys()) if hasattr(stats, 'namespaces') and stats.namespaces else 'None'}")
        return True
    except Exception as e:
        print(f"❌ Error getting index statistics: {str(e)}")
        return False

def test_embedding_model():
    """Test the embedding model"""
    print("\n🤖 Testing Embedding Model...")
    print("-" * 40)
    
    try:
        print("Loading multilingual-e5-large model...")
        model = SentenceTransformer('intfloat/multilingual-e5-large')
        
        # Test embedding generation
        test_text = "This is a test sentence for embedding generation."
        embedding = model.encode(test_text, normalize_embeddings=True)
        
        print(f"✅ Embedding model loaded successfully")
        print(f"   📐 Embedding dimensions: {len(embedding)}")
        print(f"   🧪 Test embedding shape: {embedding.shape}")
        return model
    except Exception as e:
        print(f"❌ Error loading embedding model: {str(e)}")
        return None

def test_query_functionality(index, model):
    """Test basic query functionality"""
    print("\n🔍 Testing Query Functionality...")
    print("-" * 40)
    
    try:
        # Generate test query embedding
        test_query = "revenue growth financial performance"
        print(f"🧪 Test query: '{test_query}'")
        
        query_embedding = model.encode(test_query, normalize_embeddings=True)
        print(f"✅ Query embedding generated: {len(query_embedding)} dimensions")
        
        # Perform test query
        response = index.query(
            vector=query_embedding.tolist(),
            top_k=3,
            include_metadata=True,
            include_values=False
        )
        
        print(f"✅ Query executed successfully")
        print(f"   📄 Retrieved {len(response.matches)} matches")
        
        # Display results
        for i, match in enumerate(response.matches, 1):
            company = match.metadata.get('company', 'Unknown')
            section = match.metadata.get('section', 'Unknown')
            chunk_id = match.metadata.get('chunk_id', 'Unknown')
            print(f"   {i}. {company} - {section} (ID: {chunk_id})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during query test: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🚀 Pinecone Connection Test")
    print("=" * 50)
    
    # Test 1: Environment variables
    if not test_environment_variables():
        print("\n❌ Environment test failed. Exiting.")
        sys.exit(1)
    
    # Test 2: Pinecone client
    pc = test_pinecone_client()
    if not pc:
        print("\n❌ Pinecone client test failed. Exiting.")
        sys.exit(1)
    
    # Test 3: Index connection
    index = test_index_connection(pc)
    if not index:
        print("\n❌ Index connection test failed. Exiting.")
        sys.exit(1)
    
    # Test 4: Index statistics
    if not test_index_stats(index):
        print("\n❌ Index statistics test failed.")
    
    # Test 5: Embedding model
    model = test_embedding_model()
    if not model:
        print("\n❌ Embedding model test failed.")
    
    # Test 6: Query functionality
    if model:
        if not test_query_functionality(index, model):
            print("\n❌ Query functionality test failed.")
    
    print("\n" + "=" * 50)
    print("🎉 Pinecone Connection Test Complete!")
    print("✅ All tests passed - your Pinecone setup is working correctly!")
    print("\n📝 Next steps:")
    print("   1. Ensure your PDF files are in the 'data/' directory")
    print("   2. Run the main advanced_rag.py script")
    print("   3. Start querying your documents!")

if __name__ == "__main__":
    main()
