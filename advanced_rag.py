# -*- coding: utf-8 -*-
"""Copy of Advanced-RAG.ipynb

Original file is located at
    https://colab.research.google.com/drive/1xnG93cNB2Z38JizqopAVEle7Ra8C4AcZ

# 🔍 Advanced RAG Techniques

In this notebook, we'll go beyond the basics of Retrieval-Augmented Generation (RAG) and explore advanced techniques that significantly improve the quality of generated answers.

### 🧠 What we'll build:

We'll start by loading 10-K filings from multiple companies — **Amazon**, **Tesla**, **Nvidia**, and **Apple** — and store them in a **vector database**.

Then, we'll build a simple RAG pipeline and progressively apply the following advanced retrieval techniques:

- 🔄 **Re-ranking**: Reorder retrieved chunks based on relevance to improve answer quality.
- 🔗 **Multi-hop Retrieval**: Decompose complex questions and retrieve supporting information across multiple documents.
- 🧭 **Hybrid Search**: Combine sparse (keyword-based) and dense (embedding-based) retrieval for better recall.

> This notebook gives you a working playground — not just slides — to see how these techniques really perform on real-world financial filings.

**Note:** : Download the 10-K documents frm SEC - https://www.sec.gov/search-filings
"""

import os
from dotenv import load_dotenv
import re
from typing import Dict, List, Tuple
import uuid

# Load environment variables from .env file
load_dotenv()

# Verify environment variables are loaded
required_vars = ['PINECONE_API_KEY', 'PINECONE_INDEX', 'PINECONE_URL', 'DEEPSEEK_API_KEY','COHERE_API_KEY']

print("Environment Variables Status:")
print("-" * 30)
for var in required_vars:
    value = os.getenv(var)
    if value:
        print(f"✅ {var}: Set")
    else:
        print(f"❌ {var}: Missing")

# Check if all required variables are present
missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    print(f"\n❌ Missing variables: {missing_vars}")
    print("Please create a .env file and add all required variables")
else:
    print(f"\n🎉 All environment variables loaded successfully!")
    print(f"📋 Pinecone Index: {os.getenv('PINECONE_INDEX')}")


# Setup a data directory for PDFs
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"📁 Created directory: {DATA_DIR}")
    print(f"Please add your PDF files to the '{DATA_DIR}' directory.")

# Verify PDF files in the data directory
print(f"\n📂 Checking for PDF files in '{DATA_DIR}' directory...")
pdf_files = {}
expected_companies = ['Amazon', 'Apple', 'Nvidia', 'Tesla']

for filename in os.listdir(DATA_DIR):
    if filename.endswith('.pdf'):
        # Extract company name from filename
        company = filename.replace('.pdf', '').capitalize()
        if company in expected_companies:
            pdf_files[company.lower()] = os.path.join(DATA_DIR, filename)
            file_size = os.path.getsize(os.path.join(DATA_DIR, filename)) / (1024 * 1024)  # Size in MB
            print(f"✅ Found {filename} - {file_size:.1f} MB")

# Check if all expected files are present
PDF_FILES = pdf_files
total_files = len(PDF_FILES)

if total_files == 4:
    print(f"\n🎉 All {total_files} PDF files found successfully!")
    print("Ready to proceed with document loading and chunking.")
else:
    print(f"\n⚠️  Found {total_files}/4 expected files.")
    print("Please make sure Amazon.pdf, Apple.pdf, Nvidia.pdf, and Tesla.pdf are in the 'data' directory.")


"""## 📄 Chunking 10-K Documents with Metadata

Once we load the 10-K filings from Amazon, Tesla, Apple, and Nvidia, the next step is to **split the documents into manageable chunks**.

We'll use **LangChain's `RecursiveCharacterTextSplitter`** — a smart text splitter that breaks down text at sentence and paragraph boundaries while preserving semantic coherence.

### ⚙️ Why chunking?

- LLMs have context window limits — feeding entire documents isn't feasible.
- Smaller, overlapping chunks improve the chances of retrieving relevant content.
- Recursive splitting ensures clean, readable breaks in the middle of long documents.

### 🏷️ Metadata attached to each chunk

Each chunk will include the following metadata:

- **source**: file path or document origin
- **company**: Amazon, Tesla, Apple, or Nvidia
- **year**: extracted from the file name (e.g., `2023`)
- **chunk index**: useful for tracking and debugging

> These metadata fields become crucial later — especially for **filtering**, **multi-hop reasoning**, and **transparency in retrieval**.

Let's move on to splitting the text and attaching rich metadata to each chunk!

"""

# Cell 3: Load and Chunk Documents with Enhanced Metadata
# Install required packages with correct LangChain modules
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)

def extract_year_from_filename(filename: str) -> str:
    """Extract year from filename, default to 2023 if not found."""
    year_match = re.search(r'20\d{2}', filename)
    return year_match.group() if year_match else "2023"

def detect_section(text: str, page_num: int = None) -> str:
    """
    Detect 10K section based on text content and common section headers.
    Returns the most likely section name.
    """
    text_upper = text.upper()

    # Common 10K sections with their typical identifiers
    section_patterns = [
        ("Business", ["ITEM 1", "BUSINESS", "OUR BUSINESS", "THE BUSINESS"]),
        ("Risk Factors", ["ITEM 1A", "RISK FACTORS", "RISKS", "RISK FACTOR"]),
        ("Legal Proceedings", ["ITEM 3", "LEGAL PROCEEDINGS", "LITIGATION"]),
        ("Management Discussion", ["ITEM 7", "MD&A", "MANAGEMENT'S DISCUSSION", "MANAGEMENT DISCUSSION"]),
        ("Financial Statements", ["ITEM 8", "FINANCIAL STATEMENTS", "CONSOLIDATED STATEMENTS", "BALANCE SHEET"]),
        ("Controls and Procedures", ["ITEM 9A", "CONTROLS AND PROCEDURES", "INTERNAL CONTROL"]),
        ("Directors and Officers", ["ITEM 10", "DIRECTORS", "EXECUTIVE OFFICERS", "GOVERNANCE"]),
        ("Executive Compensation", ["ITEM 11", "EXECUTIVE COMPENSATION", "COMPENSATION"]),
        ("Security Ownership", ["ITEM 12", "SECURITY OWNERSHIP", "BENEFICIAL OWNERSHIP"]),
        ("Exhibits", ["ITEM 15", "EXHIBITS", "INDEX TO EXHIBITS"]),
    ]

    # Score each section based on keyword matches
    section_scores = {}
    for section_name, keywords in section_patterns:
        score = 0
        for keyword in keywords:
            if keyword in text_upper:
                score += text_upper.count(keyword)
        section_scores[section_name] = score

    # Return section with highest score, or "General" if no clear match
    best_section = max(section_scores.items(), key=lambda x: x[1])
    return best_section[0] if best_section[1] > 0 else "General"

def create_chunk_id(company: str, year: str, section: str, chunk_index: int) -> str:
    """Create a standardized chunk ID."""
    company_clean = company.lower().replace(" ", "_")
    section_clean = section.lower().replace(" ", "_").replace("'", "")
    return f"{company_clean}_{year}_{section_clean}_{chunk_index:02d}"

def get_source_doc_id(filename: str) -> str:
    """Extract clean document ID from filename."""
    # Remove path and clean up filename
    import os
    base_name = os.path.basename(filename)
    return base_name

def process_company_documents(company: str, filename: str) -> List[Document]:
    """Process a single company's 10K document with enhanced metadata."""
    print(f"\n📄 Processing {company.upper()}: {filename}")
    print("-" * 40)

    try:
        # Load PDF using PyMuPDFLoader
        loader = PyMuPDFLoader(filename)
        documents = loader.load()
        print(f"   ✅ Loaded {len(documents)} pages")

        # Extract metadata
        year = "2024"
        source_doc_id = get_source_doc_id(filename)

        company_chunks = []
        chunk_index = 0

        # Process each page separately to maintain page number tracking
        for page_num, doc in enumerate(documents, 1):
            page_content = doc.page_content
            page_chars = len(page_content)

            if page_chars < 50:  # Skip very short pages
                continue

            # Detect section for this page
            section = detect_section(page_content, page_num)

            # Split page into chunks
            page_chunks = text_splitter.split_text(page_content)

            # Create Document objects for each chunk
            for chunk_text in page_chunks:
                chunk_id = create_chunk_id(company, year, section, chunk_index)

                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "company": company,
                        "year": int(year),
                        "section": section,
                        "chunk_id": chunk_id,
                        "source_doc_id": source_doc_id,
                        "page_number": page_num,
                        "chunk_text": chunk_text,  # Explicit field as requested
                        # Additional useful metadata
                        "chunk_index": chunk_index,
                        "chunk_size": len(chunk_text),
                        "source_file": filename
                    }
                )

                company_chunks.append(chunk_doc)
                chunk_index += 1

        print(f"   ✂️  Created {len(company_chunks)} chunks across {len(documents)} pages")
        print(f"   📊 Total characters processed: {sum(len(doc.page_content) for doc in documents):,}")

        # Section summary
        sections_found = {}
        for chunk in company_chunks:
            section = chunk.metadata['section']
            sections_found[section] = sections_found.get(section, 0) + 1

        print(f"   📋 Sections detected: {', '.join(sections_found.keys())}")

        return company_chunks

    except Exception as e:
        print(f"   ❌ Error processing {filename}: {str(e)}")
        return []

# Main processing loop
all_documents = []
chunk_counts = {}
section_breakdown = {}

print("📚 Loading and chunking PDF documents with enhanced metadata...")
print("=" * 60)

for company, filename in PDF_FILES.items():
    company_chunks = process_company_documents(company, filename)

    if company_chunks:
        all_documents.extend(company_chunks)
        chunk_counts[company] = len(company_chunks)

        # Track sections per company
        company_sections = {}
        for chunk in company_chunks:
            section = chunk.metadata['section']
            company_sections[section] = company_sections.get(section, 0) + 1
        section_breakdown[company] = company_sections

        print(f"   ✅ {company.capitalize()}: {len(company_chunks)} chunks processed")
    else:
        chunk_counts[company] = 0

print("\n" + "=" * 60)
print("📊 ENHANCED PROCESSING SUMMARY")
print("=" * 60)

# Print chunks per company
for company, count in chunk_counts.items():
    print(f"📋 {company.capitalize()}: {count:,} chunks")
    if company in section_breakdown:
        for section, section_count in section_breakdown[company].items():
            print(f"   └── {section}: {section_count} chunks")

# Overall summary
total_chunks = len(all_documents)
total_companies = len([c for c in chunk_counts.values() if c > 0])

print(f"\n🎯 TOTALS:")
print(f"   📚 Total chunks: {total_chunks:,}")
print(f"   🏢 Companies processed: {total_companies}/{len(PDF_FILES)}")
if total_companies > 0:
    print(f"   📄 Average chunks per company: {total_chunks/total_companies:.0f}")

# Enhanced sample chunk inspection
if all_documents:
    sample_chunk = all_documents[0]
    print(f"\n🔍 SAMPLE CHUNK METADATA:")
    print(f"   Company: {sample_chunk.metadata['company']}")
    print(f"   Year: {sample_chunk.metadata['year']}")
    print(f"   Section: {sample_chunk.metadata['section']}")
    print(f"   Chunk ID: {sample_chunk.metadata['chunk_id']}")
    print(f"   Source Doc ID: {sample_chunk.metadata['source_doc_id']}")
    print(f"   Page Number: {sample_chunk.metadata['page_number']}")
    print(f"   Chunk Size: {sample_chunk.metadata['chunk_size']} characters")
    #print(f"   Content Preview: {sample_chunk.page_content[:150]}...")

# Validation: Show metadata structure matches your requirements
print(f"\n✅ METADATA VALIDATION:")
if all_documents:
    sample_metadata = all_documents[0].metadata
    required_fields = ["company", "year", "section", "chunk_id", "source_doc_id", "page_number", "chunk_text"]
    for field in required_fields:
        status = "✅" if field in sample_metadata else "❌"
        print(f"   {status} {field}: {sample_metadata.get(field, 'MISSING')}")

print(f"\n🚀 All documents loaded and chunked with enhanced metadata!")
print(f"📋 Ready for embedding generation with rich context information!")

# Cell 4: Generate Embeddings and Store in Pinecone with Requested Metadata
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Initialize embedding model
print("🤖 Loading multilingual-e5-large model...")
model = SentenceTransformer('intfloat/multilingual-e5-large')
print("✅ Model loaded successfully")

# Test embedding to verify dimensions
test_embedding = model.encode("test", normalize_embeddings=True)
print(f"📊 Embedding dimensions: {len(test_embedding)}")

# Initialize Pinecone
print("\n🔗 Connecting to Pinecone...")
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index_name = os.getenv('PINECONE_INDEX')

# Check if the index exists, create if it doesn't
if index_name not in pc.list_indexes().names():
    print(f"⚠️ Index '{index_name}' not found. Please create it in your Pinecone project.")
    # Example of how to create an index (adjust dimension as needed)
    # from pinecone import ServerlessSpec
    # pc.create_index(
    #     name=index_name,
    #     dimension=len(test_embedding),
    #     metric="cosine",
    #     spec=ServerlessSpec(
    #         cloud='aws',
    #         region='us-west-2'
    #     )
    # )
    # print(f"✅ Created index: {index_name}")

index = pc.Index(index_name)
print(f"✅ Connected to index: {index_name}")


# Generate embeddings and store in Pinecone
print("\n🚀 Generating embeddings and storing in Pinecone...")
print("=" * 60)

batch_size = 100  # Process in batches
total_stored = 0
company_stored = {}

for i in range(0, len(all_documents), batch_size):
    batch_docs = all_documents[i:i + batch_size]

    print(f"\n📦 Processing batch {i//batch_size + 1}/{(len(all_documents)-1)//batch_size + 1}")
    print(f"   📄 Documents {i+1}-{min(i+batch_size, len(all_documents))} of {len(all_documents)}")

    # Extract texts from batch
    texts = [doc.page_content for doc in batch_docs]

    # Generate embeddings
    print("   🤖 Generating embeddings...")
    embeddings = model.encode(texts, normalize_embeddings=True)

    # Prepare vectors for Pinecone
    vectors = []
    for doc, embedding in zip(batch_docs, embeddings):
        vector_id = str(uuid.uuid4())

        # Prepare metadata with requested fields
        metadata = {
            'company': doc.metadata['company'],
            'year': doc.metadata['year'],
            'section': doc.metadata.get('section', 'Financial Statements'),
            'chunk_id': f"{doc.metadata['company'].lower().replace(' (1)', '')}_{doc.metadata['year']}_financial_statements_{doc.metadata.get('chunk_id', str(i).zfill(2))}",
            'source_doc_id': doc.metadata['source_file'], # Corrected from 'source'
            'page_number': doc.metadata.get('page_number', 1),
            'chunk_size': f"{len(doc.page_content)} characters",
            'source': doc.metadata['source_file'], # Corrected from 'source'
            'chunk_text': doc.page_content  # Truncate to avoid size limits
        }

        vector = {
            'id': vector_id,
            'values': embedding.tolist(),
            'metadata': metadata
        }
        vectors.append(vector)

    # Store in Pinecone
    print("   📤 Uploading to Pinecone...")
    try:
        index.upsert(vectors=vectors)

        # Count by company
        for doc in batch_docs:
            company = doc.metadata['company']
            company_stored[company] = company_stored.get(company, 0) + 1

        total_stored += len(vectors)
        print(f"   ✅ Batch stored successfully ({len(vectors)} vectors)")

    except Exception as e:
        print(f"   ❌ Error storing batch: {str(e)}")

print("\n" + "=" * 60)
print("🎯 EMBEDDING & STORAGE SUMMARY")
print("=" * 60)

# Print storage by company
for company, count in company_stored.items():
    print(f"📋 {company.capitalize()}: {count:,} vectors stored")

print(f"\n📊 TOTALS:")
print(f"   🗄️  Total vectors stored: {total_stored:,}")
print(f"   🏢 Companies: {len(company_stored)}")
print(f"   📐 Embedding dimensions: {len(test_embedding)}")
print(f"   🤖 Model: intfloat/multilingual-e5-large")

# Verify index stats
try:
    print(f"\n🔍 Verifying Pinecone index...")
    stats = index.describe_index_stats()
    print(f"   📈 Total vectors in index: {stats.total_vector_count}")
    if hasattr(stats, 'namespaces') and stats.namespaces:
        print(f"   📁 Namespaces: {list(stats.namespaces.keys())}")
except Exception as e:
    print(f"   ⚠️  Could not retrieve index stats: {str(e)}")

if total_stored == len(all_documents):
    print(f"\n🎉 SUCCESS! All {total_stored} document chunks embedded and stored!")
    print("✅ Ready for RAG querying!")
else:
    print(f"\n⚠️  Stored {total_stored}/{len(all_documents)} chunks")
    print("Some chunks may have failed to store.")

# Get DeepSeek API key
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

if not deepseek_api_key:
    raise ValueError("DEEPSEEK_API_KEY not found in environment variables or .env file")

print("✅ DeepSeek API key loaded successfully")

# Try to import ChatDeepSeek from LangChain
try:
    from langchain_community.chat_models import ChatDeepSeek

    # Initialize DeepSeek LLM
    llm = ChatDeepSeek(
        model="deepseek-chat",
        deepseek_api_key=deepseek_api_key,
        temperature=0.1,
        streaming=False
    )

    print("✅ DeepSeek LLM initialized using ChatDeepSeek")
    print(f"🤖 Model: deepseek-chat")
    print(f"🌡️ Temperature: 0.1")
    print(f"📡 Streaming: False")

except ImportError:
    print("⚠️ ChatDeepSeek not available, trying custom OpenAI-compatible wrapper...")

    # Fallback: Use OpenAI wrapper with DeepSeek endpoint
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=deepseek_api_key,
        openai_api_base="https://api.deepseek.com/v1",
        temperature=0.1,
        streaming=False
    )

    print("✅ DeepSeek LLM initialized using OpenAI-compatible wrapper")
    print(f"🤖 Model: deepseek-chat")
    print(f"🌡️ Temperature: 0.1")
    print(f"📡 Streaming: False")
    print(f"🔗 API Base: https://api.deepseek.com/v1")

# Test the LLM with a simple query
try:
    test_response = llm.invoke("Hello! Please respond with 'DeepSeek LLM is working correctly.'")
    print(f"\n🧪 Test Response: {test_response.content}")
    print("✅ LLM is ready to use!")

except Exception as e:
    print(f"❌ Error testing LLM: {str(e)}")
    print("Please check your API key and internet connection.")

from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import SentenceTransformerEmbeddings
import numpy as np

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX")
index = pc.Index(index_name)

# Initialize embedding model
embedding_model = SentenceTransformerEmbeddings(
    model_name='intfloat/multilingual-e5-large'
)

# Create VectorStore
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding_model
)

# Create retriever with similarity search and k=5
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

print("✅ Setup complete! Pinecone retriever ready.")
print(f"📊 Index name: {index_name}")
print(f"🔍 Search type: similarity, k=5")
print(f"🤖 Embedding model: intfloat/multilingual-e5-large")

def get_rag_answer(query: str, retriever, llm) -> str:
    """
    Retrieve relevant chunks and generate answer using DeepSeek LLM

    Args:
        query: User's question
        retriever: Pinecone retriever or custom retrieval function
        llm: DeepSeek LLM instance

    Returns:
        Generated answer based on retrieved context
    """

    # Method 1: Try using LangChain retriever first
    try:
        query_embedding = embedding_model.embed_query(query)
        response = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            include_values=False
        )
        chunks = []
        metadata_list = []
        for match in response.matches:
            chunk_text = match.metadata.get('chunk_text', '')
            if chunk_text:
                chunks.append(chunk_text)
                metadata_list.append(match.metadata)
        print(f"✅ Retrieved {len(chunks)} chunks using direct Pinecone query")
        if not chunks:
          return "No relevant information found in the database."
    except Exception as e:
         print("Exception",e)
    # Print retrieved metadata for transparency
    print("\n📋 RETRIEVED CHUNKS METADATA:")
    print("-" * 50)
    for i, metadata in enumerate(metadata_list, 1):
        company = metadata.get('company', 'Unknown').replace(' (1)', '')
        year = metadata.get('year', 'Unknown')
        chunk_id = metadata.get('chunk_id', 'Unknown')
        source = metadata.get('section', 'Unknown')
        chunk_text=metadata.get('chunk_text', 'Unknown')
        print(f"Chunk {i}: {company.title()} ({year}) - ID: {chunk_id} - {source}")
        #print(f"Chunk text {i}: {chunk_text}")

    # Combine chunks into context
    context = "\n\n".join([f"Document {i+1}:\n{chunk}" for i, chunk in enumerate(chunks)])

    # Create prompt for DeepSeek
    prompt = f"""Based on the following documents, please answer the user's question accurately and comprehensively.

QUESTION: {query}

CONTEXT DOCUMENTS:
{context}

INSTRUCTIONS:
- Use only the information provided in the context documents
- If the information is not sufficient to answer the question, state this clearly
- Provide specific details and numbers when available
- Structure your answer clearly and concisely
- If data spans multiple years or sources, organize it logically

ANSWER:"""

    # Send to DeepSeek LLM
    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()

        print(f"\n🤖 DeepSeek LLM Response Generated ({len(answer)} characters)")
        return answer

    except Exception as e:
        return f"Error generating answer with DeepSeek: {str(e)}"

query="Summarize key points from Apple 10k."
final_answer=get_rag_answer(query,retriever,llm)
print(final_answer)

"""# 🔄 Re-ranking in Retrieval-Augmented Generation (RAG)

In a basic RAG pipeline, we retrieve the top-k chunks based on similarity scores from a vector database (typically using bi-encoder embeddings).  
But what if:

- The **most relevant** chunk isn't in the top 5?
- Two chunks are nearly identical?
- The top chunks are only **loosely related** to the actual question?

That's where **Re-ranking** becomes essential.

---

## 🧠 What is Re-ranking?

**Re-ranking** is the process of **reordering retrieved documents** based on a deeper, **semantic evaluation** of their relevance to the user's query.

Unlike vector similarity, which compares embeddings, re-ranking uses a **cross-encoder model** that **jointly reads the query and the document chunk** and assigns a **relevance score** between `0` and `1`.

> ✅ It helps you go from "close enough" to **spot on**.

---

## ⚙️ How Re-ranking Works — With Example

### 🧪 Input Query:
> 💬 **"What was Tesla's strategy for cost reduction in 2024?"**

### 🔍 Initial Vector Retrieval Returns:

1. Chunk about 2024 vehicle production  
2. Chunk about 2023 factory output  
3. ✅ Chunk about cost-reduction initiatives *(but ranked #3!)*  
4. Chunk about Elon Musk quotes  
5. Chunk about energy storage

➡️ These chunks were retrieved based on **vector similarity**, not deep understanding.

---

### 🔁 Re-ranking Step:

The re-ranker takes the **query + each chunk** as a **pair**, like:

[QUERY]: What was Tesla's strategy for cost reduction in 2024?
[CHUNK]: In 2024, Tesla reduced costs by optimizing gigafactory processes...

It processes the pair **jointly**, and assigns a score like:

| Chunk Snippet                                             | Score |
|-----------------------------------------------------------|-------|
| Cost-reduction initiatives                                | 0.95  |
| 2024 vehicle production                                   | 0.62  |
| Elon Musk quotes                                          | 0.45  |

Then it **sorts the list** so the most relevant chunk comes first.

---

## 🔀 What is a Cross Encoder?

A **Cross Encoder** is a transformer model (like BERT) that takes two texts — usually a **query and a document** — and processes them **together** as a single sequence:

[CLS] <query> [SEP] <document> [SEP]


This enables **deep token-level interaction** between the query and document.

### Example:
- Query: `"When was Tesla founded?"`
- Document: `"Tesla was founded in 2003 by engineers in California."`

The model reads both together and outputs a **relevance score**.

---

## 🔄 Cross Encoder vs. Bi-Encoder

| Feature            | Cross Encoder                              | Bi-Encoder                                  |
|--------------------|---------------------------------------------|---------------------------------------------|
| Encoding Style     | Joint (query + doc as one input)            | Independent (separate embeddings)           |
| Accuracy           | ✅ High                                      | ❌ Lower                                     |
| Speed              | ❌ Slow (no pre-computation)                | ✅ Fast (pre-computed embeddings)           |
| Scalability        | ❌ Not ideal for large corpora              | ✅ Excellent for large-scale retrieval       |
| Use Case           | Re-ranking, QA, similarity scoring          | Initial retrieval, semantic search          |

> Retrieval from **Pinecone** or FAISS is an example of a **bi-encoder** system.

---

## 🧠 How a Re-ranker Works (Step-by-Step)

1. 🔍 Retrieve top-k chunks from vector DB (bi-encoder)
2. 📦 Create `[query, chunk]` pairs for each chunk
3. 🧠 Use a **cross-encoder model** (like Cohere Re-ranker) to score each pair
4. 📊 Sort by score to rank the most relevant chunks higher

---

### Example:

> ❓ Query: *"What was Apple's AI strategy in 2023?"*

Candidate chunk:
> 📄 "In 2023, Apple launched AI-powered features in iOS, including privacy-preserving personalization."

The re-ranker compares the two, understands they are related, and gives it a **high relevance score**, e.g., `0.92`.

Chunks that merely mention "Apple" and "2023" but **don't explain strategy** will score lower.

---

## 🎯 Why Re-ranking Improves RAG Quality

| Bi-Encoder (Retriever) | Cross Encoder (Re-ranker)                |
|------------------------|------------------------------------------|
| Surface similarity     | Deep contextual relevance                |
| Fast but shallow       | Slower but smarter                      |
| May miss best chunks   | Prioritizes truly relevant answers       |

Think of it like this:

- 🧭 **Retriever** = map to possible locations  
- 👩‍⚖️ **Re-ranker** = expert who reads them and says "this one has the answer"

---

## 🤖 What is the Cohere Re-ranker?

The **Cohere Re-ranker** is a production-ready cross-encoder model designed for **semantic re-ranking**.

You provide:
- A **query**
- A list of **retrieved document chunks**

It returns:
- A **sorted list**, with scores indicating how well each chunk answers the query.

---

### ⚙️ Model Details

| Property         | Description                        |
|------------------|------------------------------------|
| Model Name       | `rerank-english-v3.0`              |
| Developer        | [Cohere](https://cohere.com)       |
| Input            | Query + up to 100 chunks           |
| Output           | Sorted list with relevance scores  |
| Max Tokens       | ~4096                              |
| Pricing          | Paid API                           |
| Language         | English                            |

---

### ✅ Why Use It?

- Out-of-the-box performance (no fine-tuning required)
- Supports long, dense documents (like 10-Ks)
- Boosts accuracy in answers generated by the LLM

---

> In your notebook, you'll pair each chunk with the query, pass it to Cohere's re-ranker, and get a better-ordered list for downstream LLM generation.


"""

import cohere

def get_rag_answer_with_cohere_rerank(query: str, retriever, llm) -> str:
    """
    Retrieve relevant chunks, re-rank them using Cohere, and generate answer using DeepSeek LLM

    Args:
        query: User's question
        retriever: Pinecone retriever or custom retrieval function
        llm: DeepSeek LLM instance

    Returns:
        Generated answer based on re-ranked retrieved context
    """

    # Initialize Cohere client
    try:
        cohere_api_key = os.getenv("COHERE_API_KEY")
        if not cohere_api_key:
            return "COHERE_API_KEY not found in environment variables"

        co = cohere.Client(cohere_api_key)
    except Exception as e:
        return f"Error initializing Cohere client: {str(e)}"

    # Method 1: Retrieve the chunks
    try:
        query_embedding = embedding_model.embed_query(query)
        response = index.query(
            vector=query_embedding,
            top_k=10,  # Get more chunks initially for re-ranking
            include_metadata=True,
            include_values=False
        )
        chunks = []
        metadata_list = []
        documents_for_rerank = []

        for match in response.matches:
            chunk_text = match.metadata.get('chunk_text', '')
            if chunk_text:
                chunks.append(chunk_text)
                metadata_list.append(match.metadata)
                documents_for_rerank.append(chunk_text)

        print(f"✅ Retrieved {len(chunks)} chunks using direct Pinecone query")

        if not chunks:
            return "No relevant information found in the database."

    except Exception as e:
        print("Exception", e)
        return f"Error during retrieval: {str(e)}"

    # Print chunks BEFORE re-ranking
    print("\n📋 CHUNKS BEFORE RE-RANKING:")
    print("-" * 50)
    for i, metadata in enumerate(metadata_list, 1):
        company = metadata.get('company', 'Unknown').replace(' (1)', '')
        year = metadata.get('year', 'Unknown')
        chunk_id = metadata.get('chunk_id', 'Unknown')
        source = metadata.get('section', 'Unknown')
        print(f"Chunk {i}: {company.title()} ({year}) - ID: {chunk_id} - {source}")

    # Re-rank using Cohere
    try:
        rerank_response = co.rerank(
            model='rerank-english-v3.0',
            query=query,
            documents=documents_for_rerank,
            top_n=5,
            return_documents=True
        )

        # Get re-ranked chunks and their metadata
        reranked_chunks = []
        reranked_metadata = []

        for result in rerank_response.results:
            original_index = result.index
            reranked_chunks.append(chunks[original_index])
            reranked_metadata.append(metadata_list[original_index])

        print(f"✅ Re-ranked to top {len(reranked_chunks)} most relevant chunks")

    except Exception as e:
        print(f"Exception during re-ranking: {e}")
        # Fallback to original chunks if re-ranking fails
        reranked_chunks = chunks[:5]
        reranked_metadata = metadata_list[:5]

    # Print chunks AFTER re-ranking
    print("\n📋 CHUNKS AFTER RE-RANKING:")
    print("-" * 50)
    for i, metadata in enumerate(reranked_metadata, 1):
        company = metadata.get('company', 'Unknown').replace(' (1)', '')
        year = metadata.get('year', 'Unknown')
        chunk_id = metadata.get('chunk_id', 'Unknown')
        source = metadata.get('section', 'Unknown')
        chunk_text = metadata.get('chunk_text', 'Unknown')
        print(f"Chunk {i}: {company.title()} ({year}) - ID: {chunk_id} - {source}")
        #print(f"Chunk text {i}: {chunk_text}")

    # Combine chunks into context
    context = "\n\n".join([f"Document {i+1}:\n{chunk}" for i, chunk in enumerate(reranked_chunks)])

    # Create prompt for DeepSeek
    prompt = f"""Based on the following documents, please answer the user's question accurately and comprehensively.

QUESTION: {query}

CONTEXT DOCUMENTS:
{context}

INSTRUCTIONS:
- Use only the information provided in the context documents
- If the information is not sufficient to answer the question, state this clearly
- Provide specific details and numbers when available
- Structure your answer clearly and concisely
- If data spans multiple years or sources, organize it logically

ANSWER:"""

    # Send to DeepSeek LLM
    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()

        print(f"\n🤖 DeepSeek LLM Response Generated ({len(answer)} characters)")
        return answer

    except Exception as e:
        return f"Error generating answer with DeepSeek: {str(e)}"

def evaluate_answers(answer1: str, answer2: str, llm, query: str = None) -> str:
    """
    Evaluate and compare two answers using LLM to determine which is better

    Args:
        answer1: Answer generated without re-ranking
        answer2: Answer generated with Cohere re-ranking
        llm: DeepSeek LLM instance for evaluation
        query: Original query (optional, for context)

    Returns:
        Detailed comparison and evaluation from the LLM
    """

    # Create evaluation prompt
    prompt = f"""You are an expert evaluator tasked with comparing two AI-generated answers to determine which one is better. Please analyze both answers carefully and provide a detailed comparison.

{f"ORIGINAL QUERY: {query}" if query else ""}

ANSWER 1 (Without Re-ranking):
{answer1}

ANSWER 2 (With Re-ranking):
{answer2}

EVALUATION CRITERIA:
Please evaluate both answers based on the following criteria and provide a detailed analysis:

1. **ACCURACY & FACTUAL CORRECTNESS**
   - Which answer contains more accurate information?
   - Are there any factual errors or inconsistencies?

2. **COMPLETENESS & COMPREHENSIVENESS**
   - Which answer provides more complete coverage of the topic?
   - Does one answer miss important aspects that the other covers?

3. **RELEVANCE & FOCUS**
   - Which answer stays more focused on the specific question asked?
   - Does one contain more irrelevant or tangential information?

4. **CLARITY & ORGANIZATION**
   - Which answer is clearer and easier to understand?
   - How well is the information structured and organized?

5. **SPECIFIC DETAILS & EVIDENCE**
   - Which answer provides more specific details, numbers, or concrete examples?
   - How well does each answer support its claims with evidence?

6. **OVERALL QUALITY & USEFULNESS**
   - Which answer would be more helpful to someone seeking this information?
   - Consider the practical value and actionability of each response.

COMPARISON FORMAT:
Please structure your evaluation as follows:

**WINNER: [Answer 1 / Answer 2 / Tie]**

**DETAILED ANALYSIS:**

**Accuracy & Factual Correctness:**
- Answer 1: [Analysis]
- Answer 2: [Analysis]
- Winner: [Answer 1/Answer 2/Tie] - [Brief reason]

**Completeness & Comprehensiveness:**
- Answer 1: [Analysis]
- Answer 2: [Analysis]
- Winner: [Answer 1/Answer 2/Tie] - [Brief reason]

**Relevance & Focus:**
- Answer 1: [Analysis]
- Answer 2: [Analysis]
- Winner: [Answer 1/Answer 2/Tie] - [Brief reason]

**Clarity & Organization:**
- Answer 1: [Analysis]
- Answer 2: [Analysis]
- Winner: [Answer 1/Answer 2/Tie] - [Brief reason]

**Specific Details & Evidence:**
- Answer 1: [Analysis]
- Answer 2: [Analysis]
- Winner: [Answer 1/Answer 2/Tie] - [Brief reason]

**KEY DIFFERENCES:**
- [List 3-5 most significant differences between the answers]

**FINAL VERDICT:**
- Overall Winner: [Answer 1/Answer 2/Tie]
- Confidence Level: [High/Medium/Low]
- Main Reasons: [2-3 key reasons for the decision]

**RECOMMENDATIONS:**
- [Suggestions for improving the weaker answer or both answers]

Be objective, thorough, and specific in your analysis. Focus on concrete differences rather than general statements."""

    # Send to LLM for evaluation
    try:
        response = llm.invoke(prompt)
        evaluation = response.content.strip()

        print(f"\n🔍 Answer Evaluation Completed ({len(evaluation)} characters)")
        print("\n" + "="*80)
        print("📊 ANSWER COMPARISON EVALUATION")
        print("="*80)
        print(evaluation)
        print("="*80)

        return evaluation

    except Exception as e:
        return f"Error during answer evaluation: {str(e)}"

query="Summarize Amazon R&D spending in 2024"
answer=get_rag_answer(query,retriever,llm)
print(answer)
answer_with_rerank=get_rag_answer_with_cohere_rerank(query,retriever,llm)
print(answer_with_rerank)

evaluate_answers(answer, answer_with_rerank, llm)

"""## ⚖️ When Should You Use a Re-ranker?

Re-rankers can dramatically improve the **quality** of answers in RAG — but they're not always the right fit depending on your use case.

Let's look at the **trade-offs**.

---

### ✅ Good Use Cases for Re-ranking

Use a re-ranker when:

- 🧠 **Accuracy matters more than speed**  
  Great for tasks like:
  - Legal document Q&A
  - Financial filings
  - Medical summaries
  - Internal knowledge assistants

- 🔍 **Your retriever brings back noisy or similar-looking chunks**  
  Re-rankers help filter out redundancy and surface the most precise content.

- 🔄 **You want to combine multiple signals (keyword match + semantic meaning)**  
  Cross-encoders consider both surface and deep meaning — not just embeddings.

- 🧪 **You have a small number of retrieved chunks (like top-10)**  
  Re-rankers are slower, so they work best when evaluating a **small candidate set**.

---

### 🚫 When Re-ranking May Not Be Ideal

Avoid re-ranking (or use carefully) if:

- ⚡ **You need ultra-low-latency responses**  
  Re-ranking is slower because each query-chunk pair goes through a transformer model.

- 💰 **You're operating at large scale with cost concerns**  
  Running a cross-encoder on thousands of queries or hundreds of chunks can get expensive.

- 🏗️ **You already have strong domain-specific retrieval logic**  
  In highly tuned systems, re-ranking may offer only marginal gains.

- 📦 **You're retrieving from very short or very structured data**  
  Re-ranking shines in complex, messy, or multi-paragraph chunks — less useful for short, exact matches.

---

### 💡 Best Practice

> ✅ Use a fast retriever (vector or hybrid) to pull top-10 candidates, then apply a re-ranker on just those.

This gives you the **best of both worlds**:
- Speed from vector search
- Quality from deep semantic scoring

---

## 🔗 Multi-Hop Retrieval with Structured Reasoning

This notebook includes a powerful implementation of **Multi-Hop Retrieval-Augmented Generation (MultiHop-RAG)** — enhanced with **structured reasoning**, **missing information detection**, and **explicit metadata tracking**.

Instead of retrieving once and hoping for the best, this method performs **multiple reasoning-driven hops** to find missing facts and connect the dots across documents.

---

### 🧠 Why Multi-Hop Retrieval?

Single-hop RAG retrieves once, based on the user query. But **some questions are too complex** to be answered from a single chunk.

Example:
> ❓ "How did Apple's R&D spending change over the past 3 years, and what impact did it have on AI product launches?"

Answering this might need:
- R&D amounts from 2021, 2022, and 2023 (across multiple documents)
- Evidence of AI product launches and internal commentary

That's where **Multi-Hop RAG** shines.

---

### 🔄 What this Implementation Does

The function `get_multihop_rag_answer()` breaks down your original query into multiple steps:

1. **Initial Query (Q1)** is used to retrieve top-K chunks.
2. The model generates:
   - ✅ Key **insights** from the chunks
   - 🧠 Step-by-step **reasoning**
   - ❓ What's still **missing**
3. A **sub-question (Q2)** is generated to target that gap.
4. Steps 1–3 repeat for `max_hops`, collecting new insights each time.
5. A final answer is synthesized from:
   - All retrieved documents
   - All reasoning steps
   - Source metadata

---

### 🧩 What Makes This Special?

This Multi-Hop RAG includes:

| Component                          | Description |
|-----------------------------------|-------------|
| ✅ Structured Reasoning (`R1, R2`) | Tracks what was learned and how it connects to the question |
| 🔍 Sub-question Generation         | LLM creates targeted follow-ups based on gaps |
| 🧹 Deduplication with Metadata     | Same chunk isn't reused across hops |
| ✂️ Truncation                     | Avoids prompt bloat using `chunk_word_limit` |
| 📋 Final Output                   | Includes reasoning trace, answer, and document sources |

---

### 📊 Example Output Format

🧠 MULTI-HOP REASONING TRACE:
Q1: What was Apple's R&D spending from 2021 to 2023?
A1: Retrieved 3 documents – Found value: $29.9B
R1: This answers the 2023 part. Still missing 2022/2021.

Q2: What was Apple's R&D in 2021 and 2022?
A2: Retrieved 3 documents – Found $21.9B (2021), $27.5B (2022)
R2: Now we have all 3 years.

Q3: What AI products did Apple release?
A3: Found multiple references to iOS features using on-device AI.

🎯 SYNTHESIZED ANSWER:
Apple's R&D spending rose from $21.9B to $29.9B (2021–2023), which coincided with an increase in AI-powered features…

📋 DOCUMENT SOURCES:

Apple (Chunk 005, Hop 1)

Apple (Chunk 022, Hop 2)

Apple (Chunk 041, Hop 3)
---
"""

# Enhanced MultiHop-RAG Implementation with Structured Reasoning - FIXED VERSION
"""
Enhanced MultiHop-RAG system following the MultiHop-RAG paper methodology.

This implementation provides:
- Structured reasoning traces (Q1, A1, R1, Q2, A2, R2, ...)
- Missing information identification and targeting
- Document metadata tracking and transparency
- Token-efficient truncation and filtering
- Clear separation of reasoning trace and final answer

Key Components:
- Main function: get_multihop_rag_answer()
- Reasoning: _generate_structured_reasoning()
- Sub-question generation: _generate_next_subquestion_from_missing()
- Document processing: _truncate_documents(), _remove_duplicates_with_metadata()
- Final synthesis: _generate_final_answer_structured()

Author: Assistant
Date: 2024
"""

previous_context = ""

def get_multihop_rag_answer(query: str, llm, max_hops=5, docs_per_hop=5, chunk_word_limit=500) -> str:
    """
    Multi-hop retrieval with structured reasoning steps (enhanced MultiHop-RAG).

    This is the main function that orchestrates the entire multi-hop retrieval process.
    It performs iterative retrieval, reasoning, and sub-question generation to comprehensively
    answer complex questions that may require information from multiple sources.

    Args:
        query (str): The original user question to be answered
        llm: DeepSeek LLM instance for reasoning and query generation
        max_hops (int, optional): Maximum number of retrieval hops to perform. Defaults to 5.
                                 Each hop consists of query → retrieve → reason → next query
        docs_per_hop (int, optional): Number of top documents to retrieve per hop. Defaults to 5.
                                     Controls retrieval breadth vs. focus
        chunk_word_limit (int, optional): Maximum words per document chunk to prevent prompt bloat.
                                         Defaults to 500. Helps manage token usage

    Returns:
        str: Comprehensive answer formatted with:
             - Multi-hop reasoning trace (Q1, A1, R1, Q2, A2, R2, ...)
             - Synthesized final answer
             - Document source metadata for transparency
    """

    print("🔍 ENHANCED MULTIHOP-RAG WITH STRUCTURED REASONING")
    print("=" * 60)
    print(f"🎯 Max hops: {max_hops} | Docs per hop: {docs_per_hop} | Word limit: {chunk_word_limit}")

    try:
        # Validate LLM
        if llm is None:
            raise ValueError("LLM is None. Please ensure the LLM is properly initialized.")

        all_retrieved_docs = []
        current_query = query
        reasoning_trace = {
            'hops': [],  # Structured reasoning for each hop
            'summary': ''  # Overall reasoning summary
        }

        # Multi-hop retrieval process with structured reasoning
        for hop in range(max_hops):
            hop_num = hop + 1
            print(f"\n🔄 HOP {hop_num}")
            print("-" * 50)

            # Step 1: Display current query
            if hop == 0:
                print(f"📝 Q{hop_num} (Original): {current_query}")
            else:
                print(f"📝 Q{hop_num} (Sub-question): {current_query}")

            # Step 2: Retrieve documents for current query
            hop_docs = _retrieve_documents_simple(current_query, top_k=docs_per_hop)
            print(f"📄 Retrieved {len(hop_docs)} documents")

            # Step 3: Truncate documents and add metadata
            truncated_docs = _truncate_documents(hop_docs, chunk_word_limit, hop_num)
            all_retrieved_docs.extend(truncated_docs)

            # Step 4: Display retrieved document metadata
            print(f"📋 A{hop_num} (Retrieved Documents):")
            for i, doc in enumerate(truncated_docs, 1):
                metadata = doc.metadata
                company = metadata.get('company', 'Unknown').replace(' (1)', '')
                chunk_id = metadata.get('chunk_id', 'Unknown')
                source = metadata.get('source', 'Unknown')
                hop = metadata.get('hop', '?')
                print(f"   {i}. {company} | Chunk: {chunk_id} | Source: {source} | Hop: {hop}")

            # Step 5: Generate structured reasoning step
            print(f"\n💭 R{hop_num} (Reasoning):")
            reasoning_step = _generate_structured_reasoning(
                query, current_query, truncated_docs, reasoning_trace, llm, hop_num
            )

            # Store structured reasoning
            hop_reasoning = {
                'hop': hop_num,
                'question': current_query,
                'retrieved_docs': len(truncated_docs),
                'reasoning': reasoning_step['reasoning'],
                'missing_info': reasoning_step['missing_info'],
                'insights': reasoning_step['insights']
            }
            reasoning_trace['hops'].append(hop_reasoning)

            print(f"   Insights: {reasoning_step['insights']}")
            print(f"   Reasoning: {reasoning_step['reasoning']}")
            print(f"   Still Missing: {reasoning_step['missing_info']}")

            # Step 6: Generate next sub-question based on missing info (if not last hop)
            if hop < max_hops - 1 and reasoning_step['missing_info'].lower() not in ['none', 'nothing', 'no missing information']:
                # FIXED: Added current_query as the second argument
                current_query = _generate_next_subquestion_from_missing(
                    query, current_query, reasoning_step['missing_info'], llm, hop_num
                )
                print(f"\n➡️ Next sub-question generated from missing info")
            elif hop < max_hops - 1:
                print(f"\n✅ Sufficient information found, but continuing to hop {hop_num + 1} for completeness")
                break

        print(f"\n📊 RETRIEVAL SUMMARY:")
        print(f"Total documents retrieved: {len(all_retrieved_docs)}")

        # Remove duplicates while preserving hop information
        unique_docs = _remove_duplicates_with_metadata(all_retrieved_docs)
        print(f"Unique documents: {len(unique_docs)}")

        # Generate final answer with structured reasoning and metadata
        final_answer = _generate_final_answer_structured(query, unique_docs, reasoning_trace, llm)

        print(f"✅ Enhanced MultiHop-RAG Answer Generated")
        print("=" * 60)

        return final_answer

    except Exception as e:
        error_msg = f"Error in Enhanced MultiHop-RAG: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg

def _generate_structured_reasoning(original_query: str, current_query: str, hop_docs, reasoning_trace, llm, hop_num: int) -> dict:
    """
    Generate structured reasoning with explicit missing information identification.
    """

    # Prepare context from current hop documents
    hop_context = ""
    if hop_docs:
        hop_context = "\n".join([
            f"Doc {i+1}: {doc.page_content[:200]}..."
            for i, doc in enumerate(hop_docs)
        ])
    else:
        hop_context = "No relevant documents found in this hop."

    # Prepare previous reasoning context
    previous_insights = []
    if reasoning_trace['hops']:
        for i, hop_data in enumerate(reasoning_trace['hops']):
            previous_insights.append(f"Q{i+1}: {hop_data['question']}")
            previous_insights.append(f"A{i+1}: {hop_data['insights']}")
            previous_insights.append(f"R{i+1}: {hop_data['reasoning']}")

    previous_context = "\n".join(previous_insights) if previous_insights else "None (this is the first hop)"

    reasoning_prompt = f"""Analyze the retrieved documents and provide structured reasoning for this multi-hop retrieval step, the data available only for 2024.

ORIGINAL QUESTION: {original_query}
CURRENT SUB-QUESTION (Q{hop_num}): {current_query}

PREVIOUS REASONING TRACE:
{previous_context}

RETRIEVED DOCUMENTS (A{hop_num}):
{hop_context}

Provide a structured analysis with three components:

1. INSIGHTS: What key information did you learn from these documents? (1-2 sentences)
2. REASONING: How does this information help answer the original question? (1-2 sentences)
3. MISSING_INFO: What specific information is still missing to fully answer the original question? Be explicit. If nothing is missing, say "No missing information" (1 sentence)

Format your response as:
INSIGHTS: [your insights]
REASONING: [your reasoning]
MISSING_INFO: [what's still missing or "No missing information"]"""

    try:
        response = llm.invoke(reasoning_prompt)
        content = response.content.strip()

        # Parse structured response
        insights = ""
        reasoning = ""
        missing_info = ""

        lines = content.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith('INSIGHTS:'):
                current_section = 'insights'
                insights = line.replace('INSIGHTS:', '').strip()
            elif line.startswith('REASONING:'):
                current_section = 'reasoning'
                reasoning = line.replace('REASONING:', '').strip()
            elif line.startswith('MISSING_INFO:'):
                current_section = 'missing_info'
                missing_info = line.replace('MISSING_INFO:', '').strip()
            elif current_section and line:
                if current_section == 'insights':
                    insights += " " + line
                elif current_section == 'reasoning':
                    reasoning += " " + line
                elif current_section == 'missing_info':
                    missing_info += " " + line

        return {
            'insights': insights or f"Retrieved {len(hop_docs)} documents from hop {hop_num}",
            'reasoning': reasoning or "Information contributes to understanding the original question",
            'missing_info': missing_info or "Additional context may be helpful"
        }

    except Exception as e:
        print(f"⚠️ Error in structured reasoning: {e}")
        return {
            'insights': f"Retrieved {len(hop_docs)} documents",
            'reasoning': "Analysis step encountered an error",
            'missing_info': "Unable to determine missing information"
        }

def _generate_next_subquestion_from_missing(original_query: str, current_query: str, missing_info: str, llm, current_hop: int) -> str:
    """
    Generate next sub-question specifically targeting the identified missing information.
    """
    # Build previous context for better sub-question generation
    global previous_context

    subquestion_prompt = f"""Generate a specific sub-question to find the missing information identified in the reasoning step.

ORIGINAL QUESTION: {original_query}
CURRENT SUB-QUESTION: {current_query}
MISSING INFORMATION: {missing_info}
CURRENT HOP: {current_hop}
PREVIOUS REASONING TRACE: {previous_context}

Based on the missing information, create a targeted sub-question that will help retrieve documents containing this specific information.

The sub-question should:
1. Directly address the missing information gap
2. Use specific terminology related to the missing information
3. Be concise and focused
4. Be different from previous queries
5. Don't repeat the original question or question generated in previous hops
6. The question generated MUST be different from {current_query}

Provide only the sub-question:"""

    try:
        response = llm.invoke(subquestion_prompt)
        sub_question = response.content.strip().strip('"').strip("'")
        return sub_question
    except Exception as e:
        print(f"⚠️ Error generating targeted sub-question: {e}")
        return original_query

def _truncate_documents(docs, word_limit: int, hop_num: int):
    """
    Truncate documents to word limit and add hop metadata for token efficiency.
    """
    truncated_docs = []

    for doc in docs:
        # Add hop information to metadata
        doc.metadata['hop'] = hop_num

        # Truncate content to word limit
        words = doc.page_content.split()
        if len(words) > word_limit:
            truncated_content = " ".join(words[:word_limit]) + "..."
            doc.page_content = truncated_content

        truncated_docs.append(doc)

    return truncated_docs

def _generate_final_answer_structured(query: str, docs, reasoning_trace, llm) -> str:
    """
    Generate final answer with clear separation of reasoning trace and synthesized answer.
    """

    # Prepare structured reasoning trace for display
    reasoning_display = []
    for hop_data in reasoning_trace['hops']:
        hop_num = hop_data['hop']
        reasoning_display.append(f"Q{hop_num}: {hop_data['question']}")
        reasoning_display.append(f"A{hop_num}: Retrieved {hop_data['retrieved_docs']} documents - {hop_data['insights']}")
        reasoning_display.append(f"R{hop_num}: {hop_data['reasoning']}")
        if hop_data['missing_info'].lower() not in ['no missing information', 'none', 'nothing']:
            reasoning_display.append(f"Missing: {hop_data['missing_info']}")
        reasoning_display.append("")  # Empty line for readability

    reasoning_context = "\n".join(reasoning_display)

    # Prepare document context with metadata
    doc_context_parts = []
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata
        company = metadata.get('company', 'Unknown').replace(' (1)', '')
        chunk_id = metadata.get('chunk_id', 'Unknown')
        source = metadata.get('source', 'Unknown')
        hop = metadata.get('hop', '?')

        doc_header = f"Document {i} [Company: {company}, Chunk: {chunk_id}, Source: {source}, Hop: {hop}]:"
        doc_context_parts.append(f"{doc_header}\n{doc.page_content}")

    document_context = "\n\n".join(doc_context_parts)

    final_prompt = f"""Provide a comprehensive answer using the multi-hop reasoning trace and retrieved documents.

ORIGINAL QUESTION: {query}

MULTI-HOP REASONING TRACE:
{reasoning_context}

RETRIEVED DOCUMENTS WITH METADATA:
{document_context}

INSTRUCTIONS:
- Provide a clear, comprehensive answer to the original question
- Use information from all hops and reasoning steps
- Include specific details, numbers, and facts from the documents
- Reference the hop number and source when mentioning specific information
- Structure your answer logically, building from the reasoning trace

ANSWER:"""

    try:
        response = llm.invoke(final_prompt)
        synthesized_answer = response.content.strip()
        print(f"✅ Final answer generated")
        print(synthesized_answer)

        # Format final output with clear separation
        final_output = f"""
🧠 MULTI-HOP REASONING TRACE:
{'=' * 40}
{reasoning_context}

🎯 SYNTHESIZED ANSWER:
{'=' * 40}
{synthesized_answer}

📋 DOCUMENT SOURCES:
{'=' * 40}"""

        # Add document source summary
        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata
            company = metadata.get('company', 'Unknown').replace(' (1)', '')
            chunk_id = metadata.get('chunk_id', 'Unknown')
            hop = metadata.get('hop', '?')
            final_output += f"\n{i}. {company} (Chunk: {chunk_id}, Hop: {hop})"

        return final_output

    except Exception as e:
        print(f"⚠️ Error in final answer generation: {e}")
        return "Error generating final structured answer."

def _retrieve_documents_simple(query: str, top_k: int = 5):
    """Simple document retrieval using direct Pinecone query"""
    try:
        # Generate embedding for query
        query_embedding = embedding_model.embed_query(query)

        # Retrieve from Pinecone
        response = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )

        # Convert to documents
        docs = []
        for match in response.matches:
            chunk_text = match.metadata.get('chunk_text', '')
            if chunk_text:
                # Create a simple document object
                class SimpleDoc:
                    def __init__(self, content, metadata):
                        self.page_content = content
                        self.metadata = metadata

                docs.append(SimpleDoc(chunk_text, match.metadata))

        return docs

    except Exception as e:
        print(f"⚠️ Error retrieving documents: {e}")
        return []

def _remove_duplicates_with_metadata(docs):
    """Remove duplicate documents based on chunk_id while preserving hop metadata"""
    seen_ids = {}
    unique_docs = []

    for doc in docs:
        chunk_id = doc.metadata.get('chunk_id')
        if chunk_id not in seen_ids:
            seen_ids[chunk_id] = doc
            unique_docs.append(doc)
        else:
            # Keep the document from the earliest hop
            existing_hop = seen_ids[chunk_id].metadata.get('hop', float('inf'))
            current_hop = doc.metadata.get('hop', float('inf'))
            if current_hop < existing_hop:
                # Replace with earlier hop version
                unique_docs = [d for d in unique_docs if d.metadata.get('chunk_id') != chunk_id]
                unique_docs.append(doc)
                seen_ids[chunk_id] = doc

    return unique_docs

print("✅ Enhanced MultiHop-RAG with Structured Reasoning ready!")
print("Usage: get_multihop_rag_answer(question, llm, max_hops=5, docs_per_hop=5, chunk_word_limit=500)")
print("\nEnhancements:")
print("✓ Structured reasoning trace (Q1, A1, R1, Q2, A2, R2, ...)")
print("✓ Explicit missing information identification")
print("✓ Document metadata in final output (hop, chunk_id, company)")
print("✓ Configurable truncation for token efficiency")
print("✓ Clear separation of reasoning trace and synthesized answer")
print("✓ FIXED: Function call argument mismatch resolved")

query = "Compare the Risk Factors of  Amazon, Apple, Nvidia, and Tesla in 2024"
get_multihop_rag_answer(query,llm)

"""# 🔍 1. Hybrid Search in Retrieval-Augmented Generation (RAG)

In Retrieval-Augmented Generation (RAG), large language models (LLMs) retrieve relevant external information before generating a response.  
However, traditional retrieval methods often miss key pieces of context due to:

- **Dense retrieval**: May skip exact keyword matches (e.g., numbers, acronyms)
- **Sparse retrieval (like BM25)**: May miss semantic relevance or synonyms

🚀 **Hybrid Search** solves this by combining both. It gives the **semantic flexibility of dense retrieval** and the **precision of sparse retrieval**, resulting in more robust and accurate retrieval.

---

# 💡 2. What is Hybrid Search?

**Hybrid Search** combines two retrieval strategies:
- **Dense retrieval**: Uses embedding vectors to find semantically similar documents
- **Sparse retrieval (BM25)**: Uses term-frequency and keyword overlap for matching

Then it **merges and reranks** the results to provide the most relevant set of chunks.

---

### 📌 Example:

**Query**: `"Tesla 2024 revenue growth"`

- **Dense Retrieval** finds:  
  `"Tesla's financial performance improved significantly over the last fiscal year."`  
  ✅ Good semantic match, but may not mention "2024" or "revenue" explicitly.

- **Sparse Retrieval (BM25)** finds:  
  `"Tesla's revenue in 2024 reached $95 billion."`  
  ✅ Exact match for year and keyword, but no explanation or context.

🎯 **Hybrid Search** combines both — retrieving chunks that **contain exact data** and **contextually explain it**.

---

# 📚 3. What is BM25?

**BM25 (Best Matching 25)** is a **sparse retrieval algorithm** used to rank documents based on:
- How often query terms appear (TF)
- How rare those terms are across the corpus (IDF)
- How long the document is (Length normalization)

### ✅ Why BM25 Matters:
- It's fast, robust, and great for retrieving **specific facts**, **named entities**, and **numerical values**.
- BM25 is used in tools like **Elasticsearch**, **Weaviate**, and **Vespa**.

---

# 🔍 4. Term Frequency – Inverse Document Frequency (TF-IDF) with Example
The history of TF-IDF (Term Frequency–Inverse Document Frequency) dates back to the early days of information retrieval and library science, long before modern search engines or vector embeddings existed.


### ✳️ Term Frequency (TF)

Measures how often a word appears in a document.

**Example**:
- Query: `"AI regulation"`
- Document A: `"AI is transforming industries. AI regulation is growing globally."`  
  - "AI" = 2 times  
  - "regulation" = 1 time  
✅ High TF → higher relevance

---

### ✳️ Inverse Document Frequency (IDF)

Measures how **rare** a word is across the entire corpus. Rare terms get higher scores.

**Example**:
- Common word: `"report"` (appears in 900/1000 docs) → low IDF
- Rare word: `"regulation"` (appears in 30/1000 docs) → high IDF  
✅ More important if a word is **less common**

---

### ✳️ Document Length Normalization

Avoids giving long documents an unfair advantage.

**Example**:
- Document A: 20 words, mentions "AI" once → more focused  
- Document B: 200 words, mentions "AI" once → less focused  
✅ BM25 favors A due to better signal-to-noise

---

# ⚖️ 5. Sparse vs Dense Retrieval (with Examples)

| Feature               | Sparse (BM25)                            | Dense (Embeddings)                             |
|----------------------|------------------------------------------|------------------------------------------------|
| Basis                | Keyword overlap                          | Semantic similarity                            |
| Handles synonyms     | ❌ No                                     | ✅ Yes ("revenue" ≈ "sales")                   |
| Handles typos        | ❌ No                                     | ✅ Often                                       |
| Great for            | Years, figures, named entities           | Paraphrased or vague concepts                  |
| Example Match        | `"Tesla revenue 2024"` → exact match     | `"How much money did Tesla make last year?"`   |
| Storage              | Inverted index (e.g., Elasticsearch)     | Vector DB (e.g., Pinecone, FAISS)              |
| Use case             | Legal, scientific, technical docs        | Open-domain, abstract, or fuzzy queries        |

✅ **Hybrid Search** combines these for the best of both worlds.

---

# 🔄 6. Step-by-Step Workflow: How Hybrid Search Works

Let's walk through how a typical **Hybrid Search** pipeline works in a Retrieval-Augmented Generation (RAG) system — using a real-world example, not just high-level steps.

---

# 🔄 6. Step-by-Step Workflow: How Hybrid Search Works

Let's walk through how a typical **Hybrid Search** pipeline works in a Retrieval-Augmented Generation (RAG) system — using a real-world example.

---

### 🎯 Query Example: `"Apple's AI investments in 2024"`

---

## ✅ Step 1: Dense Retrieval (Semantic Search)

🔹 The query is embedded into a vector.  
🔹 Top 4 results from the vector store (e.g., Pinecone):

| Dense Rank | Chunk Snippet                                                                 |
|------------|--------------------------------------------------------------------------------|
| 1          | "Apple is integrating generative AI in its chip design efforts."              |
| 2          | "AI investments by Apple have risen significantly in recent years."           |
| 3          | "Apple announced a new LLM for Siri."                                         |
| 4          | "Apple's new data center plans support AI workloads."                         |

---

## ✅ Step 2: Sparse Retrieval (BM25 Search)

🔹 Query terms like "Apple", "AI", and "2024" are matched exactly using a BM25 engine.  
🔹 Top 4 results from sparse index:

| Sparse Rank | Chunk Snippet                                                                 |
|-------------|--------------------------------------------------------------------------------|
| 1           | "In 2024, Apple invested $1B into generative AI research."                    |
| 2           | "Apple's 2024 fiscal report highlights AI as a strategic priority."           |
| 3           | "Apple and OpenAI collaboration details surfaced in 2024 announcements."      |
| 4           | "The 2024 keynote included AI-related hardware investments."                  |

---

## ✅ Step 3: Combine Results with Reciprocal Rank Fusion (RRF)

**What is RRF?**

> RRF is a simple yet powerful method to merge ranked lists using their **rank positions** (not raw scores).  
> It **boosts chunks** that appear in both dense and sparse results.

**RRF Formula**:

RRF Score= SUM(1/k +rank)

Where `k = 60` (commonly used).

---

### 🧪 RRF Example (k = 60)

Let's combine the rankings:

| Chunk                                                                                  | Dense Rank | Sparse Rank | RRF Score (1/(60+Dense Rank) + 1/(60+Sparse Rank)) |
|----------------------------------------------------------------------------------------|------------|-------------|-----------------------------------------------------|
| **"Apple is integrating generative AI in its chip design efforts."**                  | 1          | 3           | 1/61 + 1/63 = **0.0323** ✅ |
| **"Apple's 2024 fiscal report highlights AI as a strategic priority."**               | 2          | 2           | 1/62 + 1/62 = **0.0322** ✅ |
| "In 2024, Apple invested $1B into generative AI research."                             | —          | 1           | 1 / 61 = **0.0164**                                 |
| "AI investments by Apple have risen significantly in recent years."                    | 2          | —           | 1 / 62 = **0.0161**                                 |
| "Apple announced a new LLM for Siri."                                                  | 3          | —           | 1 / 63 = **0.0159**                                 |
| "Apple and OpenAI collaboration details surfaced in 2024 announcements."               | —          | 3           | 1 / 63 = **0.0159**                                 |

---

### ✅ Top 3 Chunks Selected for LLM Context

Based on the **highest RRF scores**, the system selects the following chunks to pass to the LLM:

1. 🥇 **"Apple is integrating generative AI in its chip design efforts."** – **0.0323**  
2. 🥈 **"Apple's 2024 fiscal report highlights AI as a strategic priority."** – **0.0322**  
3. 🥉 **"In 2024, Apple invested $1B into generative AI research."** – **0.0164**

✅ These chunks strike a balance between **semantic understanding** and **exact keyword grounding**.

---

## ✅ Step 4: Rerank and Select Top Chunks

After applying RRF:
- Select top 3–5 chunks with highest combined RRF scores
- Pass them to the LLM for grounded answer generation

✅ This ensures that responses are **relevant, accurate, and well-contextualized**

---

## 🔁 Why RRF is Better Than Raw Score Fusion

| Problem                   | RRF Advantage                                |
|---------------------------|-----------------------------------------------|
| Different scoring scales  | Uses only rank positions                     |
| No overlapping documents  | Still fairly merges separate results         |
| Easy to implement         | Requires no model training                   |
| Biased top scores         | Prevents one model from dominating unfairly  |

📌 **RRF is widely supported** in LangChain, Vespa, Weaviate, or can be manually implemented in Python.




---

# 🧠 7. When Should You Use Hybrid Search?

Use **Hybrid Search** when:

- Your queries combine **natural language + keywords**
- You're working with:
  - Financial reports (e.g., 10-Ks)
  - Legal contracts
  - Scientific publications
- You want:
  - Dense for **context**  
  - Sparse for **exact values and entities**

---

### ✅ Example Use Cases:

| Use Case                           | Why Hybrid Helps                          |
|-----------------------------------|-------------------------------------------|
| `"Amazon's cloud profit in 2023"` | Dense finds explanations, sparse finds exact numbers |
| `"FDA approval for Alzheimer's"`  | Sparse finds the drug name, dense gets policy context |
| `"AI investment trends"`          | Dense gets insights, sparse gets org names and figures |

---

# ⚙️ 8. Key Considerations for Hybrid Search

### ✅ Indexing Strategy
- Maintain **two indexes**:
  - Dense (e.g., Pinecone, FAISS)
  - Sparse (e.g., Elasticsearch, Weaviate)

---

### ✅ Score Fusion
- Choose your fusion technique:
  - **Weighted sum** of scores
  - **Reciprocal Rank Fusion (RRF)** — simple, rank-based method

---

### ✅ Normalization
- Ensure scores from both retrievers are scaled fairly  
  (e.g., normalize to [0, 1] or z-scores)

---

### ✅ Evaluation
Use metrics like:
- **Contextual Precision**
- **Faithfulness**
- **Groundedness**
to test before/after using hybrid search.

---

### ✅ Tools that Support Hybrid Search

- 🔧 **LangChain**: `MultiVectorRetriever` or `EnsembleRetriever`
- 🔧 **Weaviate**: Native BM25 + dense fusion
- 🔧 **Vespa**, **Jina**, **Elasticsearch** + custom scoring

---

📌 **Final Takeaway**:  
Hybrid Search = **Precision + Context**  
It's the retrieval engine you need when accuracy, explainability, and coverage all matter.

"""

# Cell 1: Build BM25 Retriever
# Install required package
from rank_bm25 import BM25Okapi
import string
from typing import List, Tuple
import re

class BM25Retriever:
    def __init__(self, documents):
        """
        Initialize BM25 retriever with documents

        Args:
            documents: List of Document objects from LangChain (your all_documents)
        """
        self.documents = documents
        self.document_texts = [doc.page_content for doc in documents]

        # Tokenize documents for BM25
        print("🔍 Tokenizing documents for BM25...")
        tokenized_docs = [self._tokenize(text) for text in self.document_texts]

        # Build BM25 index
        print("🏗️ Building BM25 index...")
        self.bm25 = BM25Okapi(tokenized_docs)

        print(f"✅ BM25 retriever built with {len(documents)} documents")

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for BM25

        Args:
            text: Input text to tokenize

        Returns:
            List of tokens
        """
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Split on whitespace and remove empty strings
        tokens = [token for token in text.split() if token.strip()]

        return tokens

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[any, float]]:
        """
        Retrieve top-k most relevant documents for a query

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            List of tuples (document, score) sorted by relevance
        """
        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k documents with scores
        top_indices = scores.argsort()[-top_k:][::-1]  # Get indices of top-k scores in descending order

        results = []
        for idx in top_indices:
            if idx < len(self.documents):
                doc = self.documents[idx]
                score = scores[idx]
                results.append((doc, score))

        return results

    def get_relevant_documents(self, query: str, k: int = 10) -> List[any]:
        """
        Alternative method that returns just documents (for compatibility)

        Args:
            query: Search query
            k: Number of documents to retrieve

        Returns:
            List of documents
        """
        results = self.retrieve(query, k)
        return [doc for doc, score in results]

# Build BM25 retriever from your documents
print("🚀 Creating BM25 retriever from all_documents...")
bm25_retriever = BM25Retriever(all_documents)

# Test the BM25 retriever
test_query = "revenue growth financial performance"
print(f"\n🧪 Testing BM25 retriever with query: '{test_query}'")

test_results = bm25_retriever.retrieve(test_query, top_k=3)
print(f"📊 Retrieved {len(test_results)} documents:")

for i, (doc, score) in enumerate(test_results, 1):
    company = doc.metadata.get('company', 'Unknown')
    section = doc.metadata.get('section', 'Unknown')
    chunk_id = doc.metadata.get('chunk_id', 'Unknown')
    print(f"  {i}. {company} - {section} (Score: {score:.4f}) - ID: {chunk_id}")
    print(f"     Preview: {doc.page_content[:100]}...")
    print()

print("✅ BM25 retriever is ready for hybrid search!")

# Cell 2: Hybrid Search Method with Reciprocal Rank Fusion

def get_rag_answer_hybrid(query: str, dense_retriever, bm25_retriever, llm, top_k: int = 5) -> str:
    """
    Retrieve documents using hybrid search (dense + sparse) with Reciprocal Rank Fusion

    Args:
        query: User's question
        dense_retriever: Pinecone/dense retriever
        bm25_retriever: BM25 sparse retriever
        llm: DeepSeek LLM instance
        top_k: Number of final documents to use for answer generation

    Returns:
        Generated answer based on hybrid retrieved context
    """

    # Step 1: Retrieve from Dense Retriever (Pinecone)
    try:
        print("🔍 Retrieving from DENSE retriever (Pinecone)...")
        query_embedding = embedding_model.embed_query(query)
        dense_response = index.query(
            vector=query_embedding,
            top_k=10,  # Get more for fusion
            include_metadata=True,
            include_values=False
        )

        dense_docs = []
        for match in dense_response.matches:
            chunk_text = match.metadata.get('chunk_text', '')
            if chunk_text:
                # Create a document-like object with metadata
                doc_obj = type('Document', (), {
                    'page_content': chunk_text,
                    'metadata': match.metadata
                })()
                dense_docs.append((doc_obj, match.score))

        print(f"✅ Dense retriever found {len(dense_docs)} documents")

        # Print dense retriever results
        print("\n📋 DENSE RETRIEVER RESULTS:")
        print("-" * 50)
        for i, (doc, score) in enumerate(dense_docs, 1):
            company = doc.metadata.get('company', 'Unknown')
            section = doc.metadata.get('section', 'Unknown')
            chunk_id = doc.metadata.get('chunk_id', 'Unknown')
            print(f"  {i}. {company} - {section} (Score: {score:.4f}) - ID: {chunk_id}")

    except Exception as e:
        print(f"❌ Error with dense retriever: {e}")
        dense_docs = []

    # Step 2: Retrieve from BM25 Retriever
    try:
        print("🔍 Retrieving from SPARSE retriever (BM25)...")
        bm25_docs = bm25_retriever.retrieve(query, top_k=10)
        print(f"✅ BM25 retriever found {len(bm25_docs)} documents")

        # Print BM25 retriever results
        print("\n📋 BM25 RETRIEVER RESULTS:")
        print("-" * 50)
        for i, (doc, score) in enumerate(bm25_docs, 1):
            company = doc.metadata.get('company', 'Unknown')
            section = doc.metadata.get('section', 'Unknown')
            chunk_id = doc.metadata.get('chunk_id', 'Unknown')
            print(f"  {i}. {company} - {section} (Score: {score:.4f}) - ID: {chunk_id}")

    except Exception as e:
        print(f"❌ Error with BM25 retriever: {e}")
        bm25_docs = []

    # Step 3: Apply Reciprocal Rank Fusion (RRF)
    print("\n🔄 Applying Reciprocal Rank Fusion...")

    rrf_k = 60  # RRF constant
    doc_scores = {}  # Dictionary to store combined scores
    doc_objects = {}  # Store actual document objects
    doc_sources = {}  # Dictionary to track which retrievers found each document

    # Process dense retriever results
    print("   📊 Processing dense retriever rankings...")
    for rank, (doc, score) in enumerate(dense_docs, 1):
        chunk_id = doc.metadata.get('chunk_id', f'dense_{rank}')
        rrf_score = 1 / (rrf_k + rank)

        if chunk_id in doc_scores:
            doc_scores[chunk_id] += rrf_score
            print(f"      Rank {rank}: {chunk_id} - RRF: {rrf_score:.6f} (DUPLICATE - adding to existing score)")
        else:
            doc_scores[chunk_id] = rrf_score
            doc_objects[chunk_id] = doc
            print(f"      Rank {rank}: {chunk_id} - RRF: {rrf_score:.6f}")

        # Track source
        if chunk_id in doc_sources:
            if 'dense' not in doc_sources[chunk_id]:
                doc_sources[chunk_id].append('dense')
        else:
            doc_sources[chunk_id] = ['dense']

    # Process BM25 results
    print("   📊 Processing BM25 retriever rankings...")
    for rank, (doc, score) in enumerate(bm25_docs, 1):
        chunk_id = doc.metadata.get('chunk_id', f'bm25_{rank}')
        rrf_score = 1 / (rrf_k + rank)

        if chunk_id in doc_scores:
            old_score = doc_scores[chunk_id]
            doc_scores[chunk_id] += rrf_score
            print(f"      Rank {rank}: {chunk_id} - RRF: {rrf_score:.6f} (DUPLICATE - combined score: {doc_scores[chunk_id]:.6f})")
        else:
            doc_scores[chunk_id] = rrf_score
            doc_objects[chunk_id] = doc
            print(f"      Rank {rank}: {chunk_id} - RRF: {rrf_score:.6f}")

        # Track source
        if chunk_id in doc_sources:
            if 'bm25' not in doc_sources[chunk_id]:
                doc_sources[chunk_id].append('bm25')
        else:
            doc_sources[chunk_id] = ['bm25']

    # Step 4: Sort by RRF scores and get top-k
    print(f"   🏆 Ranking {len(doc_scores)} unique documents by RRF scores...")
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    top_docs = sorted_docs[:top_k]

    # Step 5: Display hybrid search results
    print(f"\n📋 HYBRID SEARCH RESULTS (Top {top_k}):")
    print("-" * 70)

    final_docs = []
    final_metadata = []

    for i, (chunk_id, rrf_score) in enumerate(top_docs, 1):
        doc = doc_objects[chunk_id]
        final_docs.append(doc.page_content)
        final_metadata.append(doc.metadata)

        company = doc.metadata.get('company', 'Unknown')
        section = doc.metadata.get('section', 'Unknown')
        source = ' + '.join(doc_sources.get(chunk_id, ['Unknown']))

        print(f"Rank {i}: {company} - {section}")
        print(f"  Sources: {source}")
        print(f"  RRF Score: {rrf_score:.6f}")
        print(f"  Chunk ID: {chunk_id}")
        print()

    # Step 6: Create context and generate answer
    if not final_docs:
        return "No relevant information found using hybrid search."

    context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(final_docs)])

    # Enhanced prompt for hybrid search
    prompt = f"""Based on the following documents retrieved using hybrid search (combining semantic similarity and keyword matching), please answer the user's question accurately and comprehensively.

QUESTION: {query}

CONTEXT DOCUMENTS (Ranked by Hybrid Search):
{context}

INSTRUCTIONS:
- Use only the information provided in the context documents
- These documents were selected using both semantic similarity and keyword relevance
- If the information is not sufficient to answer the question, state this clearly
- Provide specific details and numbers when available
- Structure your answer clearly and concisely
- If data spans multiple years or sources, organize it logically

ANSWER:"""

    # Step 7: Generate answer with LLM
    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()

        print(f"🤖 Hybrid RAG Answer Generated ({len(answer)} characters)")
        return answer

    except Exception as e:
        return f"Error generating answer with DeepSeek: {str(e)}"

# Test the hybrid search method
print("🧪 Testing hybrid search method...")
test_query = "What factors did Amazon cite for declining profit margins?"

# Make sure you have your dense retriever and llm defined
try:
    hybrid_answer = get_rag_answer_hybrid(
        query=test_query,
        dense_retriever=None,  # Your dense retriever object
        bm25_retriever=bm25_retriever,
        llm=llm,
        top_k=5
    )
    print("\n" + "="*80)
    print("🎯 HYBRID SEARCH ANSWER:")
    print("="*80)
    print(hybrid_answer)
    print("="*80)

except Exception as e:
    print(f"❌ Error testing hybrid search: {e}")
    print("Make sure your dense retriever and llm are properly defined!")