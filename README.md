# 🔍 Advanced RAG Techniques

A comprehensive implementation of advanced Retrieval-Augmented Generation (RAG) techniques using real-world 10-K financial filings from Amazon, Tesla, Apple, and Nvidia.

## 🚀 Features

### Core RAG Pipeline
- **Document Processing**: Load and chunk 10-K PDFs with rich metadata
- **Vector Storage**: Pinecone integration with multilingual embeddings
- **Query Processing**: DeepSeek LLM for intelligent answer generation

### Advanced Techniques
- **🔄 Re-ranking**: Cohere re-ranker for improved retrieval quality
- **🔗 Multi-hop Retrieval**: Structured reasoning across multiple documents
- **🧭 Hybrid Search**: Combines dense (embeddings) + sparse (BM25) retrieval
- **📊 Reciprocal Rank Fusion**: Intelligent merging of retrieval results

## 📋 Prerequisites

### Required API Keys
- **Pinecone**: Vector database for embeddings
- **OpenAI**: LLM for answer generation
- **Cohere**: Re-ranking service

### Data Requirements
- 10-K PDF files in `data/` directory:
  - `Amazon.pdf`
  - `Apple.pdf` 
  - `Nvidia.pdf`
  - `Tesla.pdf`

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd advanced-rag
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp sample.env .env
# Edit .env with your actual API keys
```

4. **Add PDF files**
```bash
mkdir data
# Add your 10-K PDF files to the data/ directory
```

## 🔧 Configuration

### Environment Variables
Copy `sample.env` to `.env` and configure:

```bash
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX=your_pinecone_index_name_here
PINECONE_URL=https://your-pinecone-url.pinecone.io

# OpenAI API Configuration  
OPENAI_API_KEY=your_openai_api_key_here

# Cohere API Configuration (for re-ranking)
COHERE_API_KEY=your_cohere_api_key_here
```

### Pinecone Index Setup
Create a Pinecone index with:
- **Dimension**: 1024 (for multilingual-e5-large embeddings)
- **Metric**: cosine
- **Cloud**: AWS (recommended)

## 📚 Usage

### Basic RAG Query
```python
from advanced_rag import get_rag_answer

query = "What was Apple's R&D spending in 2024?"
answer = get_rag_answer(query, retriever, llm)
print(answer)
```

### Re-ranking Enhanced RAG
```python
from advanced_rag import get_rag_answer_with_cohere_rerank

query = "Compare Tesla's revenue growth across different regions"
answer = get_rag_answer_with_cohere_rerank(query, retriever, llm)
print(answer)
```

### Multi-hop Retrieval
```python
from advanced_rag import get_multihop_rag_answer

query = "How did Amazon's cloud business performance impact their overall profitability?"
answer = get_multihop_rag_answer(query, llm, max_hops=5)
print(answer)
```

### Hybrid Search
```python
from advanced_rag import get_rag_answer_hybrid

query = "What risk factors did Nvidia identify in their 2024 filing?"
answer = get_rag_answer_hybrid(query, dense_retriever, bm25_retriever, llm)
print(answer)
```

## 🧠 Advanced Techniques Explained

### 1. Re-ranking
- **Problem**: Vector similarity may miss the most relevant chunks
- **Solution**: Cross-encoder model (Cohere) re-ranks retrieved chunks
- **Benefit**: Higher precision and better answer quality

### 2. Multi-hop Retrieval  
- **Problem**: Complex questions require information from multiple sources
- **Solution**: Iterative retrieval with structured reasoning
- **Process**: Query → Retrieve → Reason → Sub-question → Repeat

### 3. Hybrid Search
- **Problem**: Dense retrieval misses exact matches, sparse misses semantic meaning
- **Solution**: Combine BM25 (sparse) + embeddings (dense) with RRF
- **Benefit**: Best of both worlds - precision + context

## 📊 Performance Comparison

| Technique | Accuracy | Speed | Use Case |
|-----------|----------|-------|----------|
| Basic RAG | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Simple queries |
| Re-ranking | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | High-accuracy needs |
| Multi-hop | ⭐⭐⭐⭐⭐ | ⭐⭐ | Complex reasoning |
| Hybrid | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Mixed query types |

## 🔍 Example Queries

### Financial Analysis
- "What was Tesla's revenue growth in 2024?"
- "Compare Apple's R&D spending vs. competitors"
- "What risk factors did Amazon identify?"

### Strategic Insights  
- "How did Nvidia's AI strategy evolve?"
- "What were the key challenges mentioned across all companies?"
- "Compare cloud business strategies"

### Technical Details
- "What were the specific AI investments mentioned?"
- "How did regulatory changes impact business models?"
- "What were the cybersecurity concerns raised?"

## 🛠️ Customization

### Adding New Documents
1. Add PDF files to `data/` directory
2. Update `expected_companies` list in the code
3. Re-run document processing

### Changing Embedding Model
```python
# In the code, change:
model = SentenceTransformer('intfloat/multilingual-e5-large')
# To your preferred model
```

### Adjusting Retrieval Parameters
```python
# Modify these parameters:
chunk_size = 500          # Document chunk size
chunk_overlap = 50        # Overlap between chunks  
top_k = 5                # Number of retrieved documents
max_hops = 5             # Multi-hop iterations
```

## 🐛 Troubleshooting

### Common Issues

**"Missing environment variables"**
- Ensure `.env` file exists and contains all required API keys
- Check that variable names match exactly

**"Pinecone index not found"**
- Create index in Pinecone console
- Verify index name matches `PINECONE_INDEX`
- Check API key permissions

**"No PDF files found"**
- Ensure PDF files are in `data/` directory
- Check file names match expected format
- Verify file permissions

**"LLM connection errors"**
- Verify DeepSeek API key is valid
- Check internet connection
- Ensure API quota hasn't been exceeded

### Performance Optimization

**Slow retrieval:**
- Reduce `top_k` parameter
- Use smaller embedding model
- Enable Pinecone caching

**High memory usage:**
- Reduce `chunk_size`
- Process documents in batches
- Use smaller embedding dimensions

**Poor answer quality:**
- Increase `top_k` for more context
- Enable re-ranking
- Use multi-hop for complex queries

## 📈 Evaluation Metrics

The system supports evaluation of:
- **Retrieval Accuracy**: How relevant are retrieved chunks?
- **Answer Quality**: Are answers accurate and complete?
- **Response Time**: How fast are queries processed?
- **Cost Efficiency**: API usage optimization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain**: RAG framework and utilities
- **Pinecone**: Vector database infrastructure  
- **DeepSeek**: LLM capabilities
- **Cohere**: Re-ranking technology
- **Sentence Transformers**: Embedding models

## 📞 Support

For questions or issues:
- Check the troubleshooting section
- Review the code comments
- Open an issue on GitHub

---

**Built with ❤️ for advanced RAG research and applications**
