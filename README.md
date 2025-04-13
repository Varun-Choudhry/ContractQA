# Contract QA

## Technology Used
- Azure Document Intelligence for PDF analysis
- Azure OpenAI for embedding and completion
- Weaviate for storing chunks
- Mongodb to store filenames, insights and status
- Python

## Upload Document

- User uploads a PDF document to the UI, the filename is queried against the MongoDB database to check for duplicates
- If there is no existing entry, the file is sent to Azure Document Intelligence
- Azure DI outputs a JSON file which has has all the logical sections which could be derived from the document along with content and metadata
- The output JSON is consumed and parsed and chunks are created based on the sections in the the JSON, it uses a tree based data structure with dfs to sequentially combine sections to fit the word limit set
- The chunks are embedded using an embedding model and then uploaded to Weaviate
- A new entry is added to the MongoDB database for the file

## Query Documents

- User will select the document they want to QA against from the dropdown available, the dropdown is populated with all files which have been processed
- The query is consumed by a Decompose agent which makes an LLM call to determine if it can be broken down into subqueries, The output is either the original query or list of subqueries
- The query/subqueries are then used by the retrieval tool to perform a hybrid search (BM25 + Cosine Similarity) on the Weaviate databases
- All the queries and their associated retrieved context is consumed by the Final Answer agent which outputs the answer to the user

## Insight Generation

- This process is not user driven, the idea is that there will be a polling function which query the mongodb database to identify which documents have not been summarized.
- Those documents will be fed into a entity agent whose will loop through all the chunks related to that document sequentially and make a JSON which contains extracted inisghts.
- Current setup extracts partiy information, timelines and key obligations from the contract, this data is stored in MongoDB



## Future Improvements
  - Add redundancy check using MD5 for duplicate files
  - Add reranking
  - Add more agents to the Insight generation flow to derive more info from the documents
  - Add support for structural querying such as page numbers
  - Add support for figures
  - Add a topical searcher which will search all documents instead of of as a Knowledge base
  - Deploy an online version

- 

 

