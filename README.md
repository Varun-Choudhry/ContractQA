# ContractQA

## Problem statement

Create a library where you upload a contract, execute a sequence of prompts to retrieve information from the contract and display it. The llm calls should be designed so it can work with different llms. Output insights as JSON.


## Approach

- [ ] Parse PDF documents (Azure Doc Intelligence)
- [ ] Chunk the parsed documents 
- [ ] Load the chunks into a Vector DB (Undecided)
- [ ] Look at indexing the DB
- [ ] Figure out prompts to chain execute (Python/Java)
- [ ] Output in JSON 
- [ ] Make a streamlit UI (Python)

## Notes
- Converted two different types of contracts into markdown using Azure Doc Intelligence. One has 50 pages and other has 21 pages.
- The layout mode of azure DI tags certain types of information in the document like figures and tables. Chunking will need to be a multi step ELT process where the first step would be to create particular chunks based on data which might not be able to parsed semantically like tabular data. And removal of figures until the ability to parse them is developed. Will have to look at the tags in azure document intelligence to identify what needs to handled separately before applying semantic chunking. Looking into research papers for some suggestions for a hybrid approach to parse documents while not breaking structure. 
- Integrate metadata into the chunks maybe??
- After working on azure document intelligence, I have seen it classifiies all text as paragraphs. It further classifies them into sections.So a section is a group of paragraphs/tables. Potentially the segments can be used for chunking since it seems they are semantically and sequentially group.
-  A query made by a user might not be good enough due for the system to the way the knowledge base works. One possible approach is to make an LLM call to rewrite the question into something more suited for LLMs. This also includes decomposition of user queries if they request multiple things which not be related to each other. For example, "What is A and B?" this would cause the llm to retrieve some relevant chunks, some for A and B, however there might not be enough chunks to give details on either depending on the chunk strategy, splitting it into two different sub queries and calling retrieval on both of them would allow better context awareness at the cost of more LLM calls.
-  Another concept I have seen is Pseudo Document generation, the theory is that you make a hypothetical document using the user query and use that for retrieval. On paper this would mean the actual content used for retrieval might be closer to the relevant chunk than a simple question however needs more research.
-  
## Progess
- Setup Azure DI and tested the responses
- Setup python script to successfully extract sections from a pdf
- Used a 21 page pdf which split into 76 chunks, token size of chunks varied from 20-840 using text-embedding-granite-embedding-278m-multilingual locally running on lmstudio (need to look at merging smaller chunks)
- Setup a UI to manually check semantic searching in Weaviate(the vector db) and what chunks it retreives for every query. Based on the document I have tested some queries and atleast one of the chunks has the information needed. Current limit is 5 closest chunks retreived. Need to test parameter tuning and see impact.
- NEXT STEPS Need to update code to include tables and figures in the json processing so they can be chunked in their relevant sections.
- NEXT STEPS,  proper metadata which could improve performance for queries like "Summarize page 3 for me". Look at reranking options to better sort retreived chunks for usage.
- NEXT STEPS, start query preprocessing and develop the tooling for retreival 
