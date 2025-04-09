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
- Created a script which reads azure json and chunks it, each chunk is minimum 256 tokens except the last one, the primary logic for grouping is the section data in the json, any tables and assoicated data is converted to html markup for improved readability while conveying its a table, the chunks are then loaded into a weaviate collection along with information like page numbers added as metadata using "text-embedding-granite-embedding-278m-multilingual" hosted on lmstudio locally.
- Created a ui to test an LLM (deepseek r1 running on lmstudio). Any query sent is sent along with the retreived chunks. Current implementation has vector search, hybrid search with a slider to choose dense or sparse retrieval and HyDE approach.
- NOTE currently all scripts are experimental and are not modular or concise. That will be done once my experiments with different approaches are complete.
- NEXT WORK, define proper tooling and retrieval functions. Add tools for retreiving tables and pages etc, which is available in the weaviate metadata. Identify and decompose the singular prompt into a chain for better habdling of complex queries.


