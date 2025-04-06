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

## Progress and Notes
- Converted two different types of contracts into markdown using Azure Doc Intelligence. One has 50 pages and other has 21 pages.
- The layout mode of azure DI tags certain types of information in the document like figures and tables. Chunking will need to be a multi step ELT process where the first step would be to create particular chunks based on data which might not be able to parsed semantically like tabular data. And removal of figures until the ability to parse them is developed. Will have to look at the tags in azure document intelligence to identify what needs to handled separately before applying semantic chunking. Looking into research papers for some suggestions for a hybrid approach to parse documents while not breaking structure. 
- Integrate metadata into the chunks maybe??
