Currenty has one file to sketch out all the functions which will be used in this project. Based on the main branch which was a experimental POC and wasn't very modular.
Once sufficient functions have been implemented, they will split into folders based on their purpose and domain.

The idea is that there is an Agent, and every step of the pipline will have to inherit that Agent and have their implementation. This would add a degree of standardization for complex setup and possibly allow Orchestration setup too. 

# Orchestrator class:
- loads config
- handles init of all actions
- validates the compatibilty between action schemas
- handles branching and conditional actions
- will be provisioned to have a LLM call to act like Orchestrator AI
- will have a MCP client allowing integration with MCP registry to access external datasources


# Action class:
- Define input for the action
- Define output for the action
- Define the execution logic

# Notes
- Pydantic has a limitation where if I am trying to use the output of one schema as input for another the fields need have the same name or have a alias, to ensure cross compatibilty I will add aliases for other action fields as need for now
- Successfuly executed a sequence of actions, (convert, chunk, embed, upload). Need to update all actions so all the extra parameters like models or modes can be inserted without having to pass through the input schemas
- Need to look at MCP
- Need to test Query related sequences in Orchestrator
