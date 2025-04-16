Currenty has one file to sketch out all the functions which will be used in this project. Based on the main branch which was a experimental POC and wasn't very modular.
Once sufficient functions have been implemented, they will split into folders based on their purpose and domain.

The idea is that there is an Agent, and every step of the pipline will have to inherit that Agent and have their implementation. This would add a degree of standardization for complex setup and possibly allow Orchestration setup too. 

Orchestrator class:
- loads config
- handles init of all actions
- validates the compatibilty between action schemas
- handles branching and conditional actions
- will be provisioned to have a LLM call to act like Orchestrator AI


Action class:
- Define input for the action
- Define output for the action
- Define the execution logic