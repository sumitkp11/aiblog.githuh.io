### Everything Start to Finish

## Create an agent
- You can change the primary language, solution and schema name for the agent.
- The description can have up to 1024 characters.
- To change agent's introduction: System topic -> Conversation Start -> Message node
- We can configure up to 10 suggested prompts that users can from to start a conversation with the agent. It isnt visible when testing the agent.

## Everything Topics:
1. To respond to users, agents use either generative orchestration or classic orchestration.
2. In generative orchestrations, agent selects the most appropriate combination of topics, tools and knowledge. Each topic has a description to inform the agent its purpose.
3. In classic orchestration, each topic has trigger phrases, keywords and questions that is likely to match with user input. Agents use Natural Language Understanding the select the topic.
4. A topic contains one or more nodes and determines the conversational paths.
5. There are 2 kinds of topics: System and Custom topics.
6. System topics cant be created, deleted but can be turned off.
7. Types of nodes: Message, Question, Adaptive Card, Condition, Variable Management, Topic Management, Tool, HTTP requests.
8. Node name can be up to 500 characters in length.
9. The topics can be edited with YAML code editor.

## Everything Knowledge
1. Agents can use enterprise data from Power Platform, Dynamics 365 data, websites and external systems.
2. At agent level we can use knowledge from Knowledge page but at the topic level, we can use generative answers node for knowledge.
3. Supported knowledge sources are: Public websites, Documents, SharePoint, Dataverse, Enterprise data using connectors.




## Limitations
1. The description can have up to 1024 characters.
2. Node names can be up to 500 characters in length.
3. Avoid using periods (.) in your topic names. It isn't possible to export a solution.
4. Text file to upload trigger phrases should be max 3 MB.
5. Under knowledge sources: 25 websites under generative mode while 4 public URLs in classic mode.
6. Under knowledge source: All/Any documents under generative node while Dataverse file storage under classic mode
7. Under knowledge source: Unlimited Dataverse under generative node while 2 Dataverse knowledge source (15 tables per knowledge source) under classic node.
