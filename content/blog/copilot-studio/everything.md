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
4. Allow ungrounded responses setting manages whether the agent can generate responses using only the model's general knowledge.
5. Official sources mark a knowledge sources to be highly trusted and can be used directly by the agent without verification but generative orchestration needs to be turned off.
6. All knowledge sources added at agent level are added to the generative answers node in Conversational boosting system topic.

## Everything tools
1. Tools let agent interact with external systems.
2. Tools: Connectors, Agent flow, Prompt, REST API and Model Context Protocol (MCP) and Computer use with GUI.
3. Agent flow response:
   a. Dont respond (default)
   b. Write the response with generative AI
   c. Send specific response
   d. Send an adaptive card


## Multi-agent solution
1. Agents can hand off user interactions to each other or respond to autonomous triggers.
2. Ways to add other agent:
   a. Create child agents
   b. Connect to other Copilot Studio agents within same environment
   c. Connect to external agents: Over A2A protocol, Foundry agents, Fabric Data agents, Microsoft 365 Agents SDK.
3. Potential impact: Increased latency due to extra orchestration hops, Increased testing, management and governance surface area.
4. Use 'Add an agent' node to redirect to an agent from a topic.
5. Issues: introduces a slightly longer execution time due to context switching, complexity in maintaining multiple agents.
6. Best practices:
   a. Ensure only one agent talk to the user. Only the parent agent should deliver the final response.
   b. Always tell subagents that they are sub agents as sub agents inherently dont know they are part of an orchestration.
   c. Use clear, direct language in instructions. Avoid soft phrasing like "please", "try to", "you should" but instead use strong language like "MUST", "DO NOT", "NEVER".
   d. Use one knowledge source per subagent.
   e. Write clear, distinct descriptions for each subagent visible to the parent. The parent agent uses subagent descriptions to decide routing.
   f. Tell the parent agent how to orchestrate. The parent needs explicit instructions on the pattern: invoke agents, wait for results, combine, then respond.
   g. 




## Limitations
1. The description can have up to 1024 characters.
2. Node names can be up to 500 characters in length.
3. Avoid using periods (.) in your topic names. It isn't possible to export a solution.
4. Text file to upload trigger phrases should be max 3 MB.
5. Under knowledge sources: 25 websites under generative mode while 4 public URLs in classic mode.
6. Under knowledge source: All/Any documents under generative mode while Dataverse file storage under classic mode
7. Under knowledge source: Unlimited Dataverse under generative mode while 2 Dataverse knowledge source (15 tables per knowledge source) under classic mode.
8. Under knowledge source: Unlimited connectors under generative mode while 2 per custom agent in classic mode.
9. In knowledge source, the agent only shows content that the specific user can access.
10. File uploads to agent are limited to 15 MB.
11. Agent flow must respond to the agent within the 100-second action limit.
12. Under generative mode, the orchestrator supports up to 128 tools per agent. For best performance, limit to 25-30 tools.
13. 
