---
title: Tools in GitHub Copilot Harness
description: 
date: 2026-10-06
tags: ["copilot studio", "github copilot harness"]
---
## Tools
- Tools let the agent interact with external systems.
- Examples of tools: send emails, check current weather conditions, read and write data from Dataverse, read and post messages to Teams.
- The agents use tools to respond to users automatically using generative orchestration.
- Tools can be called explicitly within a topic.

## Mechanisms to add tools
- Connector: Connect to proprietary APIs and services by using Power Platform Connectors using prebuilt connector or custom connector.
- Agent flow
- Prompt
- REST API
- Model Context Protocol
- Computer use
- Tool-like behaviour using Azure Bot Service Skills and Client tools

## Limitations on tools
- A generative orchestrator supports up to 128 tools per agent.
- For best performance, limit to 25 to 30.
- When you use multi-agent orchestration with child agents, each child agent has its own orchestration and can manage its own set of up to 128 tools.

