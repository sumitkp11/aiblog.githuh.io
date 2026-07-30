---
title: Agent Builder in Microsoft 365 Copilot
description: 
date: 2026-07-30
tags: ["copilot"]
---
# Agent Builder in Microsoft 365 Copilot
- Agent Builder provides an easy way to build declarative agents for Microsoft 365.
- Agents can be built from following apps and sites:
  1. microsoft365.com/chat
  2. office.com/chat
  3. Microsoft Teams Desktop and web client

## Building agents using Agent Builder
- When a new agent is chosen in Microsoft 365 Copilot, we can:
1. use natural language to describe agent and copilot builds it
2. configure agent manually using Configure tab
3. Start with a template for a specific use case.

## RECOMMENDED: Using natural language to describe agent
- This approach understands a user's intent through natural language. It provides suggestions, guidance and next-based prompts.
- It generates optimized agent instructions to build accurate, high-quality agents.
- After describing the agent, the tool prepopulates: Name, Description, Instructions, Knowledge sources, Suggested prompts

## Configure agent manually
- On New agent, select `Skip to configure`.
- Following fields make up the agent:
  1. Name: name of the agent with a character limit of 30 characters.
  2. Icon: choose an icon from library or manually upload one with PNG file with a transparent background at 192x192 pixels resolution under 1 MB size limit.
  3. Model: choose the LLM model on which agent will run
  4. Description: helps the LLM identify and use agent for specific task or situation with 1000 characters as the limit.
  5. Instructions: specific instructions to the LLM to extent the capabilities with 8,000 characters limit.
  6. Knowledge: specify up to 20 knowledge sources or Copilot connectors.
  7. Capabilities: enhance user experience by adding capabilities.
  8. Starter Prompts: it helps other users understand common supported scenarios by the agent. Each one comes with a name and description with no minimum number required.
 
## Build from template
- It includes templates to reuse to build agents for specific use cases.
- It comes preconfigured with description, instructions and prompts.
- To start: New agent -> Start with a template

## Adding capabilities
- Following capabilities can be added to the agent:
  1. Code interpreter: solve complex math problems, analyze data and generate visualizations.
  2. Image generator: generate image based on user prompts.
 
## Default Response Mode
- It controls how the agent approaches each question, either to prioritize speed or take more time for in-depth analysis.
- Different response modes are:
  1. **Auto** - set as default, automatically chooses the best approach based on each question, balancing speed and depth of analysis.
  2. **Quick response**: agent replies quickly, keeps responses concise, it doesn't require in-depth analysis.
  3. **Think deeper**: it takes more time to analyze the question before responding.
 
## Adding knowledge sources
- Following knowledge sources can be added:
  1. public website URLs: must be only two levels, cannot contain query parameters, upto 4 URLs
  2. upto 100 SharePoint files, folders or sites: list can have max of 20,000 rows and 50 MB of raw text, Attachments column isnt indexed
  3. up to 50 onedrive files
  4. up to 5 Teams chat URLs
  5. embedded files
  6. copilot connectors
  
