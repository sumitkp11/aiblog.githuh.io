---
title: Interview Topics in Detail
description: 
date: 2026-10-26
tags: ["copilot studio", "visual studio code"]
---

## Autonomous Agent
- Autonomous agents in Copilot Studio extend the value of generative orchestration by enabling AI to take action without waiting for a user prompt.
- These agents perceive events, make decisions, and execute tasks independently by using triggers, instructions, and guardrails you define.
- Instead of responding only in conversations, they operate continuously in the background—monitoring data, reacting to conditions, and running workflows at scale.

## Semi-autonomous Agent
- A semi-autonomous agent works similarly, but it pauses to check in with a human for approval before taking any important action.

## Generative Orchestration
- Generative orchestration lets your custom agent answer user queries with relevant topics, tools, other agents, and knowledge sources, and respond to event triggers.
- Generative orchestration builds a plan to address a user query or event trigger using the name, description, inputs, and outputs of available topics, tools, agents, and knowledge.
- For conversations, the system references the last 10 turns of conversation history to fill in inputs.
- Link (https://learn.microsoft.com/en-us/microsoft-copilot-studio/faqs-generative-orchestration)

## Copilot Foundational Elements
1. Knowledge: Allows the agent to use predefined documents, FAQs, or external data sources to provide instructions for the agent to respond.
2. Topics: Manages the conversation flows by defining intents, triggers, and responses for user queries.
3. Actions: Defines what the agent can do, such as triggering an action or workflow, calling an API, sending an email, or integrating with external systems to trigger an external workflow.
4. Triggers: Event-based starting points that initiate an automated flow, which help define specific conditions that activate specific actions.
5. Activity: Tracks and logs interactions, providing visibility into user engagement with the agent.
6. Analytics: Provides insights and performance metrics on agent usage, effectiveness, and user interactions.
7. Channels: Configures deployment options, which allows the agent to be accessible on various platforms like Microsoft Teams or internal and external websites.
