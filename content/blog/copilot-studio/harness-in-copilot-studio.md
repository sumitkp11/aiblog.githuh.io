---
title: Harness in Microsoft Copilot Studio
description: 
date: 2026-10-01
tags: ["copilot studio"]
---
## Key
- MCS: Microsoft Copilot Studio


## What is a harness?
- A harness is the operating layer between the model and the agent's configuration in MCS.
- It determines how the model receives context, uses instructions and tools, interprets result and moves towards task completion.
- The harness is a runtime that determines when to call the model, what components to send it, interprets what comes back and calls the right tools.
- There are 3 types of harness: GitHub Copilot harness, standard harness and Copilot Chat harness.


## Why GitHub Copilot Harness?
- It is built for reasoning-heavy agents and workflows that need to complete complex business processes.
- It can take a goal, break it into steps, call the right tools across connectors, knowledge, MCP and connected agents.
- It natively creates and edits Word, Excel, PowerPoint and PDF files.
- It runs each task in a secure sandbox.
- The agents and workflows using Copilot Credits.

## Why use Standard harness?
- It is a dependable option for rule-based agents and repeated workflows where you want predictable behavior for well-understood requests.
- You define the topics, prompts and paths so that it responds consistently.
- It use case is good for an internal help-desk to answer common questions and route simple requests using a defined workflow.

## Why use Copilot chat harness?
- It allows to extend Microsoft 365 Copilot Chat.
- You can connect your enterprise knowledge to M365 Copilot Chat.
- It runs on current chat models.
- You can publish to internal teams.
- Billing is consumption-based or included in Microsoft 365 Copilot user subscription license.


## When to choose what?
- Choose standard harness when the work is short and bounded with explicit control matter the most.
- Choose GitHub Copilot harness when the task is long-running, coordination-heavy, reasoning-intensive or refine work to reach an outcome.

## When to be careful?
- Without optimization, a more capable harness can overwork a simple problem, increasing Copilot Credit consumption.
