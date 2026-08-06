---
title: Installation of Copilot Studio extension in VSS
description: 
date: 2026-08-06
tags: ["copilot studio", "visual studio code"]
---
## Installation Steps
1. Search for 'Copilot Studio' in Extension marketplace in Visual Studio Code and install.
2. Click on the Copilot Studio icon in the Activity Bar.
3. Click on 'Clone agent'.
4. In next popup, it will ask you to sign in using Microsoft, click 'Allow'.
5. After sign in, in the VS Code, select the agent or the environment from the drop down.
6. After selecting your environment, select your agent.
7. Next, select a folder to download the agent locally. This would clone the agent locally to the specified folder and open the folder in VS Code.

## Best practices (Do):
1. Clone all agents to a consistent location.
2. Initialize Git after cloning.
3. Use meaningful folder names
4. Check the clone works before making changes
5. Keep clone backed to GitHub.
6. Document clone location to internal team docs

## Best practices (Don't):
1. Don't clone to temporary folders.
2. Don't clone multiple times to different locations.
