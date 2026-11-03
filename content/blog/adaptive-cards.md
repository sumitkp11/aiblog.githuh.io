---
title: Adaptive Cards
description: 
date: 2026-10-30
tags: ["learning", "adaptive cards"]
---
# Adaptive Cards
- Adaptive cards are snippets of UI created in JSON.
- The JSON is transformed into native UI.
- It helps design and integrate light-weight UI for all major platforms and frameworks.

## Goals of Adaptive Card
1. Portable to any app, device or UI framework
2. Open: Libraries and schema are open source and shared
3. Low cost: Easy to define, easy to consume
4. Expressive
5. Purely declarative: No code is needed
6. Automatically styled

## Card Layouts
1. Hero: Ideal for sharing articles where an image tells most of the story.
2. Thumbnail: for quick interactions
3. List: to pick an item from a list
4. Digest: for news digests or round-up posts
5. Video/Media: Combines text and media
6. Form: to collect quick inputs from users to create tasks
7. Quick Input: to collect quick inputs from users
8. Expandable: allows users to select buttons and gather additional input from the same card.

## Collapisble Sections
- Action.ShowCard and Action.ToggleVisibility to show information dynamically.

## Container Layouts
1. Layout.Stack: stacks elements on top of each other. Used by default.
2. Layout.Flow: spreads elements horizontally and wraps them across multiple rows as needed.
4. Layout.AreaGrid: divides a container into named areas into which elements can be placed.

## Elements:
1. ActionSet: Displays a set of action, which can be placed anywhere in the card.
2. Table
3. Progressar
4. ImageSet
5. Image
6. Icon
7. FactSet
8. Container: to logically group a set of elements together, A container for other elements. 
9. TextBlock:

## Actions
1. Action.Execute
2. Action.InsertImage
3. Action.OpenUrl
4. Action.OpenUrlDialog
5. Action.Popover
6. Action.ResetInputs
7. Action.ShowCard
8. Action.Submit
9. Action.ToggleVisibilty

## Inputs
1. Input.ChoiceSet
2. Input.Date
3. Input.Number
4. Input.Rating
5. Input.Text
6. Input.Time
7. Input.Toggle
