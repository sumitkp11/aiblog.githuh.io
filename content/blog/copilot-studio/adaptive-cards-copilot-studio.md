---
title: Adaptive Cards in Copilot Studio
description: 
date: 2026-08-27
tags: ["copilot studio", "adaptive cards"]
---
# Adaptive Cards in Copilot Studio
- Adaptive Cards automatically adapt their UI to the host application's style but to make it responsive attention needs to given to layout is optimal in each of width groups. This can be achieved using Container layouts with `targetWidth` property for different width.

## Text features

### Using Markdown
- To insert text, we use the 'TextBlock'.
- It can be formatted using Markdown.
- Find more here: https://adaptivecards.microsoft.com/?topic=text-formatting

### Date and time formatting
- DATE and TIME functions are to be used. For DATE, default format is COMPACT.
- DATE syntax: `{{DATE(<date>|<date-time>[,COMPACT|SHORT|LONG])}}`

- TIME syntax: `{{TIME(<time>|<date-time>)}}`

* `<date-time>` is expressed in Zulu format: 2017-02-14T06:00:00Z
* <date> is expressed as 2017-02-13
* <time> is expressed as 06:00:00Z

#### Examples
1. Compact date: `{{DATE(2017-02-14T06:00:00Z)}}` -> 14/2/2017
2. Short date: `{{DATE(2017-02-14T06:00:00Z,SHORT)}}` -> Tue, 14 Feb, 2017
3. Long date: `{{DATE(2017-02-14T06:00:00Z,LONG)}}` -> Tuesday, 14 February 2017
4. Date and time: `{{DATE(2017-02-14T06:00:00Z,LONG)}}, {{TIME(2017-02-14T06:00:00Z)}}` -> Tuesday, 14 February 2017, 11:30 am


## Responsive Layouts
1. The four width groups: Wide, Standard, Narrow, Very narrow
2. Make use of atLeast and atMost prefixes in targetWidth property to make an element visible only when the card width is 'standard or above' or only when the card width is 'narrow or below'. E.g. atLeast:standard

### Wide Width
- It is applicable on mobile devices/ tablets in landscape orientation.


### Standard Width
- It is applicable for chats on desktop.



### Narrow and Very Narrow Width
- Narrow is applicable for mobile devices in portrait mode while Very narrow is applicable for compact side panels such as meeting chat pane in Teams where space is little.


## Collapsible Sections
- It allows content to be shown or hidden interactively.
- Use Action.ShowCard or Action.ToggleVisibility to present information dynamically.

## Action.ShowCard
- It displays an embedded card within an Adaptive Card.
- It allows to display additional information without going away from the current card.
- Usage: multi-step forms, progressive information reveal
- Example #1: User clicks a button to 'Reserve a seat', it then opens another card to enter name and email and a button to Submit.
- Example #2: User clicks on 'Show graphical data', it shows a pie chart in a card.

## Action.ToggleVisibility
- It allows elements within the card to hidden or shown on user interaction.
- Usage: to save space when targeting the very narrow layout, interactive elements.
- Example: User clicks on 'Show more'.
- 
