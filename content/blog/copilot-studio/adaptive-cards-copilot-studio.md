# Adaptive Cards in Copilot Studio

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
