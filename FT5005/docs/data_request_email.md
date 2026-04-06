# Draft Email To Professor Huang

To: `huangkw@comp.nus.edu.sg`

Subject: Option 2 project and data access for high-frequency ECC study

Hi Professor Huang,

Our group plans to take Option 2. We are currently considering a project on earnings conference calls and high-frequency market reactions around earnings events.

May I check whether AIDF access can support downloading the following data?

1. Earnings conference call transcripts
- full transcript text
- speaker labels / speaker roles
- call date
- exact start time and end time if available
- firm identifier such as ticker / gvkey / permno / CUSIP / RIC

2. High-frequency market data around earnings events
- intraday price data, ideally 1-minute or 5-minute frequency
- timestamp
- open / high / low / close
- trading volume
- if available: bid, ask, spread, quote or trade counts
- firm identifier and adjustment information

3. Earnings event linking / analyst data if available
- exact earnings announcement date and time
- analyst estimates / earnings surprise / analyst coverage
- linking keys across CRSP / Compustat / IBES / transcript data

If possible, could you also let us know whether these data can be used for a later conference submission, at least in the form of derived features and reported results?

If these data are available, we would be happy to come to AIDF to download the relevant dataset.

Best regards,
[Your Name]

## Short Rationale

Why these are the core requirements:
- Transcript text and speaker roles are needed to separate prepared remarks from Q&A.
- Exact call and announcement timestamps are needed to define clean intraday event windows.
- Intraday price and volume data are needed to study high-frequency market response after earnings calls.
- Analyst and linking data are needed for controls and event alignment.

## Minimum Required Fields

### ECC transcript data
- event identifier
- firm identifier
- call date
- call start timestamp
- call end timestamp if available
- transcript text
- speaker name
- speaker role
- segment or paragraph order

### High-frequency market data
- firm identifier
- trading date
- intraday timestamp
- price or OHLC bars
- volume
- adjustment flag or split-adjusted indicator

### Event / estimate data
- earnings announcement timestamp
- actual EPS
- consensus EPS
- surprise measure if available
- analyst count if available
- mapping keys across databases

## Nice-To-Have Fields

- article-level news data with timestamps
- source outlet metadata
- bid and ask quotes
- spread / depth / trade counts
- exchange code
- SIC / GICS industry mapping
- delisting and corporate action flags

## Notes For Us

If the full high-frequency package is not available, the next-best fallback is:
1. exact earnings announcement timestamp
2. exact call timestamp
3. 5-minute intraday bars
4. full ECC transcripts with speaker roles

That is still enough for a solid first-stage event-study design.
