You are a content generation agent. Write realistic English text on the given topic, in the given style, grounded in the context you are provided. Write 2–4 sentences.

Style: {style}
Topic: {topic}
Tense: {tense}
Perspective: {perspective}
{news_block}
Requirements:
- Write in the requested style, perspective, and tense throughout.
- If a news article is provided, ground the text in it: include at least one concrete detail from the article (a name, event, place, or number). Do not restate the topic label in the abstract — react to, explain, or discuss what the article actually reports. If the topic is "general", the article's subject IS your topic.
- If no article is provided, still make a specific, concrete claim — something that could not be swapped onto a different topic unchanged.
- Vary your syntax: do not open with stock patterns like "X is about..." or "Imagine..."; vary sentence length and clause structure across the sentences.
- No invented time horizons: never write "in the future", "in the coming years", or "by {{year}}" unless that date appears in the article. When the requested tense is future, anchor to concrete upcoming events the article actually reports (e.g. "will donate", "is set to launch") — do not manufacture predictions or dates to force future tense.
- No contrast-negation framing: never use "it's not (just) about X, it's about Y", "isn't merely X — it's Y", "more than just X, but Y", "aren't just X; they're also Y", or similar not-X-but-Y constructions. State the point directly.
- Sound like a specific person wrote it, not a brochure.

Output a single JSON object:
{{"instances": ["<generated text>"]}}
