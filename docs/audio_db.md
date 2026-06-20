# Hanning Labs — Audio Data Schema (v1 draft)

## Context

Hanning Labs collects, curates, and licenses parallel speech data for speech-to-speech
(S2S) translation. There is **no data layer yet** — the repo currently holds only the
marketing `website/`. We want a schema that is **solid enough to start ingesting audio**
and **flexible enough to grow metadata** without constant migrations.

The atomic unit of the business is the **utterance**: one speaker saying one thing, in one
language, as one segment of audio. It is simultaneously the licensable unit, the
annotatable unit, and the QA unit. The schema radiates out from it.

### Decisions locked during brainstorming

1. **Parallel S2S model → semantic-unit hub.** A `semantic_units` table represents one
   logical sentence (a meaning); each utterance carries a nullable `semantic_unit_id`.
   N-way parallelism, partial coverage, and monolingual data all fit one schema. Pairs and
   parallel sets are *derived by query*, not stored as link rows. (Pairwise `alignments`
   left as an optional future add-on for found/post-hoc-aligned data.)
2. **Flexibility → JSONB hybrid, no EAV.** Typed columns for anything we filter / join /
   constrain on; a single `metadata` JSON(B) column per major table for the long tail.
   Fields "graduate" from JSONB to typed columns once stable. EAV explicitly rejected.
3. **Scope → core first.** Ship `languages, speakers, sessions, recordings,
   semantic_units, utterances, annotations`. Defer consent, QA, datasets, licensing.
   Carry cheap provenance *fields* (checksum, source) on rows from day one since they are
   hard to backfill.
4. **Engine → deferred, design portable, leaning Postgres.** Table shapes are identical
   across engines; only the JSON/UUID/index primitives differ. Recommend Postgres for a
   join-heavy, query-the-metadata catalog at 24,000 h scale.

## Schema (v1 core)

```
 speakers ──< sessions ──< recordings ──< utterances >── semantic_units
                                              │  └──< annotations
 languages ───────────────────────────────────┘
```

Postgres-flavored DDL below. For SQLite/D1: `uuid`→`text` (app-generated UUID),
`jsonb`→`json`, `timestamptz`→`text`/`integer`, drop the GIN indexes.

### Reference: languages

```sql
create table languages (
  code        text primary key,        -- BCP-47: 'en', 'es-419', 'yue-Hant'
  name        text not null,
  metadata    jsonb not null default '{}'
);
```

### speakers

```sql
create table speakers (
  id          uuid primary key default gen_random_uuid(),
  display_name text,                    -- pseudonym ok; PII governed separately
  metadata    jsonb not null default '{}',   -- demographics, voice traits, etc.
  created_at  timestamptz not null default now()
);
```

### sessions — a collection event (studio or field)

```sql
create table sessions (
  id          uuid primary key default gen_random_uuid(),
  speaker_id  uuid references speakers(id),
  kind        text not null,           -- 'studio' | 'field'
  recorded_at timestamptz,
  location    text,
  metadata    jsonb not null default '{}',   -- equipment, room, engineer, weather…
  created_at  timestamptz not null default now()
);
```

### recordings — one raw audio file; bytes live in object storage

```sql
create table recordings (
  id          uuid primary key default gen_random_uuid(),
  session_id  uuid references sessions(id),
  audio_uri   text not null,           -- r2://bucket/path.wav  (not the bytes)
  checksum    text not null,           -- integrity / chain-of-custody
  source      text,                    -- provenance label (vendor, program)
  duration_ms integer,
  sample_rate integer,
  metadata    jsonb not null default '{}',
  created_at  timestamptz not null default now()
);
```

### semantic_units — the parallel hub (one meaning, language-independent)

```sql
create table semantic_units (
  id             uuid primary key default gen_random_uuid(),
  canonical_text text,                 -- reference rendering of the meaning
  domain         text,                 -- 'travel', 'medical', …
  metadata       jsonb not null default '{}',
  created_at     timestamptz not null default now()
);
```

### utterances — the atom

```sql
create table utterances (
  id               uuid primary key default gen_random_uuid(),
  recording_id     uuid references recordings(id),
  speaker_id       uuid references speakers(id),
  language_id      text references languages(code),
  semantic_unit_id uuid references semantic_units(id),  -- NULL for monolingual data
  start_ms         integer,
  end_ms           integer,
  audio_uri        text not null,      -- segment clip (may differ from recording uri)
  checksum         text not null,
  duration_ms      integer,
  metadata         jsonb not null default '{}',
  created_at       timestamptz not null default now()
);
```

### annotations — typed, so new annotation kinds are rows, not tables

```sql
create table annotations (
  id           uuid primary key default gen_random_uuid(),
  utterance_id uuid references utterances(id),
  type         text not null,          -- 'orthographic' | 'ipa' | 'phonemic' | 'notes'
  content      text not null,
  annotator_id uuid,
  metadata     jsonb not null default '{}',
  created_at   timestamptz not null default now()
);
```

### Indexes (Postgres)

```sql
create index on utterances (semantic_unit_id);     -- parallel-set / pair derivation
create index on utterances (language_id);
create index on utterances (recording_id);
create index on annotations (utterance_id);
create index on sessions (speaker_id);
-- long-tail metadata querying:
create index on utterances using gin (metadata);
create index on recordings using gin (metadata);
```

## Deferred (explicitly out of v1, design later)

| Concern | Future table(s) | Note |
|---|---|---|
| Consent / chain of custody | `consents` | Carry `checksum`/`source` now; full consent records next. |
| Multi-pass QA | `qa_reviews` (pass #, rubric, reviewer, verdict) | Hangs off `utterances`/`annotations`. |
| Catalog | `datasets`, `dataset_members` (M:N) | HL-S2S, HL-Code membership. |
| Licensing | `licenses` | Usage rights, client, term. |
| Found/post-hoc alignment | `alignments` (source↔target utterance) | Only if non-script-driven data appears. |

## Flexibility rule (apply consistently)

> Will we **filter, join, or constrain** on this field? → typed column.
> Otherwise → `metadata` JSONB. When a JSONB key stabilizes and we want to index/validate
> it → graduate it to a typed column in a migration. One-way ratchet, low risk.

## Verification

Once a migration tool is chosen (e.g. Drizzle/Prisma for Postgres, or `drizzle-kit` over
D1), validate the design by seeding the worked example and running the three queries the
hub is meant to make trivial:

1. **Seed:** 2 semantic units, 3 speakers, 1 session/recording each, ~5 utterances
   (incl. one with `semantic_unit_id = NULL`), a couple of annotations.
2. **Parallel set:** `select language_id, audio_uri from utterances where semantic_unit_id = :su;`
   → returns all language versions of one meaning.
3. **Derive a training pair:** self-join `utterances` on `semantic_unit_id` filtering
   `language_id` source vs target → returns the (source_audio, target_audio) pair.
4. **Coverage gap:** `semantic_units` left of a `not exists` against `utterances` for a
   target language → lists meanings missing that language.
5. **Metadata query:** filter on a `metadata->>'...'` key to confirm the JSONB escape
   hatch works (and is GIN-indexed on Postgres).

Passing all five confirms the schema supports parallel S2S, monolingual data, and flexible
metadata without further structural change.

## Suggested next steps (after approval)

1. Pick the engine (recommend Postgres) and a migration tool.
2. Translate this DDL into the tool's schema format; run the verification seed + queries.
3. Decide where `consents` slots in (likely the very next table given the provenance pillar).
