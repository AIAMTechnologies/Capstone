# Email Intelligence & Lead Sync — Codex Handoff

## Project Overview
WFC (Window Film Canada) Lead Administration Portal. A FastAPI + React/TypeScript app that manages dealer leads. The **Email Intelligence** feature connects to Colin MacLeod's Microsoft Outlook mailbox (`cmacleod@windowfilmcanada.ca`) via MS Graph API, syncs emails, matches them to leads in the database, and uses AI to analyze matched emails.

## Architecture
- **Backend**: FastAPI (Python 3.12), PostgreSQL on Azure, port 8000
- **Frontend**: React + TypeScript + Vite, port 3001
- **AI**: OpenAI gpt-4.1-nano (cheapest/fastest) for email analysis, gpt-4.1-mini for reasoning
- **Email**: MS Graph API with OAuth2 delegated permissions, single-tenant
- **Token encryption**: Fernet (AES-256) for storing OAuth tokens in DB

## How to Start
```bash
# Backend (must use root .venv, not backend/.venv which has broken symlinks)
cd backend && /path/to/.venv/bin/python3 -m uvicorn main:app --reload --port 8000

# Frontend
cd frontend && npm run dev -- --port 3001
```
- The `.env` file lives at the project root. The `backend/` dir needs a symlink: `ln -sf ../.env backend/.env`
- The `backend/.venv` has broken symlinks (created on a different machine). Use the root `.venv` instead.

## Key Files
- `backend/routes/email_intel.py` — Core module (~1400 lines). All email sync, matching, AI analysis, and auto-lead-creation logic.
- `frontend/src/pages/admin/EmailIntel.tsx` — Email Intel admin page with sync button, status, review queue, cost tracking.
- `frontend/src/services/api.ts` — API service (30s timeout for email-intel, 120s for sync).
- `backend/db.py` — PostgreSQL connection helper (`execute_query`).

## What Was Built

### 1. Email Sync via MS Graph API
- OAuth2 flow: authorization code → access token → refresh token (stored encrypted in `email_sync_config` table)
- Fetches emails with `$top=1000` per page, includes full `body` content (HTML) + `conversationId`
- Detects email direction by comparing sender to mailbox email (not folder-based)
- Strips HTML tags from body for plain-text matching

### 2. Lead Matching Strategy (5-tier + auto-create)
All matching is done **in-memory** — leads are pre-loaded into lookup dicts once per sync (`_load_leads_cache`), zero DB queries per email.

**Match priority:**
1. **Email match** (confidence 1.0) — other party's email matches `leads.email`
2. **Dealer email match** (0.95) — matches `leads.dealer_email`
3. **Name match** (0.85) — sender name matches `leads.first_name + last_name`
4. **Body email match** (0.90) — email address found in body text matches a lead
5. **Body name match** (0.85) — "Name: First Last" pattern in body matches a lead
6. **Phone match** (0.75) — 10+ digit phone in body matches a lead (strict: no all-zeros, no repeated digits)

**CRITICAL RULE**: The mailbox owner (Colin) is always excluded from matching. We match the OTHER party — the customer or dealer Colin is communicating with.

### 3. Auto-Create Leads from Form Submissions
When Colin forwards a Squarespace/HubSpot form submission to a dealer and the customer doesn't exist in the leads DB, the system auto-creates a lead with `source: "Email Auto-Created"`. Extracts: name, email, phone, city, province, company, project type, and which dealer it was forwarded to.

The new lead is immediately added to the in-memory cache so subsequent emails (and thread propagation) can match it.

### 4. Thread Propagation via `conversationId`
After all pages are synced, `_propagate_thread_matches()` runs. If any email in a conversation thread is matched to a lead, ALL unmatched emails in that same thread get matched to the same lead (confidence 0.80, method "thread"). This is the key insight: Colin forwards a lead to a dealer → dealer replies → all those replies are in the same conversation thread.

### 5. AI Analysis (parallel batches)
Only **matched** emails get AI analysis (unmatched are skipped to save cost). Uses `asyncio.gather()` to process all AI calls for a page concurrently. The AI extracts: summary, sentiment, deal stage, action items.

**Model fallback system**: Tries preferred model first (gpt-4.1-nano), falls back to gpt-4o-mini if unavailable. Uses `max_completion_tokens` for newer models, `max_tokens` for older ones.

**Cost tracking**: Tracks input/output tokens per model, converts USD to CAD (rate 1.44), displayed on frontend. Cost display turns red when > $0.50 CAD.

### 6. Closure Candidate Detection
After sync, checks for leads that AI flagged as "closing" or "closed" deal stage and marks them for review.

## Database Schema (relevant tables)

### `email_messages`
```
id, ms_message_id, conversation_id, subject, sender_email, sender_name,
recipient_emails (JSON), body_preview, body_text, received_at, is_read,
direction, folder, matched_lead_id (FK → leads.id), match_confidence,
match_method, ai_summary, ai_sentiment, ai_action_items, ai_ready_to_close,
ai_close_reasoning, processed_at, created_at
```

### `leads` (relevant columns)
```
id, name, first_name, last_name, email, phone, city, province,
dealer_email, source, status, email_match_count, last_email_activity,
email_sentiment, ai_closure_flagged
```

### `email_sync_config`
```
id, client_id, tenant_id, client_secret (encrypted), access_token (encrypted),
refresh_token (encrypted), user_email, last_sync_at, is_active
```

## Current State (as of 2026-03-20)
- Sync runs successfully through page 1 (~1000 emails), auto-creates leads, matches ~71 emails with AI analysis
- Error handling catches FK violations and continues (doesn't crash on bad lead references)
- Thread propagation runs after all pages complete
- Frontend shows sync progress, matched emails, review queue, and AI cost in CAD

## Known Issues / What Needs Fixing

### 1. Sync stops after page 1 crash (PRIORITY)
The sync was crashing on FK violations when `matched_lead_id` referenced a lead that didn't exist (from a previous failed sync's auto-create). Error handling was just added but needs verification that it continues through ALL pages (page 2, 3, etc.) without stopping.

**To verify**: Run a full sync and confirm all pages complete. Check logs for:
```
[EMAIL_INTEL] Page 1 complete: total synced=X, matched=Y
[EMAIL_INTEL] Page 2 complete: ...
[EMAIL_INTEL] inbound sync finished: ...
[EMAIL_INTEL] Syncing sent mail...
[EMAIL_INTEL] Thread propagation: N additional matches
```

### 2. Body text extraction could be better
Currently strips HTML tags with regex (`<[^>]+>` → space). Some emails have complex HTML that leaves artifacts. Consider using `BeautifulSoup` or `html2text` for cleaner extraction.

### 3. Auto-create leads need name AND email
Changed to require both `customer_first` AND `customer_email` (was OR before, which created leads with `email=None`). Some form submissions have different formats that the regex doesn't catch:
- HubSpot forms: different field names
- `lead@windowfilmcanadadealers.ca` dealer site submissions: different format entirely
- Need to add more regex patterns for these formats

### 4. Graph API `$top=1000` with `body` content may be slow
Adding `body` to `$select` fetches full HTML content for every email. For 1000 emails per page this is a lot of data. Consider:
- Only fetching `bodyPreview` in the initial pass
- Doing a second pass to fetch full `body` only for emails that need deeper matching (unmatched after preview-based matching)

### 5. Thread propagation only runs once at the end
If page 1 creates a lead and page 3 has emails in the same thread, those page 3 emails won't match during their page's matching phase — they'll only match during the final propagation. This is fine functionally but means the AI analysis won't run on thread-propagated emails (since AI only runs on emails matched during Phase 1).

### 6. Review Queue / Recent Activity UI
The "Recent Email Activity" section and "Review Queue" on the frontend need testing with real matched data. The review queue should show emails where `ai_ready_to_close = true`.

### 7. Cost monitor display
The AI cost tracker resets each server restart (it's in-memory via `_cost_tracker` dict). Consider persisting to DB if you want cumulative cost tracking across syncs.

## Next Steps (in order of priority)

1. **Verify full sync completes** — Run sync, confirm it processes ALL pages (inbox + sent items), thread propagation runs, and final counts are accurate.

2. **Improve form submission parsing** — Add regex patterns for:
   - HubSpot form format: `"Standard Form - Landing Page (WFC)"` submissions
   - Dealer website contact forms from `lead@windowfilmcanadadealers.ca`
   - Handle accented characters in names (e.g., `Nicolas Kébreau` — the current regex `[A-Z][a-z]+` won't match accented chars)

3. **Re-sync leads from Lasso** — The leads DB only has 3 leads created in the last 7 days but Colin is forwarding new leads daily. Run `sync_from_lasso.py` to get the latest leads, which will dramatically increase matches without auto-create.

4. **Subject-line lead extraction** — Colin's lead assignment emails follow patterns like:
   - `"3M Lead - Kitchener, ON - 03/17/2026"`
   - `"3M Sun Control lead - London - 03/19/2026"`
   - `"3M Security Film lead - Edmonton - 03/19/2026"`
   The city from the subject could be used to narrow down leads when body matching fails.

5. **Spam/newsletter filtering** — Skip AI analysis for obvious spam: LinkedIn, LCBO, Michael Kors, Golf Gods, Aeroplan, GolfNorth, etc. These waste AI tokens. Add a domain blocklist.

6. **Dealer lookup enrichment** — Colin emails ~15 dealers regularly. Building a `dealer` table mapping dealer emails to dealer names/companies would let us track which dealer handles which leads and show dealer-specific email threads.

7. **Incremental sync** — Currently re-fetches all emails. Use `receivedDateTime gt {last_sync_time}` filter in Graph API URL to only fetch new emails since last sync.

8. **Push to branch** — All work is on the `email-intelligence-lead-sync` branch in a Claude worktree at `.claude/worktrees/dreamy-mcnulty/`. Changes need to be committed and pushed.

## Environment Variables (in .env)
```
DATABASE_URL=postgresql://...@...azure.com:5432/...
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
EMAIL_ENCRYPTION_KEY=... (Fernet key for encrypting OAuth tokens)
OPENAI_EMAIL_MODELS=gpt-4.1-nano,gpt-4o-mini
OPENAI_REASONING_MODELS=gpt-4.1-mini,gpt-4o-mini
```

## Business Context
Colin MacLeod is the VP of Sales at Window Film Canada. The lead flow is:
1. **Customer** fills out a form on the WFC website (Squarespace) or a dealer website
2. **Colin** receives the form submission, identifies the right dealer for the region
3. **Colin** forwards the lead to the **dealer** via email
4. **Dealer** follows up with the customer and reports back to Colin

So Colin's mailbox contains: form submissions (inbound), lead forwarding (outbound to dealers), dealer replies (inbound), and general business correspondence + spam. The goal is to connect each email thread to the original customer lead so the admin portal can show the full communication history per lead.
