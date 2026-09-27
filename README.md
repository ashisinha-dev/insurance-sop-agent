# Insurance Claims SOP Agent

A conversational insurance claims support application that combines a fixed Standard Operating Procedure (SOP) with natural-language understanding.

The agent follows a controlled insurance workflow while still allowing users to speak naturally, provide information across multiple messages, ask follow-up questions, and request human support.

## Workflow

The application follows four main phases:

`VERIFY_ID -> RESOLVE_INTENT -> PROCESS_CASE -> POST_PROCESS`

### VERIFY_ID

Identity verification is enforced before any claim information is accessed.

A caller must provide at least three matching identity fields from:

- Full name
- Date of birth
- Phone number
- Email address
- SSN or ID last four digits

A policy number can be used to locate the policyholder record, but it does not count toward the three required identity matches.

The verification flow supports partial responses, alternate identity fields, refusals, and information provided across multiple messages.

### RESOLVE_INTENT

After identity verification, the application identifies the caller's request and determines the relevant claim.

Claim-related information mentioned earlier in the conversation is preserved. For example, if a caller mentions a denied healthcare claim from January while still completing verification, those details are remembered and used after verification succeeds.

### PROCESS_CASE

The agent answers claim-specific questions using the supplied claim data, document guidance, and field definitions.

This phase supports natural follow-up questions while keeping responses grounded in the available insurance data.

### POST_PROCESS

After case handling, the customer is offered an email summary containing the main discussion, claim status or outcome, and relevant follow-up information.

The customer can choose whether to receive the summary or skip it.

## SOP Controls

The workflow keeps important business rules deterministic while allowing flexible language understanding where appropriate.

The application controls:

- Identity verification
- Privacy gating before claim access
- SOP phase transitions
- Claim selection
- Email consent
- Out-of-scope handling
- Human escalation

The language model is used for:

- Natural-language information extraction
- Intent interpretation
- Claim hints
- Emotion and refusal recognition
- Ambiguous user language
- Grounded claim explanations

## Cross-Phase Memory

Information is stored when the caller provides it, even if it belongs to a later phase.

For example:

```text
I'm Margaret Chen. My DOB is 1985-03-15.
I'm calling about my denied healthcare claim from January.
```

The system remains in `VERIFY_ID` until enough identity information has been verified, while retaining the claim type, month, status, and intent for later use.

## Emotional Support and Recovery

The agent recognizes conversational signals such as frustration, anger, anxiety, confusion, and refusal.

It can acknowledge the caller's concern, explain why required verification steps are necessary, offer alternate identity fields, and offer human support when appropriate without bypassing required SOP gates.

## Out-of-Scope Handling

The assistant is limited to insurance claims and related customer-service requests.

Unrelated questions are rejected politely. Repeated out-of-scope requests result in an offer to connect the caller with a human representative.

## Demo Scenario

Example caller message:

```text
I'm the policyholder. My name is Margaret Chen, policy POL-9921.
I'm calling about my denied healthcare claim from January.
DOB is 1985-03-15, SSN last four is 4472.
```

The application:

1. Extracts identity and claim information.
2. Verifies the caller using three matching PII fields.
3. Preserves the denied healthcare claim information mentioned during verification.
4. Resolves the appropriate claim after verification.
5. Answers using the supplied claim and document data.
6. Offers an email summary during post-processing.

## Project Structure

```text
apps/
  insurance_claims/
    app.py
    llm.py
    sop_engine.py
    tools.py
    fixtures/
      policyholders.json
      claims.json
      representatives.json
      required_document_guideline.json
      claim_schema.json
      consent_scenarios.json

tests/
  conftest.py
  test_insurance_sop.py

README.md
requirements.txt
Dockerfile
.dockerignore
.env.example
.gitignore
```

## Setup

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file using `.env.example`:

```text
MODEL_API_KEY=your_api_token_here
MODEL_NAME=your_model_name_here
```

An OpenAI-compatible custom endpoint can also be configured with:

```text
MODEL_BASE_URL=your_base_url
```

## Run Locally

```bash
streamlit run apps/insurance_claims/app.py
```

The application will be available at:

```text
http://localhost:8501
```

## Tests

Run:

```bash
pytest -q
```

Current test suite:

```text
5 passed
```

## Docker

Build the image:

```bash
docker build -t insurance-sop-agent .
```

Run the container:

```bash
docker run --rm -p 8501:8501 --env-file .env insurance-sop-agent
```

Then open:

```text
http://localhost:8501
```

## Hosted Demo

Hosted application URL: `TO_BE_ADDED`

## Demo Integrations

Email delivery and human representative transfer are simulated within the application. Claim and policyholder information is loaded from the supplied fixture data.