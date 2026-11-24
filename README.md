# Interactive-Rag: RDoC clinical analysis with retrieval-augmented generation

A Streamlit app that turns a clinician's description of a presentation into a structured analysis in the NIMH **Research Domain Criteria (RDoC)** framework. It maps the description to RDoC domains and constructs, lists the units of analysis and the assessments to consider, then retrieves supporting passages from a knowledge base and has **Claude on Amazon Bedrock** write an analysis that cites them and asks one follow-up question. Each answer goes back into the analysis.

> Research prototype for clinical decision support. It does not diagnose and is not a medical device.

## How it works

```text
presentation ──► RDoC matcher ──► domains, constructs, units of analysis, tests ──┐
     ▲                                                                            │
     │                     retriever (Bedrock Knowledge Base or Kendra) ◄─────────┤
     │                                  │ numbered passages                       │
     │                                  ▼                                         │
answer to the ◄── follow-up ◄── Claude on Amazon Bedrock (Converse API) ◄─────────┘
follow-up         question      grounded analysis that cites the passages
```

1. **Match.** Words and phrases in the presentation are matched against a curated RDoC matrix (Negative Valence, Positive Valence and Cognitive Systems): construct names, behaviors, molecules and common symptom terms such as anxiety or ADHD. This step always runs, with no cloud services.
2. **Retrieve.** The matched constructs and the clinician's text query an Amazon Bedrock Knowledge Base, or an existing Amazon Kendra index, for passages.
3. **Generate.** Claude receives the conversation, the RDoC matches and the numbered passages, writes an analysis that cites passages as [1], [2], and asks one follow-up question.
4. **Refine.** The clinician's answer is added to the conversation and steps 1 to 3 run again.

Without AWS credentials the app runs the matching step on its own, and the sidebar shows which services are connected.

| Part | File |
| --- | --- |
| Streamlit app: conversation, results, follow-up loop | `app.py` |
| RDoC matching and retrieval from a knowledge base or Kendra | `src/analyzers/rdoc_analyzer.py` |
| RDoC matrix | `src/data/rdoc_matrix.json` |
| AWS settings and the Claude call (Bedrock Converse API) | `src/utils/model.py` |
| System prompt and prompt templates | `src/utils/prompts.py` |
| Kendra index that crawls the NIMH RDoC pages, for accounts that already use Kendra | `kendra-template.yaml`, `kendra-policy.json` |

## Run it

Requirements: Python 3.10 or newer.

```bash
git clone https://github.com/fahadalsehami/Interactive-Rag.git
cd Interactive-Rag
pip install -r requirements.txt
streamlit run app.py
```

That runs the RDoC matching on its own. To add Claude and retrieval:

1. **AWS credentials.** Configure them with `aws configure`, AWS IAM Identity Center or an IAM role, and make sure your account can invoke the Claude model you choose in Amazon Bedrock.
2. **Knowledge base (optional).** In the Amazon Bedrock console, create a knowledge base with a Web Crawler data source whose seed URL is `https://www.nimh.nih.gov/research/research-funded-by-nimh/rdoc`, sync it, and note its ID.
3. **Settings.** Copy the example file and fill in what you use:

   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```

| Setting | Purpose | Default |
| --- | --- | --- |
| `AWS_REGION` | Region for Bedrock and retrieval | your AWS configuration, else `us-east-1` |
| `BEDROCK_MODEL_ID` | Claude model or inference profile | `global.anthropic.claude-sonnet-4-6` |
| `KNOWLEDGE_BASE_ID` | Amazon Bedrock Knowledge Base to retrieve from | none |
| `KENDRA_INDEX_ID` or `KENDRA_STACK_NAME` | Existing Kendra index, or the stack made from `kendra-template.yaml` | `clinical-rag-kendra` |

Use a geographic inference profile such as `us.anthropic.claude-sonnet-4-6` to keep inference in one region group. Amazon Kendra closed to new customers on July 30, 2026, so new accounts should use a Bedrock Knowledge Base; accounts that already use Kendra can deploy the template with `aws cloudformation deploy --template-file kendra-template.yaml --stack-name clinical-rag-kendra --capabilities CAPABILITY_IAM`.

`.streamlit/secrets.toml` is ignored by git. Never commit credentials.

## Stack

Python, Streamlit, Amazon Bedrock (Claude, Converse API), Amazon Bedrock Knowledge Bases, Amazon Kendra, LangChain prompt templates, boto3, AWS CloudFormation

## License

MIT, see [LICENSE](LICENSE).
