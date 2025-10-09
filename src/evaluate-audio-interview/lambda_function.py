import os
import json
import boto3
import requests  # Needed for API Gateway call
from decimal import Decimal
from openai import OpenAI
from botocore.exceptions import ClientError
import urllib.parse
from datetime import datetime, timezone
import re
from difflib import SequenceMatcher

# =============================
# OpenAI + AWS clients
# =============================
# def get_secret(key):
#     secret_name = f"ecs/agent-example/{key}"
#     region_name = "eu-west-2"
#     session = boto3.session.Session()
#     client = session.client(service_name='secretsmanager', region_name=region_name)
#     try:
#         response = client.get_secret_value(SecretId=secret_name)
#         return response['SecretString']
#     except ClientError as e:
#         raise e

# Load OpenAI key (same as your working version)
# os.environ['OPENAI_API_KEY'] = get_secret("openai-key")
# openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Connect to DynamoDB
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.getenv("R_TABLE"))
job_table = dynamodb.Table(os.getenv("J_TABLE"))

# S3 + Events
s3 = boto3.client('s3')
events_client = boto3.client('events')

# =============================
# Scoring weights / rubric
# =============================
SCORING_CRITERIA = {
    "technical_competency": {"weight": 0.35, "description": "Depth of technical knowledge, problem-solving ability, and domain expertise relevant to the role"},
    "communication_skills": {"weight": 0.20, "description": "Clarity of expression, articulation, listening skills, and ability to explain complex concepts"},
    "behavioral_fit": {"weight": 0.15, "description": "Alignment with company values, teamwork, adaptability, and cultural fit"},
    "problem_solving": {"weight": 0.15, "description": "Analytical thinking, creativity in solutions, and structured approach to challenges"},
    "experience_relevance": {"weight": 0.10, "description": "How well past experience aligns with job requirements and potential for growth"},
    "professionalism": {"weight": 0.05, "description": "Professional demeanor, confidence, preparation, and interview etiquette"}
}

# =============================
# Helpers (robust scoring + coverage)
# =============================
def clamp_0_100(x) -> int:
    try:
        return max(0, min(100, int(round(float(x)))))
    except Exception:
        return 0

def weighted_overall_from_detailed(detailed_scores: dict) -> int:
    total = 0.0
    for k, meta in SCORING_CRITERIA.items():
        total += float(detailed_scores.get(k, 0)) * float(meta["weight"])
    return clamp_0_100(total * 10)

def determine_hiring_recommendation(score):
    if score >= 80:
        return "STRONG_HIRE"
    elif score >= 70:
        return "HIRE"
    elif score >= 50:
        return "CONSIDER"
    return "NO_HIRE"

def clean_json_output(raw):
    raw = (raw or "").strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    return raw.strip()

# ---------- Role-based Q/A extraction (preferred path) ----------
def extract_qa_pairs_from_items(transcript_payload):
    """
    If transcript is a dict with 'items' having 'role' and 'content', build Q/A pairs by roles:
      - assistant/system/interviewer/hr/recruiter => QUESTION
      - user/candidate/anything else              => ANSWER (accumulate until next question turn)
    Returns list[(question:str, answer:str)].
    """
    if not isinstance(transcript_payload, dict) or not isinstance(transcript_payload.get("items"), list):
        return []

    items = transcript_payload["items"]

    def _get_text(x):
        c = x.get("content")
        if isinstance(c, list):
            return " ".join(t for t in c if isinstance(t, str)).strip()
        if isinstance(c, str):
            return c.strip()
        for k in ("text", "value", "utterance"):
            if isinstance(x.get(k), str):
                return x[k].strip()
        return ""

    qa_pairs = []
    current_q = None
    current_a_chunks = []

    for turn in items:
        role = (turn.get("role") or "").lower()
        text = _get_text(turn)
        if not text:
            continue

        if role in ("assistant", "system", "interviewer", "hr", "recruiter"):
            # flush previous Q/A
            if current_q is not None:
                qa_pairs.append((current_q, " ".join(current_a_chunks).strip()))
                current_a_chunks = []
            current_q = text
        else:
            if current_q is not None:
                current_a_chunks.append(text)
            else:
                # leading user chatter before first Q: ignore
                pass

    if current_q is not None:
        qa_pairs.append((current_q, " ".join(current_a_chunks).strip()))

    cleaned = []
    for q, a in qa_pairs:
        q = (q or "").strip()
        a = (a or "").strip()
        if q:
            cleaned.append((q, a))
    return cleaned

# ---------- Fallback transcript handling (when roles are missing) ----------
def parse_transcript(transcript_payload):
    """
    Fallback: normalize to lines prefixed with Q:/A: if only plain text is available.
    """
    if isinstance(transcript_payload, str):
        return transcript_payload.strip()

    if isinstance(transcript_payload, dict) and isinstance(transcript_payload.get("items"), list):
        lines = []
        for it in transcript_payload["items"]:
            role = (it.get("role") or "").lower()
            texts = it.get("content") or []
            if isinstance(texts, list):
                text = " ".join(t for t in texts if isinstance(t, str)).strip()
            elif isinstance(texts, str):
                text = texts.strip()
            else:
                text = ""
            if not text:
                continue
            if role in ("assistant", "system", "interviewer", "hr", "recruiter"):
                lines.append(f"Q: {text}")
            else:
                lines.append(f"A: {text}")
        return "\n".join(lines).strip()

    return str(transcript_payload).strip()

_INTERROGATIVE_RE = re.compile(
    r"^(who|what|when|where|why|how|could|can|would|should|do|does|did|tell me|please elaborate|could you please|can you|would you|certainly|as a co-founder)\b",
    re.IGNORECASE
)

def _looks_like_question(q: str) -> bool:
    q = (q or "").strip()
    if not q:
        return False
    return q.endswith("?") or bool(_INTERROGATIVE_RE.match(q))

def split_multi_part(q: str):
    parts = [p.strip() for p in q.split("?")]
    parts = [p + "?" for p in parts if p]
    return parts if len(parts) > 1 else [q]

def _norm_q(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s.rstrip("?.! ")

def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm_q(a), _norm_q(b)).ratio()

def _near_dup(a: str, b: str, th=0.90) -> bool:
    return _sim(a, b) >= th

def dedupe_adjacent_questions(qs):
    out = []
    for q in qs:
        if out and _near_dup(out[-1], q):
            continue
        out.append(q)
    return out

def extract_questions_from_transcript(transcript_text: str):
    """
    Fallback: find 'Q:' lines, split multi-part, de-dupe near-duplicates.
    """
    questions = []
    for line in (transcript_text or "").splitlines():
        s = line.strip()
        if not s.lower().startswith("q:"):
            continue
        q = s.split(":", 1)[1].strip()
        if not q:
            continue
        if _looks_like_question(q):
            for part in split_multi_part(q):
                if _looks_like_question(part):
                    questions.append(part)
    return dedupe_adjacent_questions(questions)

def get_answers_per_question(transcript_text: str, questions: list):
    """
    Fallback Q->A slicing:
      - For each 'Q:' block, gather ALL non-empty lines until next 'Q:'.
      - Assign the same span to each sub-question from that block.
    """
    lines = (transcript_text or "").splitlines()
    q_line_indices = [i for i, l in enumerate(lines) if l.strip().lower().startswith("q:")]

    # Build split-question lists for each original Q line
    split_lists = []
    for oi in q_line_indices:
        q_text = lines[oi][2:].strip()
        parts = [p for p in split_multi_part(q_text) if _looks_like_question(p)]
        split_lists.append(parts)

    # Build answer spans per original Q block
    spans_text = []
    for idx, oi in enumerate(q_line_indices):
        start = oi + 1
        end = q_line_indices[idx + 1] if (idx + 1) < len(q_line_indices) else len(lines)
        answer_lines = []
        for l in lines[start:end]:
            s = l.strip()
            if not s:
                continue
            if s.lower().startswith("q:"):
                continue
            if s.lower().startswith("a:"):
                answer_lines.append(s[2:].strip())
            else:
                answer_lines.append(s)
        spans_text.append(" ".join(answer_lines).strip())

    # Assign spans to sub-questions
    answers_map = {}
    for span_text, parts in zip(spans_text, split_lists):
        for p in parts:
            answers_map[p] = span_text

    for q in questions:
        answers_map.setdefault(q, "")

    return answers_map

# ---------- Coverage enforcement (position-first + fuzzy fallback) ----------
def enforce_question_coverage(result: dict, expected_questions: list):
    model_qe = [x for x in (result.get("question_evaluations") or []) if isinstance(x, dict)]
    fixed = []
    used = [False] * len(model_qe)

    for idx, q in enumerate(expected_questions, start=1):
        chosen = None
        if idx - 1 < len(model_qe):
            cand = model_qe[idx - 1]
            if isinstance(cand, dict) and cand.get("question"):
                chosen = cand
                used[idx - 1] = True
        if not chosen:
            best_i, best_score = -1, 0.0
            for i, cand in enumerate(model_qe):
                if used[i] or not isinstance(cand, dict):
                    continue
                score = _sim(q, cand.get("question", ""))
                if score > best_score:
                    best_i, best_score = i, score
            if best_score >= 0.60 and best_i >= 0:
                chosen = model_qe[best_i]
                used[best_i] = True
        if not chosen:
            chosen = {
                "answer_summary": "",
                "individual_score": 0,
                "key_strengths": [],
                "areas_for_improvement": [],
                "justification": "NO_EVIDENCE",
                "evidence": ["NO_EVIDENCE"],
            }
        item = dict(chosen)
        item["question_number"] = idx
        item["question"] = q
        item.setdefault("answer_summary", "")
        item.setdefault("individual_score", 0)
        item.setdefault("key_strengths", [])
        item.setdefault("areas_for_improvement", [])
        item.setdefault("justification", item.get("justification", ""))
        item.setdefault("evidence", ["NO_EVIDENCE"])
        fixed.append(item)

    result["question_evaluations"] = fixed
    return result

def calculate_combined_score(interview_data):
    try:
        data = json.loads(interview_data) if isinstance(interview_data, str) else interview_data
        detailed = data.get("detailed_scores", {}) or {}
        if detailed:
            return weighted_overall_from_detailed(detailed)
        communication_score = data.get("detailed_scores", {}).get("communication_skills", 0)
        question_scores = [q.get("individual_score", 0) for q in data.get("question_evaluations", [])]
        all_scores = [communication_score] + question_scores
        if all_scores:
            average_score = sum(all_scores) / float(len(all_scores))
            return clamp_0_100(average_score * 10)
        return 0
    except Exception:
        return 0

# =============================
# JD fetch + prompt builder
# =============================
def get_job_description(job_id):
    try:
        response = job_table.get_item(Key={"id": job_id})
        job_item = response.get("Item", {})
        if not job_item:
            raise ValueError(f"No job found for jobId: {job_id}")
        job_description = job_item.get("jobDescriptionMarkdown", "")
        if not job_description:
            raise ValueError(f"Missing jobDescriptionMarkdown for jobId: {job_id}")
        return job_description, job_item
    except Exception as e:
        print(f"Error retrieving job description for jobId {job_id}: {str(e)}")
        raise e

def _criteria_text():
    return "\n".join([
        f"- {k.upper()}: {v['description']} (Weight: {int(v['weight'] * 100)}%)"
        for k, v in SCORING_CRITERIA.items()
    ])

def build_enhanced_prompt(resume, jd, must_haves, qa_pairs):
    must_have_block = "\n".join([f"- {s}" for s in (must_haves or [])]) if must_haves else "None provided."
    qa_block_lines = []
    for i, (q, a) in enumerate(qa_pairs, start=1):
        qa_block_lines.append(f"QUESTION {i}: {q}\nANSWER:\n{a if a else '[NO_ANSWER_FOUND]'}")
    qa_block = "\n\n".join(qa_block_lines) if qa_block_lines else "None detected."
    criteria_details = _criteria_text()
    return f"""
You are an expert interview evaluator. Respond ONLY with JSON that conforms exactly to the schema.
You will evaluate EACH question independently using ONLY the provided ANSWER text for that question.

RESUME:
{resume}

JOB DESCRIPTION:
{jd}

MUST-HAVE SKILLS:
{must_have_block}

INTERVIEW (structured Q/A pairs):
{qa_block}

SCORING CRITERIA (each scored 0–10; overall is weighted average -> 0–100):
{criteria_details}

SCORING GUIDE (per question):
- 0   = No relevant answer text at all (ANSWER empty or entirely off-topic)
- 1–3 = Very minimal/vague; touches topic but lacks specifics or examples
- 4–6 = Partially complete; some specifics/examples but missing depth or steps
- 7–8 = Good; mostly complete with relevant specifics and clear reasoning
- 9–10= Excellent; comprehensive, specific, structured, and clearly justified

IMPORTANT RULES:
- If the ANSWER is empty ("[NO_ANSWER_FOUND]") or contains no relevant content, set "individual_score" = 0 and "justification" = "NO_EVIDENCE".
- If ANY relevant content exists (even if partial), DO NOT return "NO_EVIDENCE". Give partial credit per the guide and include short quotes in "evidence".
- Evidence must be short quotes taken ONLY from the ANSWER span of that question.
- Keep 'question_evaluations' in the SAME ORDER as provided.

RESPONSE SCHEMA:
{{
  "response_id": "string",
  "candidate_name": "string",
  "overall_interview_score": 85,
  "hiring_recommendation": "STRONG_HIRE",
  "interview_summary": "string",
  "detailed_scores": {{
    "technical_competency": 0,
    "communication_skills": 0,
    "behavioral_fit": 0,
    "problem_solving": 0,
    "experience_relevance": 0,
    "professionalism": 0
  }},
  "question_evaluations": [
    {{
      "question_number": 1,
      "question": "string",
      "answer_summary": "string",
      "individual_score": 0,
      "key_strengths": ["string"],
      "areas_for_improvement": ["string"],
      "justification": "string",
      "evidence": ["short quotes or NO_EVIDENCE"]
    }}
  ],
  "overall_assessment": {{
    "top_strengths": ["string"],
    "key_concerns": ["string"],
    "role_fit_analysis": "string",
    "development_areas": ["string"]
  }}
}}
STRICT RULES:
- JSON only. No markdown. No prose outside JSON.
- The 'question_evaluations' MUST appear in the SAME ORDER as the provided QUESTION list.
"""

def validate_and_enhance_result(result_json, original_data, context=None):
    result = json.loads(result_json)
    detailed = result.get("detailed_scores", {}) or {}
    weighted_overall = weighted_overall_from_detailed(detailed)
    result["overall_interview_score"] = clamp_0_100(weighted_overall)
    result["hiring_recommendation"] = determine_hiring_recommendation(result["overall_interview_score"])
    result["evaluation_timestamp"] = context.aws_request_id if context and hasattr(context, "aws_request_id") else datetime.now(timezone.utc).isoformat()
    result["scoring_version"] = "2.0"
    result.setdefault("question_evaluations", [])
    result.setdefault("overall_assessment", {})
    return result

# ---------- Optional minimal partial-credit safety net ----------
def apply_min_partial_credit(validated_result, answers_map, min_score=2, min_chars=30):
    """
    If a question has a non-empty ANSWER span but the model returned score=0 and NO_EVIDENCE,
    bump to a tiny partial score and attach a short evidence snippet.
    """
    qe = validated_result.get("question_evaluations", []) or []
    fixed = []
    for item in qe:
        q = item.get("question", "")
        ans = (answers_map.get(q, "") or "").strip()
        score = int(item.get("individual_score", 0) or 0)
        evidence = item.get("evidence") or []
        justification = (item.get("justification") or "").strip()

        if len(ans) >= min_chars and score == 0 and ("NO_EVIDENCE" in evidence or justification == "NO_EVIDENCE"):
            snippet = ans[:180].strip()
            item["individual_score"] = max(min_score, 1)
            item["justification"] = "Partial evidence found (auto-adjusted); answer minimal/unspecific."
            item["evidence"] = [snippet + ("..." if len(ans) > len(snippet) else "")]
            item.setdefault("key_strengths", [])
            item.setdefault("areas_for_improvement", ["Be specific; include metrics/tools/examples."])
        fixed.append(item)

    validated_result["question_evaluations"] = fixed
    return validated_result

# =============================
# EventBridge rule delete (same as yours)
# =============================
def delete_event_scheduler(response_id):
    try:
        rule_name = f'interview-reminder-{response_id}'
        targets_response = events_client.list_targets_by_rule(Rule=rule_name)
        if targets_response.get('Targets'):
            target_ids = [target['Id'] for target in targets_response['Targets']]
            events_client.remove_targets(Rule=rule_name, Ids=target_ids)
        events_client.delete_rule(Name=rule_name)
        return {
            'statusCode': 200,
            'body': json.dumps({'message': f'Successfully deleted rule: {rule_name}'})
        }
    except events_client.exceptions.ResourceNotFoundException:
        return {'statusCode': 404, 'body': json.dumps({'error': f'Rule {rule_name} not found'})}
    except Exception as e:
        return {'statusCode': 500, 'body': json.dumps({'error': f'Error deleting rule: {str(e)}'})}

# =============================
# Core processing
# =============================
def process_single_record(record, context):
    """Process a single S3 record (keeps your original flow)"""
    try:
        # Step 1: Extract bucket + key
        bucket = record['s3']['bucket']['name']
        key = urllib.parse.unquote_plus(record['s3']['object']['key'])

        # Step 2: Get metadata
        head = s3.head_object(Bucket=bucket, Key=key)
        metadata = head.get("Metadata", {})
        file_type = metadata.get("audio", "false").lower()
        is_audio = file_type == "true"
        response_id = metadata.get("response_id")

        if not response_id:
            print(f"Missing response_id in metadata for file: {key}")
            return {"statusCode": 400, "body": f"Missing response_id in metadata"}

        # Step 3: Get resume and jobId from DynamoDB
        response_ = table.get_item(Key={"id": response_id})
        item = response_.get("Item", {})
        if not item:
            raise ValueError(f"No item found in DynamoDB for response_id: {response_id}")

        resume = item.get("resumeText", "")
        if not resume:
            raise ValueError(f"Resume missing in DynamoDB for response_id: {response_id}")

        job_id = item.get("jobId")
        if not job_id:
            raise ValueError(f"Missing jobId for response_id: {response_id}")

        # Step 4: Get job description + job item from job table
        job_description, job_item = get_job_description(job_id)

        # Step 5: Load transcript from the uploaded file
        s3_object = s3.get_object(Bucket=bucket, Key=key)
        content = s3_object['Body'].read().decode('utf-8')
        transcript_data = json.loads(content)
        transcript_raw = transcript_data.get("transcript", "")

        # ---------- Prefer role-based Q/A if available ----------
        qa_pairs = extract_qa_pairs_from_items(transcript_raw)
        if qa_pairs:
            expected_questions = [q for q, _ in qa_pairs]
            answers_map = {q: a for q, a in qa_pairs}
        else:
            # fallback to line-based parsing
            transcript = parse_transcript(transcript_raw)
            if not transcript:
                raise ValueError(f"Transcript missing in uploaded S3 file: {key}")
            expected_questions = extract_questions_from_transcript(transcript)
            answers_map = get_answers_per_question(transcript, expected_questions)

        must_haves = job_item.get("mustHaveSkills", [])

        # Step 6: Run evaluation (structured Q/A prompt)
        system_prompt = "You are a hiring evaluator. Respond only with JSON."
        qa_for_prompt = [(q, answers_map.get(q, "")) for q in expected_questions]
        user_prompt = build_enhanced_prompt(resume, job_description, must_haves, qa_for_prompt)

        chat_completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=3500
        )

        raw = chat_completion.choices[0].message.content.strip()
        cleaned = clean_json_output(raw)
        validated = validate_and_enhance_result(cleaned, item, context)

        # Alignment + safety net
        validated = enforce_question_coverage(validated, expected_questions)
        validated = apply_min_partial_credit(validated, answers_map)  # optional but recommended

        # --------- Scoring ----------
        try:
            score_weighted = clamp_0_100(calculate_combined_score(validated))
            qe = validated.get("question_evaluations", []) or []
            total = len(qe) if qe else 1
            answered = sum(1 for q in qe if (q.get("individual_score", 0) or 0) > 0)
            coverage = answered / float(total)

            if qe:
                q_avg = sum(float(q.get("individual_score", 0) or 0) for q in qe) / float(total)
            else:
                q_avg = 0.0

            q_avg_0_100 = q_avg * 10.0
            blended = int(round(0.7 * score_weighted + 0.3 * q_avg_0_100))
            cap_by_questions = int(round(q_avg_0_100 + 20))
            coverage_penalty = 0.5 + 0.5 * coverage
            final_overall = int(round(min(blended, cap_by_questions) * coverage_penalty))
            final_overall = clamp_0_100(final_overall)

            validated["overall_interview_score"] = final_overall
            validated["hiring_recommendation"] = determine_hiring_recommendation(final_overall)

        except Exception as e:
            print(f"Error calculating score: \n{validated}\n{e}")
        # ------------------------------------------------------

        resume_cutoff = 0.0
        similarity_score = 0.0

        # Step 7: Store results in DynamoDB based on file type (unchanged)
        if is_audio:
            try:
                update_expression = "SET atBucket = :bucket, atKey = :key, audioScore = :audio_score, audioReport = :audio_report, reportCompletion[0] = :completion"

                expression_attribute_values = {
                    ":bucket": bucket,
                    ":key": key,
                    ":audio_score": Decimal(validated["overall_interview_score"]),
                    ":audio_report": json.loads(json.dumps(validated), parse_float=Decimal),
                    ":completion": True
                }

                table.update_item(
                    Key={"id": response_id},
                    UpdateExpression=update_expression,
                    ExpressionAttributeValues=expression_attribute_values,
                    ReturnValues="ALL_NEW"
                )

                api_gateway_url = "https://ovuftgqb27.execute-api.eu-west-2.amazonaws.com/dev/api/v1/system/instant-audio-shortlist"
                resume_cutoff = job_item.get("resumeCutoff", 90.0)
                similarity_score = validated['overall_interview_score']

            except Exception as e:
                print(f"Error updating audio record: {str(e)}")
                raise

        else:
            try:
                update_expression = "SET vtBucket = :bucket, vtKey = :key, videoScore = :video_score, videoAudioReport = :video_report, reportCompletion[1] = :completion"

                response_update = table.update_item(
                    Key={"id": response_id},
                    UpdateExpression=update_expression,
                    ExpressionAttributeValues={
                        ":bucket": bucket,
                        ":key": key,
                        ":video_score": Decimal(validated["overall_interview_score"]),
                        ":video_report": json.loads(json.dumps(validated), parse_float=Decimal),
                        ":completion": True
                    },
                    ReturnValues="ALL_NEW"
                )

                updated_item = response_update.get("Attributes", {})
                body_score = updated_item.get('bodyScore', 0.0)
                similarity_score = (validated['overall_interview_score'] + float(body_score)) / 2

                api_gateway_url = "https://ovuftgqb27.execute-api.eu-west-2.amazonaws.com/dev/api/v1/system/instant-video-shortlist"
                resume_cutoff = job_item.get("resumeCutoff", 90.0)

            except Exception as e:
                print(f"Error updating video record: {str(e)}")
                raise

        # Threshold + notify (unchanged)
        try:
            print("Checking threshold and making API call...")
            print(f"Resume cutoff threshold: {resume_cutoff}")
            print(f"Candidate similarity score: {similarity_score}")

            if similarity_score >= resume_cutoff:
                print("Candidate meets threshold requirements. Making API Gateway call...")

                payload = {"responseId": response_id}
                headers = {"Content-Type": "application/json"}

                response_post = requests.post(
                    api_gateway_url,
                    json=payload,
                    headers=headers,
                    timeout=30
                )

                if response_post.status_code == 200:
                    print("API Gateway call successful")
                    print(f"API Response: {response_post.json()}")
                else:
                    print(f"API Gateway call failed: {response_post.status_code} - {response_post.text}")
            else:
                print(f"Candidate does not meet threshold requirements. Score {similarity_score} < {resume_cutoff}")

        except Exception as e:
            print(f"[ERROR in check_threshold_and_notify] {e}")

        # Clean up EventBridge rule (best effort)
        delete_event_scheduler(response_id)

        # Return body includes per-question analysis
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": f"✅ {'Audio' if is_audio else 'Video'} evaluation completed successfully",
                "response_id": response_id,
                "file_type": "audio" if is_audio else "video",
                "overall_score": validated["overall_interview_score"],
                "recommendation": validated["hiring_recommendation"],
                "questions_evaluated": len(validated.get("question_evaluations", [])),
                "question_evaluations": validated.get("question_evaluations", [])
            })
        }

    except Exception as e:
        print(f"Error processing record {record}: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Record processing failed: {str(e)}"})
        }

# =============================
# Batch handler (unchanged)
# =============================
def lambda_handler(event, context):
    """Main handler that processes all records in batch"""
    results = []
    successful_processes = 0
    failed_processes = 0

    try:
        for record in event['Records']:
            result = process_single_record(record, context)
            results.append(result)

            if result["statusCode"] == 200:
                successful_processes += 1
            else:
                failed_processes += 1

        return {
            "statusCode": 200 if failed_processes == 0 else 207,
            "body": json.dumps({
                "message": "Batch processing completed",
                "total_records": len(event['Records']),
                "successful": successful_processes,
                "failed": failed_processes,
                "results": results
            })
        }

    except Exception as e:
        print(f"Error during batch processing: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Batch processing failed: {str(e)}"})
        }