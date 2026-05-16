# """
# web_ui.py  —  Flask inference server for the RL Email Triage Agent.
# """

# import os
# import sys
# from datetime import datetime
# from pathlib import Path

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# import logging

# import torch
# from flask import Flask, jsonify, render_template, request

# # Response generation / personalization (exported instances)
# from app.api.response_routes import (
#     feedback_tracker,
#     personalization_memory,
#     response_generator,
#     safety_filter,
# )
# from app.workflow.actions import apply_approved_action
# from app.workflow.analytics import build_dashboard_metrics
# from app.workflow.feedback import FeedbackStore
# from app.workflow.gemini_engine import GeminiContextEngine

# # Workflow utilities
# from app.workflow.preprocessing import parse_gmail_message
# from app.workflow.priority_engine import score_email, should_ignore_sender
# from app.workflow.recommender import build_recommendation
# from app.workflow.reply_service import create_reply_draft
# from app.workflow.thread_context import fetch_thread_context
# from config import Config
# from data.email_data import Email
# from environment.reward import RewardCalculator
# from gmail_auth import gmail_authenticate
# from main import get_trained_agent
# from mlflow_config import MLflowConfig, init_mlflow

# # Local/session services
# from simulation.sources.nlp_extractor import NLPEmailExtractor
# from simulation.sources.sender_memory import SenderMemory

# # Application and logging
# app = Flask(__name__)
# logger = logging.getLogger("rl-email-agent")
# _arrival_times: dict = {}

# # Instantiate session-scoped services used by the UI layer
# gemini_engine = GeminiContextEngine()
# feedback_store = FeedbackStore()
# extractor = NLPEmailExtractor()
# sender_memory = SenderMemory()

"""
web_ui.py  —  Flask inference server for the RL Email Triage Agent.
"""

# =========================
# STANDARD LIBRARY IMPORTS
# =========================
import logging
import os
import sys
from datetime import datetime

# =========================
# THIRD-PARTY IMPORTS
# =========================
import torch
from flask import Flask, jsonify, render_template, request

# Response generation / personalization
from app.api.response_routes import (
    feedback_tracker,
    personalization_memory,
    response_generator,
    safety_filter,
)
from app.workflow.gmail_actions import apply_approved_action
from app.workflow.analytics import build_dashboard_metrics
from app.workflow.feedback import FeedbackStore
from app.workflow.gemini_engine import GeminiContextEngine

# Workflow utilities
from app.workflow.preprocessing import parse_gmail_message
from app.workflow.priority_engine import score_email, should_ignore_sender
from app.workflow.recommender import build_recommendation
# from app.workflow.reply_service import create_reply_draft
from app.workflow.thread_context import fetch_thread_context
from config import Config
from data.email_data import Email
from environment.reward import RewardCalculator
from gmail_auth import gmail_authenticate
from main import get_trained_agent
from mlflow_config import MLflowConfig, init_mlflow

# Local/session services
from simulation.sources.nlp_extractor import NLPEmailExtractor
from simulation.sources.sender_memory import SenderMemory

# =========================
# LOCAL APPLICATION IMPORTS
# =========================

# =========================
# PATH CONFIGURATION
# =========================
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# =========================
# APPLICATION SETUP
# =========================
app = Flask(__name__)
logger = logging.getLogger("rl-email-agent")
_arrival_times: dict = {}

# Instantiate session-scoped services used by the UI layer
gemini_engine = GeminiContextEngine()
feedback_store = FeedbackStore()
extractor = NLPEmailExtractor()
sender_memory = SenderMemory()

# ── Learning validation: session reward tracking ──────────────────────────────
ROLLING_WINDOW = 50  # last N decisions for rolling average
_session_stats = {
    "decisions": 0,
    "total_reward": 0.0,
    "history": [],  # rolling window of recent rewards (capped at ROLLING_WINDOW)
    "action_counts": {0: 0, 1: 0, 2: 0, 3: 0},  # distribution of actions taken
}

AGENT_MODE = os.environ.get("AGENT_MODE", "dqn")
_agent = None
_gmail_service = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = get_trained_agent(mode=AGENT_MODE)
        # Start epsilon at 0.3 so the online learning buffer gets diverse samples.
        # Decays every 10 decisions via decay_epsilon() called in _decide().
        _agent.epsilon = 0.3
        print("[INIT] Online learning enabled. epsilon = 0.3 (will decay over interactions)")
    return _agent


def get_gmail_service():
    global _gmail_service
    if _gmail_service is None:
        _gmail_service = gmail_authenticate()
    return _gmail_service


def _priority_level(score: int) -> int:
    if score >= 75:
        return 3
    if score >= 35:
        return 2
    return 1


def _sender_importance(sender: str, category: str) -> int:
    sender = sender.lower()
    if category in {"PROMOTION", "SPAM"}:
        return 1
    if any(token in sender for token in [".edu", "professor", "recruiter", "hr@", "careers"]):
        return 3
    return 2


def _waiting_time_minutes(timestamp: str) -> int:
    from datetime import timezone

    if not timestamp:
        return 0
    try:
        dt = datetime.fromisoformat(timestamp)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0, int((datetime.now(timezone.utc) - dt).total_seconds() / 60))
    except ValueError:
        return 0


def _hour_of_day(timestamp: str) -> int:
    if not timestamp:
        return datetime.now().hour
    try:
        return datetime.fromisoformat(timestamp).hour
    except ValueError:
        return datetime.now().hour


def _email_state_from_record(email_record, thread_context=None) -> Email:
    priority = _priority_level(email_record.priority)
    if thread_context and thread_context.follow_up_count >= 2:
        priority = min(3, priority + 1)
    return Email(
        subject=email_record.subject,
        sender=email_record.sender,
        priority=priority,
        sender_importance=_sender_importance(email_record.sender, email_record.category),
        waiting_time=_waiting_time_minutes(email_record.timestamp),
        workload=2,
        time_of_day=_hour_of_day(email_record.timestamp),
    )


def _predict_with_q_values(email: Email):
    state = email.to_state_vector()
    agent = get_agent()
    saved_eps = agent.epsilon
    agent.epsilon = 0.0
    action = agent.select_action(state)
    agent.epsilon = saved_eps

    with torch.no_grad():
        t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent._device)
        q_tensor = agent._policy_net(t)[0]
        probabilities = torch.softmax(q_tensor, dim=0)
        q_values = [round(float(x), 3) for x in q_tensor.tolist()]

    labels = ["reply_now", "delay_reply", "mark_important", "archive"]
    return {
        "action_id": action,
        "action": labels[action],
        "confidence": round(float(probabilities[action].item()), 4),
        "q_values": q_values,
        "state_vector": [round(float(x), 2) for x in state.tolist()],
        "raw_p": email.priority,
        "raw_si": email.sender_importance,
        "raw_w": email.waiting_time,
        "raw_wl": email.workload,
        "raw_t": email.time_of_day,
    }


def _serialize_workflow_record(email, thread, intelligence, recommendation, prediction):
    return {
        "id": email.id,
        "thread_id": email.thread_id,
        "sender": email.sender,
        "subject": email.subject,
        "body": email.body[:1200],
        "timestamp": email.timestamp,
        "category": email.category,
        "priority": email.priority,
        "reasons": email.reasons,
        "action": prediction["action"],
        "confidence": prediction["confidence"],
        "q_values": prediction["q_values"],
        "state_vector": prediction["state_vector"],
        "gemini": intelligence.to_dict(),
        "thread": thread.to_dict() if thread else None,
        "recommendation": recommendation.to_dict(),
    }


# ── Routes ────────────────────────────────────────────────────────────────────


@app.route("/agent_status")
def agent_status():
    agent = get_agent()
    return jsonify(
        {
            "mode": AGENT_MODE,
            "epsilon": round(agent.epsilon, 4),
        }
    )


@app.route("/debug")
def debug():
    """Full debug snapshot — epsilon, sender memory, device info, session stats."""
    agent = get_agent()
    hist = _session_stats["history"]
    rolling_avg = (sum(hist) / len(hist)) if hist else 0.0
    return jsonify(
        {
            "epsilon": round(agent.epsilon, 4),
            "sender_memory": sender_memory.snapshot(),
            "model_device": str(agent._device),
            "agent_mode": AGENT_MODE,
            "session": {
                "decisions": _session_stats["decisions"],
                "total_reward": round(_session_stats["total_reward"], 2),
                "rolling_avg_reward": round(rolling_avg, 3),
                "rolling_window": len(hist),
                "action_distribution": {
                    "reply_now": _session_stats["action_counts"][0],
                    "delay_reply": _session_stats["action_counts"][1],
                    "mark_important": _session_stats["action_counts"][2],
                    "archive": _session_stats["action_counts"][3],
                },
            },
        }
    )


@app.route("/gmail_inbox")
def gmail_inbox():
    """Live Gmail inbox intelligence: Gmail -> preprocessing -> DQN -> Gemini/context."""
    max_results = int(request.args.get("max_results", 10))
    query = request.args.get("q", "in:inbox newer_than:14d")
    service = get_gmail_service()
    results = (
        service.users().messages().list(userId="me", q=query, maxResults=max_results).execute()
    )
    messages = results.get("messages", [])
    records = []

    for item in messages:
        raw_message = (
            service.users().messages().get(userId="me", id=item["id"], format="full").execute()
        )
        email_record = parse_gmail_message(raw_message)

        if should_ignore_sender(email_record.sender):
            thread_context = None
            email_record = score_email(email_record)
        else:
            thread_context = (
                fetch_thread_context(service, email_record.thread_id)
                if email_record.thread_id
                else None
            )
            email_record = score_email(
                email_record,
                follow_up_count=thread_context.follow_up_count if thread_context else 0,
            )

        email_state = _email_state_from_record(email_record, thread_context)
        prediction = _predict_with_q_values(email_state)
        intelligence = gemini_engine.analyze(email_record, thread_context)
        recommendation = build_recommendation(email_record, intelligence, thread_context)
        recommendation.action = prediction["action"]
        recommendation.confidence = prediction["confidence"]
        recommendation.reasons.insert(0, f"DQN policy selected {prediction['action']}")

        records.append(
            _serialize_workflow_record(
                email_record,
                thread_context,
                intelligence,
                recommendation,
                prediction,
            )
        )

    analytics = build_dashboard_metrics(records, feedback_store.events())
    return jsonify({"records": records, "analytics": analytics})


@app.route("/workflow/approve", methods=["POST"])
def workflow_approve():
    """Human approval/correction endpoint. Applies Gmail action and stores reward."""
    try:
        data = request.get_json()
        message_id = data["message_id"]
        thread_id = data.get("thread_id", "")
        predicted_action = data["predicted_action"]
        user_action = data["user_action"]
        draft_text = data.get("draft_text", "")
        sender = data.get("sender", "")
        subject = data.get("subject", "")

        event = feedback_store.record(
            message_id=message_id,
            thread_id=thread_id,
            recommended_action=predicted_action,
            user_action=user_action,
            notes="flask-dashboard",
        )

        service = get_gmail_service()
        try:
            apply_approved_action(service, message_id, user_action)
        except Exception as e:
            logger.exception(
                "Failed to apply gmail action %s for %s: %s", user_action, message_id, e
            )
            return (
                jsonify(
                    {"success": False, "error": "Failed to apply Gmail action", "details": str(e)}
                ),
                500,
            )

        if user_action == "reply_now" and draft_text:
            try:
                create_reply_draft(service, sender, subject, draft_text, thread_id)
            except Exception as e:
                logger.exception("Failed to create reply draft for %s: %s", message_id, e)
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "Failed to create reply draft",
                            "details": str(e),
                        }
                    ),
                    500,
                )

        # Log MLflow metrics but do not allow logging errors to break the response
        try:
            import mlflow

            init_mlflow(MLflowConfig.EXPERIMENT_INFERENCE)
            with mlflow.start_run(run_name="flask-human-feedback", nested=True):
                mlflow.log_metric("feedback_reward", event.reward)
                mlflow.log_param("recommended_action", predicted_action)
                mlflow.log_param("user_action", user_action)
        except Exception:
            logger.warning("MLflow logging failed, continuing")

        return jsonify({"success": True, "reward": event.reward, "event": event.to_dict()})
    except KeyError as e:
        logger.exception("Bad request to /workflow/approve: missing %s", e)
        return jsonify({"success": False, "error": "Bad request", "details": str(e)}), 400
    except Exception as e:
        logger.exception("Unhandled error in /workflow/approve: %s", e)
        return jsonify({"success": False, "error": "Internal server error", "details": str(e)}), 500


@app.route("/workflow/analytics")
def workflow_analytics():
    """Feedback analytics for dashboard charts."""
    events = feedback_store.events()
    rewards = [event.get("reward", 0) for event in events]
    corrections = sum(
        1 for event in events if event.get("recommended_action") != event.get("user_action")
    )
    return jsonify(
        {
            "feedback_events": events,
            "reward_trend": rewards[-50:],
            "corrections": corrections,
        }
    )


def _decide(email: Email):
    """
    Greedy inference pass — UI always sees the model's current best action.
    After deciding, the agent learns from the reward signal (online adaptation).
    Epsilon decays every 10 decisions; exploration only affects the learning
    buffer, NOT the action shown in the UI (which is always greedy).
    """
    state = email.to_state_vector()
    agent = get_agent()
    saved_eps = agent.epsilon
    agent.epsilon = 0.0  # greedy for UI — user always sees best action
    action = agent.select_action(state)
    agent.epsilon = saved_eps

    reward = RewardCalculator().calculate(email, action)
    label = ["reply_now", "delay_reply", "mark_important", "archive"][action]

    # ── Structured debug log + Q-value explosion guard ────────────────────────
    with torch.no_grad():
        t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent._device)
        q_vals = [round(x, 3) for x in agent._policy_net(t)[0].tolist()]

    q_max = max(q_vals)
    q_warn = "  *** Q-VALUE EXPLOSION — check training stability ***" if q_max > 500 else ""

    print(
        f"\n[DECIDE] subject={email.subject!r:.45}  sender={email.sender!r:.30}\n"
        f"         state    = {[round(x, 3) for x in state.tolist()]}\n"
        f"         q_values = {q_vals}   (0=reply 1=delay 2=mark 3=archive){q_warn}\n"
        f"         action   = {action} ({label})   reward = {reward}   "
        f"epsilon = {round(saved_eps, 4)}   wait = {email.waiting_time}min"
    )

    # ── Online learning: adapt to new reward signal without full retrain ───────
    # next_state = state (same-state approximation — we don't have the next email yet).
    # Once replay buffer hits batch_size (64), each call does a gradient step.
    agent.learn(state, action, reward, state, False)

    # ── Session stats + epsilon decay ─────────────────────────────────────────
    _session_stats["decisions"] += 1
    _session_stats["total_reward"] += reward
    _session_stats["action_counts"][action] += 1
    hist = _session_stats["history"]
    hist.append(reward)
    if len(hist) > ROLLING_WINDOW:
        hist.pop(0)

    rolling_avg = sum(hist) / len(hist)

    # Decay epsilon every 10 decisions so exploration gradually reduces
    if _session_stats["decisions"] % 10 == 0:
        agent.decay_epsilon()
        print(f"         [DECAY]  epsilon now = {agent.epsilon:.4f}")

    print(
        f"         session  = {_session_stats['decisions']} decisions  "
        f"rolling_avg = {rolling_avg:.2f}  "
        f"total = {_session_stats['total_reward']:.1f}"
    )

    return label, reward, state, q_vals


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/infer", methods=["POST"])
def infer():
    """
    Live-preview inference (called on every keystroke via debounce).
    Also records the email's 'arrival time' for Phase 2 waiting_time calculation.
    """
    try:
        data = request.get_json()
        subject = data.get("subject", "").strip()
        sender = data.get("sender", "").strip()

        # Record arrival time the first time this subject+sender is typed.
        key = hash(f"{subject}|{sender}")
        if key not in _arrival_times:
            _arrival_times[key] = datetime.now()

        reasoning = extractor.explain(subject, sender)
        return jsonify(reasoning)
    except Exception as e:
        logger.exception("Error in /infer: %s", e)
        return jsonify({"success": False, "error": "Failed to infer", "details": str(e)}), 500


@app.route("/decide_nlp", methods=["POST"])
def decide_nlp():
    """Auto-mode decision: NLP extracts features, agent decides."""
    try:
        data = request.get_json()
        subject, sender = data["subject"].strip(), data["sender"].strip()

        key = hash(f"{subject}|{sender}")
        arrival = _arrival_times.pop(key, None)
        waiting_mins = int((datetime.now() - arrival).total_seconds() / 60) if arrival else 0

        rule_si = extractor._classify_sender(sender)
        adapted_si = sender_memory.get_importance(sender, rule_si)

        email = extractor.extract(
            subject, sender, waiting_time=waiting_mins, sender_importance=adapted_si
        )
        reasoning = extractor.explain(subject, sender)

        label, reward, state, q_vals = _decide(email)
        action_int = ["reply_now", "delay_reply", "mark_important", "archive"].index(label)

        sender_memory.update(sender, action_int, reward)

        return jsonify(
            {
                "action": label,
                "reward": reward,
                "state_vector": [round(x, 2) for x in state.tolist()],
                "q_values": q_vals,
                "raw_p": email.priority,
                "raw_si": email.sender_importance,
                "raw_w": email.waiting_time,
                "raw_wl": email.workload,
                "raw_t": email.time_of_day,
                "subject": subject,
                "sender": sender,
                "reasoning": reasoning,
                "epsilon": round(get_agent().epsilon, 4),
                "agent_mode": AGENT_MODE,
                "sender_memory": sender_memory.snapshot(),
            }
        )
    except Exception as e:
        logger.exception("Error in /decide_nlp: %s", e)
        return jsonify({"success": False, "error": "Decision failed", "details": str(e)}), 500


@app.route("/decide", methods=["POST"])
def decide():
    """Manual-mode decision: caller provides raw feature values via sliders."""
    try:
        data = request.get_json()
        email = Email(
            subject=data["subject"],
            sender=data["sender"],
            priority=int(data["priority"]),
            sender_importance=int(data["sender_importance"]),
            waiting_time=int(data["waiting_time"]),
            workload=int(data["workload"]),
            time_of_day=int(data["time_of_day"]),
        )
        label, reward, state, q_vals = _decide(email)
        return jsonify(
            {
                "action": label,
                "reward": reward,
                "state_vector": [round(x, 2) for x in state.tolist()],
                "q_values": q_vals,
                "raw_p": email.priority,
                "raw_si": email.sender_importance,
                "raw_w": email.waiting_time,
                "raw_wl": email.workload,
                "raw_t": email.time_of_day,
                "subject": data["subject"],
                "sender": data["sender"],
                "epsilon": round(get_agent().epsilon, 4),
                "agent_mode": AGENT_MODE,
            }
        )
    except Exception as e:
        logger.exception("Error in /decide: %s", e)
        return jsonify({"success": False, "error": "Decision failed", "details": str(e)}), 500


# ── AI Response Generation Routes ────────────────────────────────────────────


@app.route("/api/responses/generate", methods=["POST"])
def generate_ai_response():
    """
    Generate AI response draft for an email.

    JSON request:
    {
        "message_id": "...",
        "thread_id": "...",
        "sender": "...",
        "subject": "...",
        "body": "...",
        "tone": "professional"
    }
    """
    try:
        data = request.get_json()

        from app.workflow.schemas import EmailRecord

        email = EmailRecord(
            id=data.get("message_id", ""),
            thread_id=data.get("thread_id", ""),
            sender=data.get("sender", ""),
            subject=data.get("subject", ""),
            body=data.get("body", ""),
        )

        tone = data.get("tone", "professional")
        if tone not in response_generator.SUPPORTED_TONES:
            tone = "professional"

        # Get personalization hints
        personalization = personalization_memory.get_personalization_hints()

        # Generate response
        generated = response_generator.generate(
            email=email,
            tone=tone,
            personalization=personalization,
            max_length=data.get("max_length", 500),
        )

        # Safety check
        safety_result = safety_filter.validate(
            generated.generated_text,
            sender=email.sender,
        )

        requires_approval = (
            not safety_result.is_safe
            or generated.confidence < 0.7
            or data.get("require_approval", True)
        )

        return jsonify(
            {
                "success": True,
                "response_id": generated.message_id,
                "generated_text": generated.generated_text,
                "tone_used": generated.tone_used,
                "confidence": generated.confidence,
                "warnings": generated.warnings + safety_result.warnings,
                "requires_approval": requires_approval,
                "safety": safety_result.to_dict(),
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/responses/regenerate", methods=["POST"])
def regenerate_ai_response():
    """
    Regenerate response with different tone.

    JSON request:
    {
        "response_id": "...",
        "new_tone": "friendly"
    }
    """
    try:
        data = request.get_json()
        new_tone = data.get("new_tone", "professional")

        if new_tone not in response_generator.SUPPORTED_TONES:
            return jsonify({"success": False, "error": f"Unsupported tone: {new_tone}"}), 400

        personalization = personalization_memory.get_personalization_hints()

        # Create minimal email for regeneration
        from app.workflow.schemas import EmailRecord

        email = EmailRecord(
            id=data.get("response_id", ""),
            thread_id="",
            sender="",
            subject="",
            body="",
        )

        generated = response_generator.generate(
            email=email,
            tone=new_tone,
            personalization=personalization,
        )

        safety_result = safety_filter.validate(generated.generated_text)

        return jsonify(
            {
                "success": True,
                "generated_text": generated.generated_text,
                "tone_used": generated.tone_used,
                "confidence": generated.confidence,
                "warnings": safety_result.warnings,
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/responses/feedback", methods=["POST"])
def record_response_feedback():
    """
    Record user feedback on AI-generated response.

    JSON request:
    {
        "response_id": "...",
        "feedback_type": "approved|edited|rejected",
        "edited_text": "...",
        "approval_time_seconds": 5.2,
        "feedback_notes": "..."
    }
    """
    try:
        data = request.get_json()

        feedback_type = data.get("feedback_type", "approved")
        if feedback_type not in ["approved", "edited", "rejected"]:
            return jsonify({"success": False, "error": "Invalid feedback_type"}), 400

        # Record in feedback tracker
        feedback = feedback_tracker.record_feedback(
            response_id=data.get("response_id", ""),
            sender=data.get("sender", "unknown"),
            original_action="reply_now",
            generated_tone="professional",
            feedback_type=feedback_type,
            approval_time_seconds=data.get("approval_time_seconds", 0.0),
            feedback_notes=data.get("feedback_notes", ""),
        )

        # Record in personalization memory
        personalization_memory.record_action(
            response_id=data.get("response_id", ""),
            sender=data.get("sender", "unknown"),
            action=feedback_type,
            tone="professional",
            generated_text="",
            user_text=data.get("edited_text", ""),
            feedback=data.get("feedback_notes", ""),
        )

        return jsonify(
            {
                "success": True,
                "reward": feedback.reward_signal,
                "feedback_type": feedback_type,
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/responses/tones", methods=["GET"])
def get_supported_tones():
    """Get list of supported response tones."""
    return jsonify(
        {
            "tones": response_generator.SUPPORTED_TONES,
            "descriptions": {
                tone: response_generator._get_tone_description(tone)
                for tone in response_generator.SUPPORTED_TONES
            },
        }
    )


@app.route("/api/responses/personalization", methods=["GET"])
def get_personalization_profile():
    """Get user's personalization profile."""
    return jsonify(personalization_memory.export_profile())


# Draft management endpoints
@app.route("/api/drafts/create", methods=["POST"])
def api_create_draft():
    """Create a Gmail draft from AI-generated content."""
    try:
        data = request.get_json()
        # Lazy import to avoid circulars during module load
        from app.workflow.gmail_actions import create_ai_draft, get_draft_preview

        service = get_gmail_service()
        draft = create_ai_draft(
            service,
            to_address=data.get("to"),
            subject=data.get("subject", ""),
            body_text=data.get("body", ""),
            thread_id=data.get("thread_id", ""),
            labels=data.get("labels", ["AI/Draft"]),
        )

        preview = get_draft_preview(service, draft.get("id"))
        return jsonify({"success": True, "draft": draft, "preview": preview})
    except Exception as e:
        try:
            logger.exception("Failed to create AI draft: %s", e)
        except Exception:
            pass
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/drafts/update", methods=["POST"])
def api_update_draft():
    try:
        data = request.get_json()
        from app.workflow.gmail_actions import get_draft_preview, update_draft

        service = get_gmail_service()
        updated = update_draft(
            service,
            draft_id=data["draft_id"],
            to_address=data.get("to"),
            subject=data.get("subject", ""),
            body_text=data.get("body", ""),
            thread_id=data.get("thread_id", ""),
        )
        preview = get_draft_preview(service, updated.get("id"))
        return jsonify({"success": True, "draft": updated, "preview": preview})
    except KeyError as e:
        return jsonify({"success": False, "error": "Missing field", "details": str(e)}), 400
    except Exception as e:
        try:
            logger.exception("Failed to update draft: %s", e)
        except Exception:
            pass
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/drafts/send", methods=["POST"])
def api_send_draft():
    """Send a saved draft (explicit user action)."""
    try:
        data = request.get_json()
        draft_id = data.get("draft_id")
        if not draft_id:
            return jsonify({"success": False, "error": "draft_id required"}), 400

        from app.workflow.gmail_actions import get_draft, send_draft

        service = get_gmail_service()
        sent = send_draft(service, draft_id)

        # After explicit user send, record feedback (positive reward) and personalize
        try:
            # Try to fetch draft/message headers for sender context
            draft = get_draft(service, draft_id)
            headers = draft.get("message", {}).get("payload", {}).get("headers", [])
            header_map = {h.get("name"): h.get("value") for h in headers}
            sender = header_map.get("To", "unknown")
        except Exception:
            sender = "unknown"

        try:
            # Record feedback signal: user sent the draft → approved
            feedback = feedback_tracker.record_feedback(
                response_id=draft_id,
                sender=sender,
                original_action="reply_now",
                generated_tone="unknown",
                feedback_type="approved",
                approval_time_seconds=data.get("approval_time_seconds", 0.0),
                feedback_notes=data.get("feedback_notes", "sent_via_ui"),
            )

            # Update personalization memory
            try:
                personalization_memory.record_action(
                    response_id=draft_id,
                    sender=sender,
                    action="approved",
                    tone="unknown",
                    generated_text="",
                    user_text="",
                    feedback="sent",
                )
            except Exception:
                pass

            # Log to MLflow (best-effort)
            try:
                import mlflow

                init_mlflow(MLflowConfig.EXPERIMENT_INFERENCE)
                with mlflow.start_run(run_name="ui-draft-send", nested=True):
                    mlflow.log_param("draft_id", draft_id)
                    mlflow.log_param("sender", sender)
                    mlflow.log_metric("draft_send_reward", feedback.reward_signal)
            except Exception:
                try:
                    logger.warning("MLflow logging failed for draft send %s", draft_id)
                except Exception:
                    pass
        except Exception:
            # Non-fatal: continue even if feedback recording fails
            try:
                logger.exception("Failed to record feedback for draft %s", draft_id)
            except Exception:
                pass

        return jsonify({"success": True, "sent": sent})
    except Exception as e:
        try:
            logger.exception("Failed to send draft %s: %s", data.get("draft_id"), e)
        except Exception:
            pass
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/responses/stats", methods=["GET"])
def get_response_stats():
    """Get response generation statistics."""
    return jsonify(
        {
            "feedback_stats": feedback_tracker.get_session_stats(),
            "tone_performance": feedback_tracker.get_tone_performance(),
            "filter_stats": safety_filter.get_stats(),
        }
    )


if __name__ == "__main__":
    print("Loading agent...")
    get_agent()
    print(f"Starting server on http://{Config.FLASK_HOST}:{Config.FLASK_PORT}")
    app.run(host=Config.FLASK_HOST, port=Config.FLASK_PORT, debug=Config.FLASK_DEBUG)
