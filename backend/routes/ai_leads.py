from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from auth import AdminUser, get_current_user
from db import execute_query
from ai_service import ai_client
from access_control import build_ai_pause_message
from audit_logger import log_event
from cost_control import build_spend_limit_message
from mcp_tools import get_dealer_tool, get_lead_score_tool, get_lead_tool

router = APIRouter(prefix="/api/ai", tags=["AI Features"])


class LeadScoreResponse(BaseModel):
    priority: str
    score: int
    reasoning: str
    suggested_actions: List[str]


class MatchExplanationResponse(BaseModel):
    explanation: str
    confidence: str
    considerations: List[str]


class EmailDraftRequest(BaseModel):
    lead_id: int
    purpose: str  # intro, follow-up, reminder, escalation
    dealer_id: Optional[int] = None


class EmailDraftResponse(BaseModel):
    subject: str
    body: str
    tone: str


class EnrichmentResponse(BaseModel):
    inferred_business_category: Optional[str] = None
    estimated_project_size: Optional[str] = None
    suggested_products: List[str] = []
    confidence: float = 0.0


class ConversionPredictionResponse(BaseModel):
    likelihood: float
    label: str
    explanation: str
    risk_factors: List[str]


@router.post("/lead-score")
async def score_lead(lead_id: int, current_user: AdminUser = Depends(get_current_user)):
    """AI-powered lead scoring and priority classification."""
    try:
        lead = get_lead_tool(lead_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Lead not found")

    prompt = f"""Analyze this window film installation lead and classify its priority.

Lead Details:
- Name: {lead.get('name', 'N/A')}
- Company: {lead.get('company_name', 'N/A')}
- City: {lead.get('city', 'N/A')}, Province: {lead.get('province', 'N/A')}
- Job Type: {lead.get('job_type', 'N/A')}
- Product Type: {lead.get('product_type', 'N/A')}
- Square Footage: {lead.get('square_footage', 'N/A')}
- Project Type: {lead.get('project_type', 'N/A')}
- Business Category: {lead.get('business_category', 'N/A')}
- Lead Source: {lead.get('lead_source', 'N/A')}
- Comments: {lead.get('comments', 'N/A')}

Respond in JSON format:
{{"priority": "Hot|Warm|Cold", "score": 0-100, "reasoning": "brief explanation", "suggested_actions": ["action1", "action2"]}}

Consider: commercial leads and larger square footage are typically higher priority. Leads with specific product needs and clear project details are warmer."""

    result = ai_client.call_json(
        system="You are a lead scoring AI for a window film installation company. Classify leads as Hot (high priority, likely to convert), Warm (moderate interest), or Cold (low priority). Return valid JSON only.",
        user=prompt,
        actor=current_user.username,
        entity_type="lead",
        entity_id=str(lead_id),
        payload={"operation": "lead_score"},
        task_type="reasoning",
    )

    if result:
        # Persist to DB
        execute_query(
            """UPDATE leads SET ai_priority = %s, ai_score = %s, ai_reasoning = %s, ai_scored_at = CURRENT_TIMESTAMP
            WHERE id = %s""",
            (result.get('priority', 'Warm'), result.get('score', 50), result.get('reasoning', ''), lead_id),
            fetch=False
        )
        call_meta = ai_client.last_call_meta or {}
        log_event(
            event_type="AI_LEAD_SCORING_BATCH",
            entity_type="lead",
            entity_id=str(lead_id),
            actor=current_user.username,
            model_used=call_meta.get("model_used"),
            tokens_used=call_meta.get("tokens_used"),
            cost_cad=call_meta.get("cost_cad"),
            latency_ms=call_meta.get("latency_ms"),
            payload={
                "lead_ids": [lead_id],
                "leads_scored": 1,
                "priority": result.get("priority"),
                "score": result.get("score"),
            },
        )
        return result

    if ai_client.last_error_meta.get("type") == "agent_paused":
        return {"priority": "Warm", "score": 50, "reasoning": build_ai_pause_message(ai_client.last_error_meta), "suggested_actions": ["Resume AI operations in Tools"]} 

    spend_message = build_spend_limit_message(ai_client.last_error_meta)
    if ai_client.last_error_meta.get("type") == "spend_limit":
        return {"priority": "Warm", "score": 50, "reasoning": spend_message, "suggested_actions": ["Review monthly and daily AI spend limits"]}

    return {"priority": "Warm", "score": 50, "reasoning": "Unable to analyze - using default", "suggested_actions": ["Review manually"]}


@router.get("/lead-score/{lead_id}")
async def get_lead_score(lead_id: int, current_user: AdminUser = Depends(get_current_user)):
    """Get cached AI score for a lead."""
    try:
        lead_score = get_lead_score_tool(lead_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Lead not found")
    return lead_score


@router.post("/explain-match")
async def explain_match(lead_id: int, current_user: AdminUser = Depends(get_current_user)):
    """AI explains why a dealer recommendation or assignment was made for this lead."""
    try:
        lead = get_lead_tool(lead_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Lead not found")

    assigned_to = lead.get('dealer_name_assigned') or lead.get('recommended_dealer_name') or 'No assignment'

    prompt = f"""Explain why this dealer was recommended or assigned to this lead for a window film project.

Lead: {lead.get('name')} in {lead.get('city')}, {lead.get('province')}
Project: {lead.get('job_type', 'N/A')} - {lead.get('product_type', 'N/A')}
Square Footage: {lead.get('square_footage', 'N/A')}
Assigned To: {assigned_to} in {lead.get('dealer_city') or lead.get('recommended_dealer_city', 'N/A')}
Distance: {lead.get('distance_to_dealer_km', 'N/A')} km
Allocation Score: {lead.get('allocation_score', 'N/A')}

Respond in JSON: {{"explanation": "...", "confidence": "high|medium|low", "considerations": ["point1", "point2"]}}"""

    result = ai_client.call_json(
        system="You are an AI that explains dealer-lead matching decisions for a window film company. Be specific and data-driven.",
        user=prompt,
        task_type="reasoning",
    )

    if result:
        execute_query(
            "UPDATE leads SET ai_match_explanation = %s WHERE id = %s",
            (result.get('explanation', ''), lead_id), fetch=False
        )
        return result

    if ai_client.last_error_meta.get("type") == "agent_paused":
        return {
            "explanation": build_ai_pause_message(ai_client.last_error_meta),
            "confidence": "low",
            "considerations": ["AI operations paused"],
        }

    if ai_client.last_error_meta.get("type") == "spend_limit":
        return {
            "explanation": build_spend_limit_message(ai_client.last_error_meta),
            "confidence": "low",
            "considerations": ["AI spend limit reached"],
        }

    return {"explanation": "Match based on proximity and availability.", "confidence": "low", "considerations": []}


@router.post("/draft-email")
async def draft_email(req: EmailDraftRequest, current_user: AdminUser = Depends(get_current_user)):
    """AI generates contextual email draft."""
    try:
        lead = get_lead_tool(req.lead_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Lead not found")

    dealer_info = ""
    if req.dealer_id:
        try:
            dealer = get_dealer_tool(req.dealer_id)
            dealer_info = f"Dealer: {dealer['name']} ({dealer.get('email', 'N/A')})"
        except ValueError:
            dealer_info = ""

    # Get interaction history
    logs = execute_query(
        "SELECT log_type, message, created_at FROM lead_logs WHERE lead_id = %s ORDER BY created_at DESC LIMIT 5",
        (req.lead_id,)
    )
    history = "\n".join([f"- {l['log_type']}: {l['message']} ({l['created_at']})" for l in logs]) if logs else "No previous interactions"

    prompt = f"""Draft a {req.purpose} email for a window film project lead.

Lead: {lead.get('name')} - {lead.get('email', 'N/A')}
Project: {lead.get('job_type', 'N/A')} in {lead.get('city', 'N/A')}, {lead.get('province', 'N/A')}
Product: {lead.get('product_type', 'N/A')}
{dealer_info}

Interaction History:
{history}

Email Purpose: {req.purpose}
Respond in JSON: {{"subject": "...", "body": "...", "tone": "professional"}}"""

    result = ai_client.call_json(
        system="You are an email assistant for Window Film Canada. Draft professional, concise emails. Use proper business formatting.",
        user=prompt,
        task_type="realtime",
    )

    if result:
        return result

    if ai_client.last_error_meta.get("type") == "agent_paused":
        return {
            "subject": f"Re: Window Film Project - {lead.get('city', '')}",
            "body": build_ai_pause_message(ai_client.last_error_meta),
            "tone": "professional",
        }

    if ai_client.last_error_meta.get("type") == "spend_limit":
        return {
            "subject": f"Re: Window Film Project - {lead.get('city', '')}",
            "body": build_spend_limit_message(ai_client.last_error_meta),
            "tone": "professional",
        }

    return {"subject": f"Re: Window Film Project - {lead.get('city', '')}", "body": "Dear ...,\n\n", "tone": "professional"}


@router.post("/enrich-lead")
async def enrich_lead(lead_id: int, current_user: AdminUser = Depends(get_current_user)):
    """AI enriches lead data by inferring missing fields."""
    try:
        lead = get_lead_tool(lead_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Lead not found")

    prompt = f"""Based on the available lead information, infer missing fields for a window film project.

Known Info:
- Name: {lead.get('name', 'N/A')}
- Company: {lead.get('company_name', 'N/A')}
- Job Type: {lead.get('job_type', 'N/A')}
- Product Type: {lead.get('product_type', 'N/A')}
- City: {lead.get('city', 'N/A')}, Province: {lead.get('province', 'N/A')}
- Comments: {lead.get('comments', 'N/A')}

Respond in JSON:
{{"inferred_business_category": "Commercial Office|Residential|Retail|Government|Healthcare|Education|Other or null",
  "estimated_project_size": "small|medium|large|enterprise",
  "suggested_products": ["product1", "product2"],
  "confidence": 0.0-1.0}}"""

    result = ai_client.call_json(
        system="You are a lead enrichment AI for a window film company. Infer realistic values based on available data. Be conservative with confidence scores.",
        user=prompt,
        task_type="reasoning",
    )

    if result:
        return result

    if ai_client.last_error_meta.get("type") == "agent_paused":
        return {
            "inferred_business_category": None,
            "estimated_project_size": "medium",
            "suggested_products": [build_ai_pause_message(ai_client.last_error_meta)],
            "confidence": 0.0,
        }

    if ai_client.last_error_meta.get("type") == "spend_limit":
        return {
            "inferred_business_category": None,
            "estimated_project_size": "medium",
            "suggested_products": [build_spend_limit_message(ai_client.last_error_meta)],
            "confidence": 0.0,
        }

    return {"inferred_business_category": None, "estimated_project_size": "medium", "suggested_products": [], "confidence": 0.0}


@router.post("/predict-conversion")
async def predict_conversion(lead_id: int, current_user: AdminUser = Depends(get_current_user)):
    """AI predicts conversion likelihood with explanation."""
    try:
        lead = get_lead_tool(lead_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Lead not found")

    # Get conversion stats for context
    stats = execute_query("""
        SELECT
            COUNT(CASE WHEN status = 'converted' THEN 1 END)::float / NULLIF(COUNT(*), 0) as overall_rate,
            AVG(CASE WHEN status = 'converted' THEN allocation_score END) as avg_converted_score
        FROM leads
    """)
    overall_rate = stats[0].get('overall_rate', 0) if stats else 0

    prompt = f"""Predict the conversion likelihood for this window film lead.

Lead: {lead.get('name')} in {lead.get('city', 'N/A')}, {lead.get('province', 'N/A')}
Job Type: {lead.get('job_type', 'N/A')}
Product: {lead.get('product_type', 'N/A')}
Square Footage: {lead.get('square_footage', 'N/A')}
ML Allocation Score: {lead.get('allocation_score', 'N/A')}
AI Priority: {lead.get('ai_priority', 'N/A')}
Assigned Dealer: {lead.get('dealer_name_assigned', 'Unassigned')}
Lead Source: {lead.get('lead_source', 'N/A')}
Overall Conversion Rate: {round(overall_rate * 100, 1) if overall_rate else 'N/A'}%

Respond in JSON:
{{"likelihood": 0.0-1.0, "label": "Very Likely|Likely|Possible|Unlikely", "explanation": "...", "risk_factors": ["factor1"]}}"""

    result = ai_client.call_json(
        system="You are a conversion prediction AI for a window film company. Be realistic and data-driven in your predictions.",
        user=prompt,
        task_type="reasoning",
    )

    if result:
        execute_query(
            "UPDATE leads SET ai_conversion_likelihood = %s, ai_conversion_explanation = %s WHERE id = %s",
            (result.get('likelihood', 0.5), result.get('explanation', ''), lead_id), fetch=False
        )
        return result

    if ai_client.last_error_meta.get("type") == "agent_paused":
        return {
            "likelihood": 0.5,
            "label": "Possible",
            "explanation": build_ai_pause_message(ai_client.last_error_meta),
            "risk_factors": ["AI operations paused"],
        }

    if ai_client.last_error_meta.get("type") == "spend_limit":
        return {
            "likelihood": 0.5,
            "label": "Possible",
            "explanation": build_spend_limit_message(ai_client.last_error_meta),
            "risk_factors": ["AI spend limit reached"],
        }

    return {"likelihood": 0.5, "label": "Possible", "explanation": "Insufficient data for prediction", "risk_factors": []}
