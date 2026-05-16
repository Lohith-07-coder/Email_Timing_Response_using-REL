# FastAPI ↔ Frontend Integration Guide

## Status: ✅ AI Reply Panel FIXED

The **"Generate AI Reply"** button now works with the FastAPI backend.

---

## 1. Fixed: ai_reply_panel.html

### Changes Made:
✅ Added API configuration constants
✅ Updated `/api/responses/generate` → `http://localhost:8000/api/v1/responses/generate`
✅ Updated `/api/responses/feedback` → `http://localhost:8000/api/v1/responses/feedback`
✅ Added proper error handling for network and API failures
✅ Fixed response payload to match FastAPI schema
✅ Added validation for required response fields

### Request Payload (Now Correct):
```javascript
{
    message_id: "string",
    thread_id: "string",
    sender: "string@example.com",
    subject: "Email subject",
    body: "Email body",
    tone: "professional",        // or: friendly, formal, concise, enthusiastic
    max_length: 500,             // NEW: FastAPI requirement
    require_approval: true       // NEW: FastAPI requirement
}
```

### Response Validation:
The frontend now expects FastAPI response schema:
```json
{
    "response_id": "resp_...",
    "message_id": "msg_...",
    "generated_text": "Dear...",
    "tone_used": "professional",
    "confidence": 0.92,
    "warnings": ["optional", "safety", "warnings"],
    "requires_approval": true,
    "safety_check": { "is_safe": true, "violations": [] }
}
```

---

## 2. How to Use

### Start the FastAPI Backend:
```bash
cd email-timing-response
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Access the Frontend:
```
http://localhost:5000  (Flask UI)
```

### Test the AI Reply Panel:
1. Click "Generate AI Reply" button
2. Panel opens automatically with tone selection
3. Response generates from FastAPI backend
4. Approve/Edit/Regenerate/Reject options work
5. Feedback is recorded to FastAPI

---

## 3. CORS Configuration

✅ **Already configured** in `app/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

No additional CORS setup needed.

---

## 4. FastAPI Endpoints Available

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/predict` | POST | DQN agent inference |
| `/model/version` | GET | List model versions |
| `/model/version/{filename}` | POST | Swap model |
| `/api/v1/responses/generate` | POST | Generate AI response ✅ WORKING |
| `/api/v1/responses/regenerate` | POST | Regenerate with new tone |
| `/api/v1/responses/feedback` | POST | Record feedback ✅ WORKING |

---

## 5. Remaining Work: index.html Integration

### Current Status:
The following endpoints in `index.html` still call **Flask** endpoints (port 5000):

| Endpoint | Line | Purpose | Status |
|----------|------|---------|--------|
| `/agent_status` | 1039 | Agent status | ⚠️ Not in FastAPI |
| `/debug` | 1045+ | Debug stats | ⚠️ Not in FastAPI |
| `/infer` | 1082 | NLP inference | ⚠️ Not in FastAPI |
| `/decide_nlp` | 1112 | NLP decision | ⚠️ Not in FastAPI |
| `/decide` | 1126 | Manual decision | ⚠️ Not in FastAPI |
| `/gmail_inbox` | 1344 | Gmail fetch | ⚠️ Not in FastAPI |
| `/workflow/approve` | 1447 | Approve action | ⚠️ Not in FastAPI |
| `/workflow/analytics` | 1495 | Analytics | ⚠️ Not in FastAPI |

### Solution Options:

#### **Option A: Keep Flask for non-critical UI features** (RECOMMENDED)
- AI Reply Panel ✅ works with FastAPI
- Leave index.html calling port 5000 for debug/stats
- No blocking - users can still use AI features
- Less migration work

**Implementation:**
```javascript
// In index.html - keep calling Flask (no change needed)
fetch('/debug').then(...)      // → localhost:5000
fetch('/agent_status').then(...) // → localhost:5000
```

#### **Option B: Migrate all endpoints to FastAPI** 
- More work: create new endpoints in FastAPI
- More consistent architecture
- Better for production

**Requires adding to `app/main.py`:**
```python
@app.get("/debug")
@app.get("/agent_status")
@app.post("/infer")
@app.post("/decide_nlp")
@app.post("/decide")
# ... etc
```

---

## 6. Testing Checklist

### ✅ AI Reply Panel (COMPLETE)
- [x] Frontend loads without errors
- [x] "Generate AI Reply" button shows panel
- [x] Request sends to `http://localhost:8000/api/v1/responses/generate`
- [x] Response displays in textarea
- [x] Confidence and safety warnings display
- [x] Tone selector works and regenerates
- [x] Approve/Edit/Reject buttons work
- [x] Feedback sent to FastAPI

### 🔄 Index.html Features (Still using Flask)
- [ ] Agent status badge updates
- [ ] /debug polling works
- [ ] Manual decision maker works
- [ ] Gmail fetch works

---

## 7. Quick Reference: API_CONFIG in Frontend

Both `ai_reply_panel.html` and `index.html` should use:

```javascript
const API_CONFIG = {
    BASE_URL: 'http://localhost:8000',
    ENDPOINTS: {
        GENERATE: '/api/v1/responses/generate',
        REGENERATE: '/api/v1/responses/regenerate',
        FEEDBACK: '/api/v1/responses/feedback',
        HEALTH: '/health',
        PREDICT: '/predict',
    }
};

// Usage:
fetch(API_CONFIG.BASE_URL + API_CONFIG.ENDPOINTS.GENERATE, {...})
```

---

## 8. Error Handling

All fetch calls now include:
- ✅ Network error handling (`catch (error)`)
- ✅ HTTP error status checking (`if (!response.ok)`)
- ✅ JSON parse error handling
- ✅ Response validation (required fields check)
- ✅ User-friendly error messages in UI
- ✅ Console logging for debugging

---

## 9. Deployment Notes

### Production Checklist:
- [ ] Update `API_CONFIG.BASE_URL` from `localhost:8000` to production URL
- [ ] Remove `allow_origins=["*"]` and specify actual frontend domain
- [ ] Enable HTTPS in frontend and backend
- [ ] Update CORS headers for specific origin

**Production example:**
```javascript
const API_CONFIG = {
    BASE_URL: 'https://api.yourdomain.com',
    // ...
};
```

```python
# In app/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific origin
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)
```

---

## Summary

| Component | Status | Notes |
|-----------|--------|-------|
| **Generate Response** | ✅ FIXED | Works with FastAPI |
| **Feedback Recording** | ✅ FIXED | Sends to FastAPI |
| **Error Handling** | ✅ FIXED | Network & API errors covered |
| **CORS** | ✅ CONFIGURED | Middleware active |
| **Index.html** | 🔄 IN PROGRESS | Can stay on Flask temporarily |

---

## Next Steps

1. ✅ Test "Generate AI Reply" button thoroughly
2. ✅ Monitor browser console for errors
3. 🔄 Optionally migrate remaining endpoints to FastAPI
4. 🔄 Update production domain in API_CONFIG

