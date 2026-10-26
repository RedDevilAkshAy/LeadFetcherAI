# LeadFetcherAI — Local Business Intelligence & Outreach Engine

LeadFetcherAI turns raw Google Places local business searches into fully audited, scored, and pitch-ready leads with personalized outreach copy.

---

## ⚡ Architecture & Pipeline

```text
Google Places (any niche & location, e.g. "dental clinics in Kochi")
        ↓
Extract Details (name, website, phone, rating, reviews, owner via Gemini)
        ↓
Crawl Website (services, pricing, contact, booking, WhatsApp)
        ↓
pgvector RAG (embed & understand business services, competitive market context)
        ↓
LangGraph Agent (diagnose conversion gaps & growth opportunities)
        ↓
Lead Score (0-100 score mapped to High / Medium / Low priority)
        ↓
Generate Personalized Pitch (custom WhatsApp message + cold outreach email)
```

---

## 🚀 Key Endpoints

### 1. `POST /find_new`
Fetches businesses, crawls websites, stores pgvector embeddings, diagnoses growth opportunities via LangGraph agent, scores leads, generates pitches, and persists everything to PostgreSQL.

**Request:**
```json
{
  "niche": "dental clinics",
  "locations": ["Kochi, Kerala"],
  "filter_ratings": false
}
```

### 2. `GET /leads/{id}`
Returns the comprehensive intelligence report for any lead:

**Example Response:**
```json
{
  "id": 1,
  "business_name": "SmileCare Dental",
  "location": "Kakkanad, Kochi",
  "business_website": "https://smilecaredental.in",
  "owner_name": "Dr. Rajesh Kumar",
  "rating": 4.2,
  "review_count": 18,
  "analysis": {
    "services_found": ["implants", "root canal", "teeth cleaning"],
    "has_online_booking": false,
    "has_whatsapp": false,
    "has_contact_form": true,
    "pricing_found": false,
    "website_summary": "Present, but no online booking or WhatsApp button"
  },
  "score": {
    "score": 85.0,
    "priority": "High",
    "opportunities": [
      "Local SEO + Google Business Profile optimization",
      "Instant WhatsApp booking integration"
    ],
    "recommended_actions": "Improve review volume, add WhatsApp booking, and create service pages for implants/aligners"
  },
  "pitch": {
    "whatsapp": "Hi Dr. Rajesh Kumar, I came across SmileCare Dental on Google. I noticed your clinic has strong services but only 18 Google reviews and no direct WhatsApp booking option on the website. I help local businesses improve Google visibility and turn searches into appointment enquiries. Would you be open to a free 10-minute audit this week?",
    "email": "Subject: Quick growth audit for SmileCare Dental's local visibility\n\nHi Dr. Rajesh Kumar,\n\n..."
  },
  "formatted_card": "Business: SmileCare Dental, Kakkanad\nWebsite: Present, but no online booking or WhatsApp button\nReviews: 4.2 stars, 18 reviews\nOpportunity: Local SEO + Google Business Profile optimization\nRecommended pitch: Improve review volume, add WhatsApp booking, and create service pages for implants/aligners\nPriority: High\n\nGenerated message:\nHi Dr. Rajesh Kumar, I came across SmileCare Dental on Google..."
}
```

### 3. `POST /leads/{id}/analyze`
Re-runs or initiates the intelligence pipeline on demand for any existing lead.

---

## 🐳 Containerization (Docker & Docker Compose)

The project includes a multi-container setup with:
- `db`: PostgreSQL 16 with `pgvector` pre-installed (`pgvector/pgvector:pg16`)
- `web`: FastAPI application with Python 3.12 and all dependencies

### Start the containers:
```bash
docker compose up --build -d
```

### Stop containers:
```bash
docker compose down
```

---

## 💻 Local Setup (Without Docker)

### 1. Prerequisites
- Python 3.12+
- PostgreSQL with `pgvector` extension

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```

### 4. Database Migrations
```bash
alembic upgrade head
```

### 5. Run Server
```bash
uvicorn server:app --reload --port 8000
```
Swagger API docs available at: `http://localhost:8000/docs`

---

## 🧪 Testing

Run all unit and integration tests:
```bash
python -m unittest discover -s . -p "test_*.py"
```
