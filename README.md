
[project_deep_dive.md](https://github.com/user-attachments/files/26323541/project_deep_dive.md)
# 🧠 Smart Retail Shelf Intelligence — Complete Technical Deep Dive

> **Purpose:** This document explains every single technology, library, pattern, and decision used in this project at a master level. You can use this to explain the project to anyone — from a fellow developer to a senior executive.

---

## 🏗️ Big Picture — What Is This System?

**ShelfIQ** is a full-stack AI-powered retail shelf monitoring platform. It solves a real business problem: **retailers lose revenue every single day because shelves go empty, products get misplaced, and nobody knows until it's too late.**

This system watches shelves 24/7 (using computer vision), detects problems in real-time, sends instant alerts to store staff, and uses machine learning to forecast which products will run out before they even do.

**The system has two main parts:**
1. **Backend** — A Python API server (the brain) that does all the intelligence
2. **Frontend** — A React dashboard (the eyes) that shows store managers everything in real-time

---

## 🐍 BACKEND — Python (FastAPI)

### 1. `FastAPI` — The Web Framework

**What it is:** FastAPI is a modern Python web framework for building APIs.

**Why we used it:**
- It is **extremely fast** — benchmarks show it's on par with Node.js and Go, because it uses Python's `asyncio` (async/await) model
- It **auto-generates API documentation** (Swagger UI at `/docs`) — anyone can test every endpoint without writing a single line of extra code
- It uses **Pydantic** for data validation built-in, so if someone sends wrong data, it rejects it with a clear error message
- It natively supports **WebSockets**, which we need for real-time alert streaming

**In our code:** `main.py` is the entire FastAPI app — all 30+ API endpoints are defined here.

---

### 2. `Uvicorn` — The ASGI Server

**What it is:** Uvicorn is the server that actually runs our FastAPI app. Think of it like a delivery vehicle — FastAPI is the package, Uvicorn is the truck that carries it.

**Why we used it:**
- It's an **ASGI** (Asynchronous Server Gateway Interface) server — designed specifically for async Python apps like ours
- It supports **live reloading** during development (`--reload` flag) so you don't have to restart the server every time you change code
- **Start command in production:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

**ASGI vs WSGI:** Old Python servers like Gunicorn use WSGI (synchronous). ASGI handles thousands of simultaneous connections without blocking — critical for WebSockets and real-time features.

---

### 3. `Pydantic` — Data Validation & Schemas

**What it is:** A Python library that enforces data types and structures using Python type hints.

**Why we used it:**
- FastAPI uses Pydantic automatically for **request body validation**
- If someone calls `POST /api/alerts/{id}/acknowledge` with a missing field, Pydantic immediately returns a `422 Unprocessable Entity` error with details — no custom validation code needed
- Models like `AlertAcknowledgeRequest` and `ShelfAnalysisRequest` are Pydantic `BaseModel` classes

**Example from code:**
```python
class AlertAcknowledgeRequest(BaseModel):
    alert_id: int
    acknowledged_by: Optional[str] = "store_manager"
```
This guarantees `alert_id` is always an integer. If you send a string, it auto-rejects with a helpful error.

---

### 4. `CORSMiddleware` — Cross-Origin Resource Sharing

**What it is:** A security mechanism in browsers that blocks a web page from making requests to a different domain than the one that served it.

**Why we need it:** Our **frontend is on Vercel** (e.g., `shelfiq.vercel.app`) and our **backend is on Render** (e.g., `shelfiq-api.onrender.com`). These are two different domains. Without CORS, the browser would block all frontend → backend requests.

**What we did:** We added `CORSMiddleware` to FastAPI and allowed all origins (`allow_origins=["*"]`):
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```
In production, you'd restrict this to just your frontend domain for security.

---

### 5. `asynccontextmanager` + Lifespan — App Startup/Shutdown

**What it is:** A way to run setup code when the server starts and cleanup code when it stops.

**Why we used it:**
- When the server starts, we need to **initialize the alert pipeline** (connect to Redis, set up queues)
- When the server stops, we log the shutdown cleanly
- This is FastAPI's modern replacement for the older `@app.on_event("startup")` decorators

**In our code:**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline = get_alert_pipeline()
    await pipeline.initialize()   # ← runs on startup
    yield
    logger.info("Shutting down...")  # ← runs on shutdown
```

---

### 6. `WebSocket` — Real-Time Communication

**What it is:** A protocol that keeps a persistent, two-way connection open between client and server. Unlike HTTP (which is request-response), WebSocket is like a phone call — once connected, both sides can send messages anytime.

**Why we needed it:**
- When a stockout is detected, the store manager needs to know **instantly** — not after refreshing the page
- Regular HTTP polling (asking "any new alerts?" every few seconds) is wasteful and slow
- WebSocket pushes the alert the moment it happens — **< 1 second delivery**

**How it works in our code:**
1. Frontend connects to `wss://shelfiq-api.onrender.com/ws/alerts`
2. Backend registers this connection in `AlertPipeline._ws_connections`
3. When an alert is generated, backend broadcasts JSON to **all connected clients**
4. Frontend receives it and shows a live notification immediately

**Keepalive mechanism:** We send a `ping` every 25 seconds from the frontend and the server responds with `pong`, so the connection doesn't time out.

---

### 7. Computer Vision Engine (`cv_engine/detector.py`)

**What it is:** The module that "looks at" shelf images and detects what products are present, their stock levels, and their positions.

#### a) `YOLOv8` (You Only Look Once v8) — Object Detection
**What it is:** A state-of-the-art real-time object detection neural network made by Ultralytics.
**Why YOLOv8:** It processes an image in a single pass (hence "You Only Look Once") — making it extremely fast. Version 8 is the latest and most accurate. It can detect and locate multiple objects in milliseconds.
**In our code:** Used in `_full_analysis()` method when `DEMO_MODE=false`. The model outputs bounding boxes (`xyxyn` = normalized coordinates), confidence scores, and class IDs.

#### b) `OpenCV` (cv2) — Image Processing
**What it is:** Open Computer Vision library — the industry standard for image manipulation.
**Why:** Used to decode uploaded image bytes into a numpy array that YOLOv8 can process:
```python
nparr = np.frombuffer(image_bytes, np.uint8)
img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
```

#### c) `CLIP` (Contrastive Language-Image Pre-Training) — SKU Recognition
**What it is:** An OpenAI model that understands images and text together. It can match an image to a product name.
**Why:** Used in `sku_recognizer.py` to identify which specific product a detected object is (e.g., is this a Coca-Cola or a Pepsi?) — without training a custom model.

#### d) `numpy` — Numerical Computing
**What it is:** The foundational library for scientific computing in Python — provides fast array operations.
**Why:** Image data is fundamentally a multi-dimensional array of pixel values. Numpy handles all mathematical operations (bounding box math, pixel arrays, calculations like WMAPE).

#### e) `Pillow` — Image I/O
**What it is:** Python Imaging Library — for opening, manipulating, and saving image files.
**Why:** Used as a dependency for loading and preprocessing images before feeding them to neural networks.

#### f) **Demo Mode** — The Simulation Layer
**Why:** Running YOLOv8 requires a GPU and large model files — impossible on a free cloud server (Render free tier). So we built a **Demo Mode** that generates mathematically realistic fake data using `random.seed(shelf_id)` so results are consistent per shelf. This lets the system run on any machine without any AI dependencies.

**Environment variable control:**
```
DEMO_MODE=true  → uses synthetic data (cloud deployment)
DEMO_MODE=false → uses YOLOv8 (production with GPU)
```

---

### 8. `@dataclass` — Structured Data Containers

**What it is:** A Python decorator that auto-generates `__init__`, `__repr__` etc. for classes that just hold data.

**Why:** `DetectedProduct` and `ShelfAnalysisResult` are plain data containers — they don't have complex behavior. Using `@dataclass` is cleaner and faster than writing a full class manually.

```python
@dataclass
class DetectedProduct:
    sku: str
    name: str
    confidence: float
    bbox: List[float]
    stock_level: str
    ...
```

---

### 9. **Singleton Pattern** — `get_detector()`, `get_forecaster()`, `get_alert_pipeline()`

**What it is:** A design pattern that ensures only **one instance** of a class exists in the entire application.

**Why we used it:** Loading a YOLOv8 model takes 2-5 seconds. If we created a new detector for every API request, the server would be impossibly slow. The Singleton pattern creates the detector once and reuses it forever.

```python
_detector: Optional[ShelfDetector] = None

def get_detector() -> ShelfDetector:
    global _detector
    if _detector is None:
        _detector = ShelfDetector()  # ← only created once
    return _detector
```

---

### 10. Forecasting Engine (`forecasting/demand_forecaster.py`)

#### a) `Facebook Prophet` — Time Series Forecasting
**What it is:** An open-source forecasting library developed by Meta (Facebook) Research, designed to handle time series data with strong seasonal patterns.

**Why Prophet:**
- Handles **weekly seasonality** (weekends sell more snacks), **annual seasonality** (summer beverages spike), and **holidays** automatically
- Allows adding **external regressors** — we add `temperature` as a regressor because hot days sell more cold drinks
- Robust to missing data and outliers
- **No manual parameter tuning needed** for basic use — it auto-learns patterns

**In our code (full mode):**
```python
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    changepoint_prior_scale=0.05
)
model.add_regressor("temperature")
model.fit(df[["ds", "y", "temperature"]])
```

#### b) `pandas` — Data Manipulation
**What it is:** The core data analysis library in Python — provides `DataFrame` (a table-like structure).

**Why:** Prophet requires data in a specific format with columns `ds` (date) and `y` (value). We use pandas to build this DataFrame from our synthetic POS data, manipulate time series, and compute statistics.

#### c) `scikit-learn` — Machine Learning Utilities
**What it is:** The most widely used ML library in Python.

**Why:** Used for statistical utilities like computing accuracy metrics. Also a dependency of Prophet.

#### d) `statsmodels` — Statistical Models
**Why:** Used as a statistical backend for some forecasting computations. Listed in full requirements.

#### e) **Synthetic POS Data Generation** — The Math Behind It
Since we don't have a real POS system, we generate 2 years of realistic sales data using:
- **Annual seasonality:** `sin(2π × (day_of_year - 90) / 365)` — creates a summer peak around July
- **Weekly seasonality:** `weekend_boost` factor (1.3x–1.5x on weekends)
- **Random promotions:** 10% probability of a 1.3x–2.0x demand spike
- **Weather effect:** Hot days boost beverage sales
- **Growth trend:** Slight month-over-month increase (`0.001 × days_elapsed / 30`)
- **Gaussian noise:** `random.gauss(0, base × 0.10)` — adds realistic randomness

#### f) **Reorder Point Formula** — Supply Chain Math
```
Reorder Point = (Avg Daily Demand × Lead Time) + Safety Stock
Safety Stock  = Z-score × Daily Std Dev × √(Lead Time)
```
- Z-score of 1.645 = 95% service level (meaning 95% of the time, stock won't run out before the order arrives)
- Lead time = 2 days (how long it takes to get restocked after ordering)

#### g) **WMAPE** — Forecast Accuracy Metric
**What it is:** Weighted Mean Absolute Percentage Error — the industry standard metric for forecast accuracy.
- `WMAPE = Σ|Actual - Forecast| / Σ|Actual|`
- **Lower is better.** Our system achieves ~8-18% WMAPE, which is excellent for retail.

---

### 11. Alert Pipeline (`alerting/alert_pipeline.py`)

#### a) `asyncio.Queue` — In-Memory Message Queue
**What it is:** A Python built-in async-safe queue that allows producers and consumers to exchange data without race conditions.

**Why:** When multiple CV analyses happen simultaneously and all trigger alerts, they need to be queued and broadcast one at a time without conflicts. `asyncio.Queue(maxsize=1000)` can hold up to 1000 pending alerts.

#### b) `Redis` + `aioredis` — Persistent Message Broker
**What Redis is:** An in-memory key-value store used as a database, cache, and message broker.

**Why Redis:**
- Used as a **Pub/Sub message broker** — when alert is published on channel `shelf:alerts`, all subscribers receive it instantly
- Stores alert history using a Redis List (`LPUSH`, `LTRIM`) — keeps the last 1000 alerts
- If the server restarts, we don't lose alert history
- **`aioredis`** is the async version that works with Python's `asyncio`

**Graceful degradation:** If Redis is not available (free cloud tier), the system falls back to in-memory queue automatically:
```python
try:
    self._redis = await aioredis.from_url(REDIS_URL)
except Exception:
    self._redis = None  # ← falls back to in-memory
```

#### c) **Priority Scoring** — Revenue-Weighted Alerts
Alerts are scored based on business impact: `Priority Score = Revenue Impact × Urgency Multiplier`
- Critical × 4.0, High × 3.0, Medium × 2.0, Low × 1.0
- So a ₹620/hr Amul stockout (critical) scores 2480 — **always shown first**

#### d) `Python Enum` — Type-Safe Constants
**Why:** Alert types, priorities, and statuses are defined as Python `Enum` classes — this prevents typos. If you accidentally type `"critcal"` instead of `"critical"`, the Enum catches it immediately.

---

### 12. Database Layer (`models/db_models.py`)

#### a) `SQLAlchemy` — ORM (Object-Relational Mapper)
**What it is:** Maps Python classes to database tables, so you interact with the database using Python objects instead of raw SQL.

**Why:** Writing raw SQL is error-prone and database-specific. SQLAlchemy lets you write Python, and it generates the correct SQL for PostgreSQL, SQLite, or any other database.

**Key models defined:**
- `Store` → `stores` table
- `Shelf` → `shelves` table (with health_score, compliance_score, camera_id)
- `Product` → `products` table (includes `embedding` JSON field for CLIP vectors)
- `Planogram` + `PlanogramItem` → planogram layout specification
- `Alert` → `alerts` table (type, priority, status, revenue_impact, timestamps)
- `SalesRecord` → POS transaction history
- `ForecastRecord` → stored demand forecasts
- `ShelfSnapshot` → historical shelf images and analysis results

#### b) `Alembic` — Database Migrations
**What it is:** A database migration tool for SQLAlchemy — manages changes to the database schema over time.

**Why:** When you need to add a column (e.g., adding `revenue_impact` to the alerts table), you can't just change the Python model. Alembic creates versioned migration scripts that safely update the live database.

#### c) `psycopg2-binary` — PostgreSQL Driver
**What it is:** The Python adapter for PostgreSQL — the bridge between Python and the Postgres database.

**Why binary:** The `-binary` variant includes compiled C extensions for better performance, without needing PostgreSQL development headers installed.

#### d) `JSON Column Type` — Flexible Schema
**Why:** Fields like `detected_products`, `violations`, `planogram spec`, and `CLIP embeddings` have complex, nested structure that changes over time. Storing them as PostgreSQL JSON columns avoids needing a normalized schema for every nested field.

---

### 13. Planogram Compliance Engine (`planogram/`)

**What a planogram is:** A visual specification from retail brands (like Coca-Cola or P&G) that dictates exactly which products go where on the shelf, how many facings each product gets, and at what price.

**What the compliance engine does:**
- Compares detected shelf state vs. the expected planogram
- Flags: missing products, misplaced products, wrong facing counts, unauthorized products
- Outputs a **compliance score** (0-100%)

---

### 14. Logging — `logging` Module

**What it is:** Python's built-in logging module.

**Why:** `print()` statements are not suitable for production. The `logging` module allows:
- Setting log levels (DEBUG, INFO, WARNING, ERROR)
- Structured log messages with timestamps and module names
- Future integration with log aggregation services (like Datadog or CloudWatch)

```python
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("✅ Smart Retail Shelf Intelligence API started")
```

---

### 15. Security Libraries (in `requirements.txt`)

| Library | Purpose |
|---|---|
| `python-jose` | JWT (JSON Web Token) — for user authentication tokens |
| `passlib` + `bcrypt` | Password hashing — stores passwords securely using bcrypt algorithm |
| `python-multipart` | Parses `multipart/form-data` requests — required for file uploads (shelf images) |
| `python-dotenv` | Loads environment variables from a `.env` file in development |

---

### 16. `httpx` — Async HTTP Client

**What it is:** A modern, async-capable HTTP client for Python.

**Why `httpx` over `requests`:** The `requests` library is synchronous — it blocks while waiting for a response. `httpx` is async, so it doesn't block the event loop. Critical in an async FastAPI app.

---

### 17. `Faker` — Synthetic Data Generation

**What it is:** A Python library for generating realistic fake data (names, addresses, emails, etc.)

**Why:** Used in development and demo mode to populate the database with realistic-looking store/product/alert data without needing real data.

---

### 18. `aiosmtplib` — Async Email

**What it is:** An async SMTP library for sending emails.

**Why:** Future feature — when a critical stockout happens, the system should email the store manager. The async version ensures email sending doesn't block the API.

---

### 19. `Celery` — Background Task Queue

**What it is:** A distributed task queue system for Python.

**Why:** Planned for production — scheduled background jobs like:
- Running shelf analysis every 5 minutes automatically
- Sending daily compliance reports
- Cleaning up old alert history
Works with Redis as the broker.

---

## ⚛️ FRONTEND — React + Vite

### 20. `React 18` — The UI Framework

**What it is:** A JavaScript library (made by Meta) for building user interfaces using reusable components.

**Why React:**
- **Component-based** — each piece of UI (AlertCard, Sidebar, MetricGauge) is its own self-contained component
- **Virtual DOM** — React only updates the parts of the page that actually changed, making it fast
- **Hooks** — `useState`, `useEffect`, `useContext`, `useCallback` let functional components manage state and side effects
- Same team that built the backend (Python) can also contribute to the React frontend — massive ecosystem

**Version 18 specifically:** Adds Concurrent Mode features, automatic batching of state updates, and `createRoot` API — resulting in smoother UI updates.

---

### 21. `Vite` — The Build Tool & Dev Server

**What it is:** A next-generation frontend build tool that replaces webpack.

**Why Vite over webpack:**
- **Instant dev server start** — uses native ES modules, no bundling during development
- **Hot Module Replacement (HMR)** — when you save a file, only that component updates in the browser, without full page refresh
- **Lightning-fast builds** — uses esbuild (written in Go) for bundling

**Key config in `vite.config.js`:**
```javascript
proxy: {
  '/api': { target: 'http://localhost:8000' },
  '/ws': { target: 'ws://localhost:8000', ws: true }
}
```
This **proxy** means during development, any request to `/api/*` is automatically forwarded to the FastAPI backend on port 8000. No CORS issues in dev.

---

### 22. `React Router DOM v6` — Client-Side Routing

**What it is:** A library for handling navigation in a single-page React app.

**Why:** Without a router, the browser would reload the entire page every time you click a nav link. React Router intercepts clicks and renders the correct component — no page reload, instant navigation.

**Key concepts used:**
- `<BrowserRouter>` — wraps the entire app, enables URL-based routing
- `<Routes>` + `<Route>` — maps URL paths to page components
- `<Navigate>` — redirect `/` to `/dashboard` automatically
- `useLocation()` — Sidebar uses this to know which page is active and highlight the correct nav item
- `useNavigate()` — programmatic navigation (e.g., clicking "Analyze Shelf" button goes to `/shelves`)
- `<Link>` — renders as an `<a>` tag but uses React Router (no page reload)

---

### 23. `Recharts` — Data Visualization

**What it is:** A charting library for React, built on top of D3.js.

**Why Recharts:**
- Built specifically for React — components are React components, not raw SVG/canvas
- Responsive by default (charts resize with the screen)
- Covers all chart types we need: LineChart (compliance trend), BarChart (demand forecast), AreaChart (revenue), HeatMap (stockout patterns)

**Used in:**
- `ForecastPage.jsx` — demand forecast bar chart with confidence intervals
- `AnalyticsPage.jsx` — compliance trend line chart, stockout heatmap
- `Dashboard.jsx` — KPI visualizations

---

### 24. Page Components (`src/pages/`)

| Page | What it Shows |
|---|---|
| `Dashboard.jsx` | Live KPI cards, store floor map, real-time alert feed, shelf health grid |
| `ShelvesPage.jsx` | Individual shelf cards, image upload for CV analysis, violation details |
| `AlertsPage.jsx` | Full alert history, filtering by type/priority, acknowledge button |
| `ForecastPage.jsx` | SKU selector, 7-day demand forecast chart, replenishment recommendations |
| `AnalyticsPage.jsx` | Compliance trend over 30 days, stockout heatmap by aisle/hour |

---

### 25. Reusable Components (`src/components/`)

| Component | Purpose |
|---|---|
| `Sidebar.jsx` | Navigation sidebar with custom SVG logo, live connection status dot |
| `Topbar.jsx` | Page header with title, subtitle, and action buttons |
| `AlertCard.jsx` | Individual alert display with priority badge, time-ago, revenue impact |
| `MetricGauge.jsx` | Circular gauge component for health/compliance scores |
| `ShelfMap.jsx` | Interactive SVG store floor map showing aisle colors by health score |

---

### 26. Custom Hook — `useAlerts` / `AlertProvider` (`src/hooks/useAlerts.jsx`)

**What a custom hook is:** A reusable JavaScript function that encapsulates React state and effects, shareable across multiple components.

**What `useAlerts` does — this is critical:**
1. On mount (`useEffect`), fetches initial 50 alerts from the REST API
2. Simultaneously establishes a **WebSocket connection** for live alerts
3. When a new live alert arrives via WebSocket, prepends it to the alerts list
4. Exposes `alerts`, `liveAlerts`, `connected` status, and `acknowledge()` function

**Why Context + Provider pattern:**
- The `AlertProvider` wraps the entire app in `App.jsx`
- **Any component** in the tree can call `useAlerts()` to get the alert data — no prop drilling
- One single WebSocket connection is shared across all pages — Dashboard, Alerts page, Sidebar badge all see the same data

**`useCallback`:**
The `acknowledge` function is wrapped in `useCallback` so its reference stays stable across renders — prevents unnecessary re-renders of children that receive it as a prop.

---

### 27. `api.js` — The API Client Layer

**What it is:** A centralized module for all backend communication.

**Why centralize:**
- API base URL (`VITE_API_URL`) is configured in one place
- All HTTP requests go through one `fetchJson` function that handles errors uniformly
- WebSocket connection is managed here with auto-reconnect logic

**Key techniques:**

`import.meta.env.VITE_API_URL` — Vite's way of reading environment variables. Variables prefixed with `VITE_` are exposed to the frontend bundle. This switches between local dev URL and production Render URL automatically.

**WebSocket auto-reconnect:**
```javascript
ws.onclose = () => {
  reconnectTimer = setTimeout(connect, 3000); // retry after 3 seconds
}
```
If the backend restarts or the connection drops, the frontend automatically reconnects in 3 seconds. Users never notice.

**FormData for file uploads:**
```javascript
const formData = new FormData();
formData.append('file', file);
fetch(`/api/analyze-shelf?shelf_id=${shelfId}`, { method: 'POST', body: formData })
```
Using `multipart/form-data` to upload actual shelf images to the CV engine.

---

### 28. `React.StrictMode`

**What it is:** A development tool that helps find potential problems. It renders components twice to detect side effects that shouldn't run twice.

**Why:** It's a zero-cost safety net in development — it disappears in production builds. It catches bugs early, like forgetting to clean up event listeners.

---

### 29. `index.css` — Global Design System

**What it is:** A single CSS file containing all design tokens and utilities — no external CSS framework.

**What it includes:**
- CSS custom properties (variables): `--accent-blue`, `--critical`, `--text-muted`, `--surface-2`, etc.
- This is the **design system** — one change to `--accent-blue` updates every button, badge, and highlight in the app simultaneously
- Dark theme by default (modern SaaS look)
- CSS animations: `@keyframes slideIn` for alert cards, `@keyframes ticker` for the scrolling live feed
- Glassmorphism effects: `backdrop-filter: blur()` for the sidebar
- Skeleton loading states: animated placeholder shimmer while data loads
- Responsive grid layouts: `kpi-grid`, `shelf-grid`, `grid-2-1`

**Why not TailwindCSS:** Vanilla CSS gives full control, no build step, no purging config. For a project this size, it's simpler and more maintainable.

---

### 30. `index.html` — The Shell

**What it is:** The single HTML file that Vite serves. React mounts into `<div id="root">`.

**Why a single HTML file:** This is a **Single Page Application (SPA)** — the initial HTML is minimal, and React dynamically renders the entire UI in JavaScript. The HTML just provides the mount point.

---

## 🚀 INFRASTRUCTURE & DEPLOYMENT

### 31. `Vercel` — Frontend Deployment

**What it is:** A cloud platform specialized for deploying frontend applications (React, Next.js, Vite, etc.)

**Why Vercel:**
- Zero-configuration deployment for Vite/React apps
- **Global CDN** — your frontend is served from the nearest server to the user worldwide → sub-100ms load times
- **Automatic HTTPS** — SSL certificate provided and auto-renewed
- **Git integration** — every push to `main` auto-deploys

**`vercel.json` config:**
```json
{
  "rewrites": [{ "source": "/(.*)", "destination": "/index.html" }]
}
```
This is the **SPA routing fix** — without it, refreshing `/alerts` would return a 404 (because Vercel would look for a real file at that path). The rewrite sends all routes to `index.html`, and React Router handles the rest.

---

### 32. `Render.com` — Backend Deployment

**What it is:** A cloud platform for deploying backend services, Docker containers, databases, etc.

**Why Render:**
- Free tier available for hobby/demo projects
- Native Python support — no Dockerfile needed for basic deployments
- Automatic HTTPS and custom domains
- Supports WebSocket connections (critical for our real-time alerts)

**`render.yaml` — Infrastructure as Code:**
```yaml
startCommand: cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT
envVars:
  - key: DEMO_MODE
    value: "true"
```
The `$PORT` environment variable is automatically set by Render — you never hardcode the port.

**Why `--host 0.0.0.0`:** By default, uvicorn listens only on `localhost` (only accessible from the same machine). `0.0.0.0` makes it listen on all network interfaces — required for cloud deployment.

---

### 33. `requirements-render.txt` — Optimized Dependencies for Cloud

**Why a separate requirements file:**
The full `requirements.txt` includes heavy libraries like `prophet` (500MB+), `torch`, `ultralytics` — which would take 10+ minutes to install and exceed memory limits on Render's free tier.

The render-specific file strips it down to only what Demo Mode needs:
- `fastapi`, `uvicorn`, `pydantic` — core API
- `numpy`, `pandas`, `scikit-learn` — basic ML utilities
- `opencv-python-headless` — headless = no GUI dependencies (no display needed on server)
- `redis`, `httpx` — connectivity

---

### 34. `Docker` + `docker-compose.yml` — Local Development Environment

**What Docker is:** A technology that packages an application and all its dependencies into a portable container. The container runs identically on any machine.

**Why Docker Compose:** Starts multiple services together with one command.

**Our services:**
```yaml
postgres:  The relational database (PostgreSQL 15)
redis:     The message broker and cache (Redis 7)
backend:   The FastAPI application
frontend:  The React/Vite dev server
```

**Why `alpine` images** (`postgres:15-alpine`, `redis:7-alpine`): Alpine Linux is a minimal OS image (~5MB vs ~100MB for Ubuntu). Makes containers start faster and use less disk.

**`healthcheck`:** Docker waits for Postgres to be actually ready (not just started) before starting the backend:
```yaml
depends_on:
  postgres:
    condition: service_healthy
```
Without this, the backend would crash because it tried to connect to Postgres before it finished initializing.

**Volume mounting (`./backend:/app`):** The backend code folder is mounted directly into the container. You can edit code on your Mac and the container sees the changes instantly — no rebuilding needed.

---

### 35. `Dockerfile` — Containerization

**What it is:** Instructions for building a Docker container image for the backend.

**Standard production pattern:** Install dependencies → copy code → expose port → run with uvicorn.

---

### 36. `runtime.txt` + `.python-version`

**`runtime.txt`:** Tells Render which Python version to use (`python-3.11.7`). Without this, Render might use a different default version, causing compatibility issues.

**`.python-version`:** Used by `pyenv` (Python version manager) on local machines to automatically switch to Python 3.11.7 when you `cd` into the project folder.

---

### 37. `.gitignore` — Version Control Hygiene

**What it does:** Tells Git which files NOT to track.

**Critical entries:**
- `node_modules/` — 200,000+ tiny files, never committed (recreated with `npm install`)
- `__pycache__/` — Python bytecode, auto-generated
- `.env` — environment variables (API keys, database passwords) — NEVER committed to Git
- `dist/` — built frontend files — generated fresh for each deployment
- `*.pyc` — compiled Python files

**Why this matters:** Committing `node_modules` or `.env` files is a common beginner mistake that can expose secrets and bloat the repository.

---

## 🎯 KEY ARCHITECTURAL DECISIONS EXPLAINED

### A) Why "Demo Mode" Architecture?

Running YOLOv8 + Prophet requires a GPU server costing $50-200/month. For showcasing the project, we built a **dual-mode** architecture:
- `DEMO_MODE=true` → runs on free cloud, uses math-based simulation
- `DEMO_MODE=false` → runs on GPU server with real AI

The interfaces are identical — switching is just one environment variable change.

### B) Why WebSocket Instead of Polling?

HTTP Polling: client asks "any alerts?" every 5 seconds = 12 requests/minute/user = server load.
WebSocket: connection stays open, server pushes when alert happens = 0 wasted requests.
For real-time in retail, the difference is critical — a stockout that's detected 5 seconds late vs instantly.

### C) Why Separate Frontend and Backend Deployment?

- **Separation of concerns** — frontend team and backend team can deploy independently
- **Scaling** — frontend (static files) scales infinitely via CDN; backend can scale vertically
- **Cost** — static files on Vercel are free; backend only needs a small server

### D) Why React Context for Alerts (vs. Redux)?

Redux is powerful but requires boilerplate. Our alert state is simple enough that React's built-in Context + useReducer pattern handles it cleanly. Rule of thumb: use Context for state shared across many components; use Redux when the state logic becomes very complex.

### E) Why PostgreSQL?

- **ACID compliance** — transactions are atomic, consistent, isolated, durable. Critical for financial data (revenue_impact, sales records)
- **JSON columns** — stores complex nested data (planogram specs, CLIP embeddings) alongside relational data
- **Spatial support** — future extension for shelf position queries

---

## 📊 SYSTEM FLOW — How Everything Connects

```
                    BROWSER (User)
                         │
          ┌──────────────┼──────────────┐
          │              │              │
     HTTP/REST      WebSocket      Static Files
          │              │              │
    FastAPI App     FastAPI App      Vercel CDN
     (Render)        (Render)
          │              │
    ┌─────┴─────┐        │
    │           │        │
  CV Engine  Forecaster  Alert Pipeline
  (Demo/    (Demo/       (Redis Pub/Sub)
  YOLOv8)   Prophet)          │
                           Redis
                              │
                           Broadcast →
                        All WS clients
```

**Step-by-Step User Flow:**
1. User opens the dashboard → React app loads from Vercel CDN
2. React calls `GET /api/shelves` → FastAPI runs CV engine → returns shelf health data
3. React opens WebSocket to `/ws/alerts` → backend registers the connection
4. User uploads a shelf image → `POST /api/analyze-shelf` → CV engine processes → returns violations
5. If a stockout is detected → Alert Pipeline publishes alert → Redis stores it → WebSocket broadcasts to all dashboards
6. All connected store managers see the alert **simultaneously, instantly**

---

## 📦 Complete File Structure Map

```
retail-shelf-intelligence/
├── backend/
│   ├── main.py                    # FastAPI app, all 30+ endpoints
│   ├── cv_engine/
│   │   ├── detector.py            # YOLOv8/Demo shelf analysis engine
│   │   └── sku_recognizer.py      # CLIP-based product identification
│   ├── forecasting/
│   │   └── demand_forecaster.py   # Prophet/Demo demand forecasting
│   ├── alerting/
│   │   └── alert_pipeline.py      # Redis Pub/Sub + WebSocket broadcaster
│   ├── models/
│   │   └── db_models.py           # SQLAlchemy ORM models
│   ├── planogram/
│   │   └── compliance_engine.py   # Planogram violation detector
│   ├── requirements.txt           # Full dependencies (with AI)
│   ├── requirements-render.txt    # Lightweight for cloud deployment
│   └── Dockerfile                 # Container definition
├── frontend/
│   ├── src/
│   │   ├── main.jsx               # React entry point
│   │   ├── App.jsx                # Router + AlertProvider wrapper
│   │   ├── api.js                 # HTTP + WebSocket client layer
│   │   ├── index.css              # Complete design system
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx      # Main overview page
│   │   │   ├── ShelvesPage.jsx    # Shelf monitor + CV upload
│   │   │   ├── AlertsPage.jsx     # Alert management
│   │   │   ├── ForecastPage.jsx   # Demand forecasting UI
│   │   │   └── AnalyticsPage.jsx  # Charts and heatmaps
│   │   ├── components/
│   │   │   ├── Sidebar.jsx        # Navigation with SVG logo
│   │   │   ├── Topbar.jsx         # Page header
│   │   │   ├── AlertCard.jsx      # Individual alert component
│   │   │   ├── MetricGauge.jsx    # Circular score gauge
│   │   │   └── ShelfMap.jsx       # SVG store floor map
│   │   └── hooks/
│   │       └── useAlerts.jsx      # WebSocket + Context provider
│   ├── index.html                 # SPA shell
│   ├── package.json               # npm dependencies
│   ├── vite.config.js             # Dev server + proxy config
│   └── vercel.json                # Vercel deployment config
├── docker-compose.yml             # Local dev: Postgres + Redis + API + Frontend
├── render.yaml                    # Render.com deployment config
├── runtime.txt                    # Python version for Render
└── .gitignore                     # Version control hygiene
```

---

## 🎤 Summary — What to Tell Anyone

> "We built a **real-time AI-powered retail automation platform** called ShelfIQ. The backend is written in **Python using FastAPI**, which is one of the fastest Python frameworks available. It runs a **computer vision engine** (designed to use YOLOv8 in production) that analyzes shelf images to detect stock levels, compliance violations, and misplaced products. A **demand forecasting engine** (built on Facebook Prophet time-series model) predicts which products will run out and when, so stores can reorder proactively. When something goes wrong — a stockout, a planogram violation — the system instantly broadcasts alerts to store managers using **WebSocket technology** (think real-time messaging, like WhatsApp but for shelf data). The frontend is a **React dashboard** deployed on Vercel's global CDN, with live metrics, interactive charts (Recharts), and a floor map. The backend is deployed on Render.com. The entire system is containerized with **Docker** for a one-command local setup. What makes it special is the **dual-mode architecture** — it runs on free cloud infrastructure using simulation mode, but can be switched to full GPU-based AI with one environment variable."
