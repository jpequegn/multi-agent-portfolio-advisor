# Portfolio Advisor API Documentation

## Overview

The Portfolio Advisor API provides AI-powered portfolio analysis using a multi-agent system. Submit your portfolio holdings and receive comprehensive analysis including market research, risk assessment, and trade recommendations.

## Base URL

```
http://localhost:8000
```

## Interactive Documentation

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **OpenAPI Spec**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

## Quick Start

### 1. Start the Services

```bash
docker compose up -d
```

### 2. Verify Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "timestamp": "2024-01-15T10:00:00Z"
}
```

### 3. Analyze a Portfolio

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "positions": [
      {"symbol": "AAPL", "quantity": 100, "cost_basis": 150.00},
      {"symbol": "GOOGL", "quantity": 50, "cost_basis": 2800.00}
    ],
    "user_request": "Analyze risk and suggest rebalancing"
  }'
```

## Endpoints

### Analysis Endpoints

#### POST /analyze

Run a full portfolio analysis using the multi-agent workflow.

**Request Body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `positions` | array | Yes | List of portfolio positions |
| `positions[].symbol` | string | Yes | Stock ticker symbol (1-10 chars) |
| `positions[].quantity` | number | Yes | Number of shares (> 0) |
| `positions[].cost_basis` | number | No | Average cost per share in USD |
| `positions[].sector` | string | No | Sector classification |
| `user_request` | string | No | Natural language analysis request |
| `total_value` | number | No | Total portfolio value in USD |
| `cash` | number | No | Available cash balance (default: 0) |
| `account_type` | string | No | Account type: taxable, ira, roth_ira, 401k |
| `user_id` | string | No | User identifier for tracking |

**Example Request**

```json
{
  "positions": [
    {
      "symbol": "AAPL",
      "quantity": 100,
      "cost_basis": 150.00,
      "sector": "Technology"
    },
    {
      "symbol": "GOOGL",
      "quantity": 50,
      "cost_basis": 2800.00,
      "sector": "Technology"
    },
    {
      "symbol": "JNJ",
      "quantity": 75,
      "cost_basis": 160.00,
      "sector": "Healthcare"
    }
  ],
  "user_request": "Analyze portfolio risk and suggest rebalancing to reduce tech concentration",
  "total_value": 250000.00,
  "cash": 10000.00,
  "account_type": "taxable",
  "user_id": "user-123"
}
```

**Response**

| Field | Type | Description |
|-------|------|-------------|
| `workflow_id` | string | Unique workflow identifier |
| `trace_id` | string | Trace ID for observability (use in Langfuse) |
| `status` | string | Workflow status: pending, running, completed, failed |
| `research` | object | Research agent output |
| `analysis` | object | Analysis agent output |
| `recommendations` | object | Recommendation agent output |
| `errors` | array | Any errors encountered during processing |
| `latency_ms` | number | Total processing time in milliseconds |
| `started_at` | string | ISO 8601 timestamp when processing started |
| `completed_at` | string | ISO 8601 timestamp when processing completed |

**Example Response**

```json
{
  "workflow_id": "wf-abc123",
  "trace_id": "trace-xyz789",
  "status": "completed",
  "research": {
    "market_data": {
      "AAPL": {"price": 185.50, "change_percent": 1.2},
      "GOOGL": {"price": 142.30, "change_percent": -0.5}
    },
    "news": [
      {"title": "Apple Reports Strong Q4 Earnings", "sentiment": "positive"}
    ],
    "summary": "Tech sector showing mixed signals...",
    "symbols_researched": ["AAPL", "GOOGL", "JNJ"]
  },
  "analysis": {
    "risk_metrics": {
      "portfolio_volatility": 0.18,
      "beta": 1.15,
      "sharpe_ratio": 0.85,
      "max_drawdown": -0.12
    },
    "correlations": {
      "AAPL-GOOGL": 0.72,
      "AAPL-JNJ": 0.25
    },
    "benchmark_comparison": {
      "vs_spy_ytd": 0.05,
      "tracking_error": 0.08
    },
    "recommendations": [
      "Consider reducing technology sector concentration",
      "Portfolio has higher beta than benchmark"
    ],
    "summary": "Portfolio shows moderate risk with tech concentration..."
  },
  "recommendations": {
    "trades": [
      {
        "symbol": "AAPL",
        "action": "SELL",
        "quantity": 20,
        "rationale": "Reduce tech concentration"
      },
      {
        "symbol": "VTI",
        "action": "BUY",
        "quantity": 15,
        "rationale": "Increase diversification"
      }
    ],
    "summary": "Recommend rebalancing to reduce sector concentration...",
    "total_trades": 2,
    "buy_count": 1,
    "sell_count": 1,
    "hold_count": 1
  },
  "errors": [],
  "latency_ms": 2543.21,
  "started_at": "2024-01-15T10:00:00Z",
  "completed_at": "2024-01-15T10:00:02.543Z"
}
```

#### GET /analyze/{trace_id}

Retrieve a previous analysis by its trace ID.

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `trace_id` | string | The trace ID returned from a previous analysis |

**Note**: Requires state persistence to be configured. Returns 404 if not found.

#### GET /analyze

List previous analyses with optional filtering.

**Query Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user_id` | string | - | Filter by user ID |
| `status` | string | - | Filter by status |
| `limit` | integer | 50 | Maximum results to return |
| `offset` | integer | 0 | Number of results to skip |

**Note**: Requires state persistence to be configured. Returns empty list if not configured.

### Health Endpoints

#### GET /health

Basic liveness check to confirm the service is running.

**Response**

```json
{
  "status": "ok",
  "timestamp": "2024-01-15T10:00:00Z"
}
```

#### GET /health/live

Kubernetes-style liveness probe. Alias for `/health`.

#### GET /health/ready

Full readiness check that verifies all dependencies.

**Response (Ready)**

```json
{
  "status": "ready",
  "checks": {
    "postgresql": {"status": "ok", "latency_ms": 1.5},
    "redis": {"status": "ok", "latency_ms": 0.8},
    "llm": {"status": "ok", "latency_ms": 50.2}
  },
  "timestamp": "2024-01-15T10:00:00Z",
  "version": "1.0.0"
}
```

**Response (Not Ready)** - Returns HTTP 503

```json
{
  "status": "not_ready",
  "checks": {
    "postgresql": {"status": "unhealthy", "error": "Connection refused"},
    "redis": {"status": "ok", "latency_ms": 0.8}
  },
  "timestamp": "2024-01-15T10:00:00Z",
  "version": "1.0.0"
}
```

## Error Responses

All errors follow a consistent format:

```json
{
  "error": "Error message",
  "detail": "Detailed error information (optional)",
  "trace_id": "trace-id-if-available"
}
```

### HTTP Status Codes

| Code | Description | Example |
|------|-------------|---------|
| 200 | Success | Analysis completed successfully |
| 400 | Bad Request | Invalid JSON in request body |
| 404 | Not Found | Analysis with given trace_id not found |
| 422 | Validation Error | Missing required field or invalid value |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Unexpected error during processing |
| 503 | Service Unavailable | Service not ready (dependencies down) |

### Validation Error Response

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "positions"],
      "msg": "Field required",
      "input": {}
    }
  ]
}
```

## Rate Limits

| Limit | Value |
|-------|-------|
| Requests per minute (per IP) | 100 |
| Concurrent analyses (per user) | 10 |

When rate limited, the API returns HTTP 429 with a `Retry-After` header.

## Observability

### Trace IDs

Every analysis request returns a `trace_id` that can be used to:

1. **Track the request** through the multi-agent system
2. **View in Langfuse** for detailed execution traces
3. **Debug issues** by correlating logs and metrics
4. **Analyze performance** of individual agents

### Langfuse Integration

View traces in the Langfuse dashboard:

```
https://cloud.langfuse.com/project/<project-id>/traces/<trace-id>
```

Or if running locally:

```
http://localhost:3000/project/<project-id>/traces/<trace-id>
```

## Authentication

**Note**: Authentication is not yet implemented. The API currently accepts all requests.

Future authentication will support:
- API Key authentication via `X-API-Key` header
- JWT tokens via `Authorization: Bearer <token>` header

## SDK Examples

### Python

```python
import httpx

async def analyze_portfolio():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/analyze",
            json={
                "positions": [
                    {"symbol": "AAPL", "quantity": 100},
                    {"symbol": "GOOGL", "quantity": 50}
                ]
            }
        )
        return response.json()
```

### JavaScript/TypeScript

```typescript
const response = await fetch("http://localhost:8000/analyze", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    positions: [
      { symbol: "AAPL", quantity: 100 },
      { symbol: "GOOGL", quantity: 50 },
    ],
  }),
});

const result = await response.json();
console.log(result.trace_id);
```

### cURL

```bash
# Basic analysis
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"positions": [{"symbol": "AAPL", "quantity": 100}]}'

# With custom request
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "positions": [
      {"symbol": "AAPL", "quantity": 100, "cost_basis": 150},
      {"symbol": "VTI", "quantity": 200}
    ],
    "user_request": "Focus on tax-loss harvesting opportunities",
    "account_type": "taxable"
  }'
```

## Changelog

### v1.0.0

- Initial release
- Portfolio analysis endpoint
- Health check endpoints
- Multi-agent workflow integration
- Langfuse observability
