# Operations Runbook

This runbook provides operational procedures for the Multi-Agent Portfolio Advisor system.

## Table of Contents

- [Quick Reference](#quick-reference)
- [Startup Procedures](#startup-procedures)
- [Shutdown Procedures](#shutdown-procedures)
- [Health Checks](#health-checks)
- [Common Issues](#common-issues)
- [Debugging Workflows](#debugging-workflows)
- [Monitoring & Alerts](#monitoring--alerts)
- [Maintenance Tasks](#maintenance-tasks)
- [Escalation Procedures](#escalation-procedures)

## Quick Reference

### Service URLs

| Service | URL | Description |
|---------|-----|-------------|
| API | http://localhost:8000 | Main API endpoint |
| Swagger UI | http://localhost:8000/docs | API documentation |
| Health Check | http://localhost:8000/health/ready | Readiness probe |
| Langfuse | http://localhost:3000 | Observability dashboard |
| PostgreSQL | localhost:5432 | Database |
| Redis | localhost:6379 | Cache |

### Key Commands

```bash
# Start all services
docker compose up -d

# Check service status
docker compose ps

# View logs
docker compose logs -f api

# Stop all services
docker compose down

# Restart a service
docker compose restart api
```

## Startup Procedures

### Full System Startup

1. **Start infrastructure services first:**

```bash
# Start database and cache
docker compose up -d postgres redis

# Wait for services to be ready (10-15 seconds)
sleep 15

# Verify PostgreSQL
docker compose exec postgres pg_isready
# Expected: accepting connections

# Verify Redis
docker compose exec redis redis-cli ping
# Expected: PONG
```

2. **Start Langfuse (optional but recommended):**

```bash
docker compose up -d langfuse

# Wait for Langfuse to initialize
sleep 30

# Verify Langfuse
curl -s http://localhost:3000/api/public/health | jq
# Expected: {"status":"OK"}
```

3. **Start the API:**

```bash
docker compose up -d api

# Wait for API to start
sleep 10

# Verify API health
curl -s http://localhost:8000/health/ready | jq
```

4. **Verify full system:**

```bash
# Run a test analysis
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"positions": [{"symbol": "AAPL", "quantity": 10}]}' | jq
```

### Environment Verification

Before starting, verify environment variables are set:

```bash
# Required variables
echo "ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:+SET}"
echo "DATABASE_URL: ${DATABASE_URL:+SET}"

# Optional variables
echo "LANGFUSE_PUBLIC_KEY: ${LANGFUSE_PUBLIC_KEY:-NOT SET}"
echo "REDIS_URL: ${REDIS_URL:-NOT SET}"
```

## Shutdown Procedures

### Graceful Shutdown

```bash
# Stop API first (allows in-flight requests to complete)
docker compose stop api

# Wait for requests to drain (30 seconds)
sleep 30

# Stop remaining services
docker compose down
```

### Emergency Shutdown

```bash
# Immediate stop (may lose in-flight requests)
docker compose down --timeout 5
```

### Data Preservation

Before shutting down for maintenance:

```bash
# Backup PostgreSQL data
docker compose exec postgres pg_dump -U postgres portfolio_advisor > backup.sql

# Export Redis data
docker compose exec redis redis-cli BGSAVE
```

## Health Checks

### Liveness Check

Simple check that the service is running:

```bash
curl -s http://localhost:8000/health
# Expected: {"status": "ok", "timestamp": "..."}
```

### Readiness Check

Full check including all dependencies:

```bash
curl -s http://localhost:8000/health/ready | jq
```

**Healthy Response:**
```json
{
  "status": "ready",
  "checks": {
    "postgresql": {"status": "ok", "latency_ms": 1.5},
    "redis": {"status": "ok", "latency_ms": 0.8},
    "llm": {"status": "ok", "latency_ms": 50.2}
  },
  "version": "1.0.0"
}
```

**Unhealthy Response (HTTP 503):**
```json
{
  "status": "not_ready",
  "checks": {
    "postgresql": {"status": "unhealthy", "error": "Connection refused"},
    "redis": {"status": "ok", "latency_ms": 0.8}
  }
}
```

### Component-Specific Checks

```bash
# PostgreSQL
docker compose exec postgres pg_isready -U postgres

# Redis
docker compose exec redis redis-cli ping

# API process
docker compose exec api ps aux | grep python
```

## Common Issues

### Issue: High Latency (>30s responses)

**Symptoms:**
- API responses taking longer than expected
- Timeouts in client applications
- P95 latency alerts firing

**Diagnosis:**

1. Check Langfuse for slow spans:
   - Navigate to Langfuse → Traces
   - Sort by duration descending
   - Identify which agent is slow

2. Check external API latency:
```bash
# Test market data API
time curl -s "https://api.example.com/quote/AAPL"

# Check API logs for slow responses
docker compose logs api | grep "latency_ms" | awk -F'latency_ms=' '{print $2}' | sort -n | tail -10
```

3. Check Redis cache hit rate:
```bash
docker compose exec redis redis-cli INFO stats | grep hit
# keyspace_hits / (keyspace_hits + keyspace_misses) should be >80%
```

**Resolution:**

| Cause | Solution |
|-------|----------|
| Cold cache | Wait for cache to warm up |
| External API slow | Enable degradation mode |
| LLM slow | Check Anthropic status page |
| Database slow | Check PostgreSQL performance |

### Issue: High Error Rate (>5%)

**Symptoms:**
- Increased 500 responses
- Error alerts firing
- Failed analysis requests

**Diagnosis:**

1. Check circuit breaker states:
```bash
# Check API logs for circuit breaker events
docker compose logs api | grep "circuit"
```

2. Check Langfuse for error patterns:
   - Navigate to Langfuse → Traces
   - Filter by status = "error"
   - Look for common error messages

3. Review recent failures:
```bash
# Get recent error logs
docker compose logs api --since 1h | grep -i error | tail -50
```

**Resolution:**

| Cause | Solution |
|-------|----------|
| Circuit breaker open | Wait for recovery or restart |
| API rate limited | Reduce request rate |
| Invalid API key | Verify ANTHROPIC_API_KEY |
| Database connection | Check PostgreSQL status |

### Issue: High Costs (>$0.50/request)

**Symptoms:**
- Anthropic billing higher than expected
- Cost alerts firing

**Diagnosis:**

1. Check per-agent costs in Langfuse:
   - Navigate to Langfuse → Analytics
   - Group by agent/model
   - Identify expensive operations

2. Review cache hit rates:
```bash
docker compose exec redis redis-cli INFO stats
```

3. Check for retry storms:
```bash
docker compose logs api | grep "retry" | wc -l
```

**Resolution:**

| Cause | Solution |
|-------|----------|
| Low cache hit rate | Increase TTL, verify Redis working |
| Retry storms | Fix underlying issue, add backoff |
| Large prompts | Review agent prompts |
| Expensive model | Consider model switching |

### Issue: Service Won't Start

**Symptoms:**
- Container exits immediately
- Health checks failing

**Diagnosis:**

```bash
# Check container status
docker compose ps

# Check container logs
docker compose logs api

# Check for port conflicts
lsof -i :8000
```

**Common Causes:**

| Error | Solution |
|-------|----------|
| Port already in use | Stop conflicting service |
| Missing env vars | Check .env file |
| Database not ready | Start postgres first, wait |
| Permission denied | Check file permissions |

### Issue: Memory Exhaustion

**Symptoms:**
- OOM killer terminating containers
- Slow performance, swapping

**Diagnosis:**

```bash
# Check container memory usage
docker stats

# Check system memory
free -h
```

**Resolution:**

```bash
# Restart with memory limits
docker compose down
docker compose up -d --scale api=1
```

## Debugging Workflows

### Debug a Failed Request

1. **Get the trace_id** from the error response or logs:
```json
{
  "error": "Analysis failed",
  "trace_id": "trace-abc123"
}
```

2. **Find the trace in Langfuse:**
   - Navigate to: `http://localhost:3000/project/<project>/traces/<trace_id>`
   - Or search in the Traces view

3. **Identify the failed span:**
   - Look for red/error status spans
   - Check the span's input and output
   - Review error messages

4. **Analyze the failure:**
   - Check if it's a transient error (retry may help)
   - Check if input data was malformed
   - Check if external service was unavailable

5. **Replay if needed:**
   - Copy the input from the failed span
   - Modify if necessary
   - Submit a new request

### Debug Agent Behavior

1. **Enable debug logging:**
```bash
export LOG_LEVEL=DEBUG
docker compose restart api
```

2. **Watch agent execution:**
```bash
docker compose logs -f api | grep -E "(research|analysis|recommendation)_node"
```

3. **Check agent inputs/outputs in Langfuse:**
   - Each agent has its own span
   - Input: What the agent received
   - Output: What the agent produced

### Debug Database Issues

```bash
# Connect to PostgreSQL
docker compose exec postgres psql -U postgres portfolio_advisor

# Check connections
SELECT count(*) FROM pg_stat_activity;

# Check locks
SELECT * FROM pg_locks WHERE NOT granted;

# Check slow queries
SELECT query, calls, mean_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;
```

### Debug Redis Issues

```bash
# Connect to Redis
docker compose exec redis redis-cli

# Check memory
INFO memory

# Check keys
KEYS *

# Monitor commands in real-time
MONITOR
```

## Monitoring & Alerts

### Key Metrics

| Metric | Normal Range | Alert Threshold |
|--------|--------------|-----------------|
| Request latency P50 | <5s | >10s |
| Request latency P95 | <15s | >30s |
| Error rate | <1% | >5% |
| Cache hit rate | >80% | <50% |
| Cost per request | <$0.20 | >$0.50 |

### Setting Up Alerts

#### Latency Alert

```yaml
# prometheus-rules.yaml
- alert: HighLatency
  expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 30
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "High P95 latency detected"
```

#### Error Rate Alert

```yaml
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Error rate exceeds 5%"
```

### Langfuse Dashboards

Create dashboards for:

1. **Overview Dashboard:**
   - Request volume
   - Success/failure rate
   - Average latency

2. **Agent Performance:**
   - Per-agent latency
   - Per-agent success rate
   - Token usage by agent

3. **Cost Tracking:**
   - Daily cost trends
   - Cost by model
   - Cost by agent

## Maintenance Tasks

### Daily Tasks

- [ ] Review error logs
- [ ] Check Langfuse for anomalies
- [ ] Verify backup completion

### Weekly Tasks

- [ ] Review cost trends
- [ ] Check cache performance
- [ ] Update dependencies (security patches)

### Monthly Tasks

- [ ] Database maintenance (VACUUM, ANALYZE)
- [ ] Review and rotate logs
- [ ] Capacity planning review

### Database Maintenance

```bash
# Run VACUUM ANALYZE
docker compose exec postgres psql -U postgres portfolio_advisor -c "VACUUM ANALYZE;"

# Check table sizes
docker compose exec postgres psql -U postgres portfolio_advisor -c "
SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC;"
```

### Log Rotation

```bash
# Rotate Docker logs
docker compose logs --no-log-prefix api > logs/api-$(date +%Y%m%d).log
docker compose down
docker compose up -d
```

### Cache Maintenance

```bash
# Clear all cache (use with caution)
docker compose exec redis redis-cli FLUSHALL

# Clear specific patterns
docker compose exec redis redis-cli KEYS "cache:market:*" | xargs redis-cli DEL
```

## Escalation Procedures

### Severity Levels

| Level | Description | Response Time | Examples |
|-------|-------------|---------------|----------|
| P1 - Critical | Service down | 15 minutes | API unreachable, all requests failing |
| P2 - High | Major degradation | 1 hour | >50% error rate, severe latency |
| P3 - Medium | Partial issues | 4 hours | Single agent failing, cache issues |
| P4 - Low | Minor issues | 24 hours | Cosmetic bugs, minor performance |

### Escalation Path

1. **On-Call Engineer** (P1-P4)
   - First responder
   - Can restart services
   - Can enable degradation mode

2. **Team Lead** (P1-P2)
   - Escalate if unresolved in 30 minutes
   - Can approve emergency changes

3. **Engineering Manager** (P1)
   - Escalate if customer impact >1 hour
   - Communication coordination

### Communication Templates

**Incident Start:**
```
[INCIDENT] Portfolio Advisor - P{1/2/3} - {Brief Description}
Impact: {Who is affected}
Status: Investigating
Next Update: {Time}
```

**Incident Update:**
```
[UPDATE] Portfolio Advisor Incident
Status: {Investigating/Identified/Monitoring}
Root Cause: {If known}
Mitigation: {Actions taken}
Next Update: {Time}
```

**Incident Resolved:**
```
[RESOLVED] Portfolio Advisor Incident
Duration: {X hours Y minutes}
Root Cause: {Description}
Resolution: {What fixed it}
Follow-up: {Post-mortem scheduled for X}
```

## Appendix

### Useful Log Queries

```bash
# Find all errors in last hour
docker compose logs --since 1h api | grep -i error

# Find slow requests (>10s)
docker compose logs api | grep "latency_ms" | awk -F'latency_ms=' '{if ($2 > 10000) print}'

# Find circuit breaker events
docker compose logs api | grep -E "(circuit_opened|circuit_closed)"

# Count requests by status
docker compose logs api | grep "status=" | grep -oP 'status=\d+' | sort | uniq -c
```

### Environment Variable Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | - | Claude API key |
| `DATABASE_URL` | Yes | - | PostgreSQL connection string |
| `REDIS_URL` | No | `redis://localhost:6379` | Redis connection string |
| `LANGFUSE_PUBLIC_KEY` | No | - | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | No | - | Langfuse secret key |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `ENABLE_CACHING` | No | `true` | Enable Redis caching |
| `ENABLE_TRACING` | No | `true` | Enable Langfuse tracing |

### Docker Compose Services

| Service | Image | Ports | Dependencies |
|---------|-------|-------|--------------|
| api | portfolio-advisor | 8000 | postgres, redis |
| postgres | postgres:16 | 5432 | - |
| redis | redis:7 | 6379 | - |
| langfuse | langfuse/langfuse | 3000 | postgres |
