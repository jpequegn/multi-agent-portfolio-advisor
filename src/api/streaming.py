"""Streaming API routes for Server-Sent Events (SSE).

This module provides the /analyze/stream endpoint for real-time
streaming of portfolio analysis progress.
"""

import asyncio
import contextlib
import time

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from src.api.routes import PortfolioRequest
from src.observability.tracing import TraceContext
from src.orchestration.state import (
    Portfolio,
    PortfolioState,
    Position,
    WorkflowStatus,
    create_initial_state,
)
from src.streaming import WorkflowEventEmitter

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["Analysis"])


async def run_streaming_workflow(
    state: PortfolioState,
    emitter: WorkflowEventEmitter,
) -> PortfolioState:
    """Run the workflow with streaming events.

    This function executes the workflow while emitting events
    to the provided emitter for SSE streaming.

    Args:
        state: Initial workflow state.
        emitter: Event emitter for streaming.

    Returns:
        Final workflow state.
    """
    from src.orchestration.workflow import create_workflow

    # Inject emitter into state for nodes to access
    state["_emitter"] = emitter

    # Create and run workflow
    workflow = create_workflow()

    try:
        result: PortfolioState = await workflow.ainvoke(state)
        return result
    finally:
        # Clean up emitter reference
        if "_emitter" in result:
            del result["_emitter"]


@router.post(
    "/analyze/stream",
    responses={
        200: {
            "description": "Streaming analysis with SSE events",
            "content": {"text/event-stream": {}},
        },
        400: {"description": "Invalid request"},
        500: {"description": "Internal error"},
    },
    summary="Stream portfolio analysis",
    description="""
Run portfolio analysis with real-time streaming updates via Server-Sent Events (SSE).

Events are streamed in the following format:
```
data: {"type": "workflow_started", "workflow_id": "...", ...}

data: {"type": "agent_started", "agent": "research", ...}

data: {"type": "agent_completed", "agent": "research", ...}

data: {"type": "workflow_completed", "status": "completed", ...}
```

Event types include:
- `workflow_started` - Analysis has begun
- `agent_started` - An agent has started processing
- `agent_progress` - Intermediate progress update
- `agent_completed` - An agent has finished
- `workflow_completed` - Analysis complete with results
- `error` - An error occurred
- `heartbeat` - Keep-alive signal (every 15s)

Use EventSource API on the client:
```javascript
const response = await fetch('/analyze/stream', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({positions: [{symbol: 'AAPL', quantity: 100}]})
});
const reader = response.body.getReader();
// Process SSE events...
```
""",
)
async def stream_analysis(
    request_body: PortfolioRequest,
    request: Request,
) -> StreamingResponse:
    """Stream portfolio analysis with real-time SSE events.

    Args:
        request_body: Portfolio analysis request.
        request: FastAPI request for disconnect detection.

    Returns:
        StreamingResponse with SSE events.
    """
    symbols = [p.symbol.upper() for p in request_body.positions]
    start_time = time.monotonic()

    # Create trace context
    async with TraceContext(
        session_id=request_body.user_id,
        user_id=request_body.user_id,
        metadata={
            "portfolio_size": len(request_body.positions),
            "symbols": symbols,
            "streaming": True,
        },
        tags=["portfolio-analysis", "streaming"],
    ):
        # Convert request to Portfolio model
        positions = [
            Position(
                symbol=p.symbol.upper(),
                quantity=p.quantity,
                cost_basis=p.cost_basis,
                sector=p.sector,
            )
            for p in request_body.positions
        ]

        portfolio = Portfolio(
            positions=positions,
            total_value=request_body.total_value or sum(p.quantity for p in positions),
            cash=request_body.cash,
            account_type=request_body.account_type,
        )

        # Create initial state
        state = create_initial_state(
            portfolio=portfolio,
            user_request=request_body.user_request,
            user_id=request_body.user_id,
        )

        workflow_id = state["workflow_id"]
        trace_id = state["trace_id"]

        logger.info(
            "streaming_analysis_started",
            workflow_id=workflow_id,
            trace_id=trace_id,
            symbol_count=len(symbols),
        )

        # Create event emitter
        emitter = WorkflowEventEmitter(
            workflow_id=workflow_id,
            trace_id=trace_id,
        )

        async def event_generator():
            """Generate SSE events from the workflow."""
            workflow_task: asyncio.Task[PortfolioState] | None = None

            try:
                # Start heartbeat
                emitter.start_heartbeat()

                # Emit workflow started
                await emitter.emit_workflow_started(
                    symbols=symbols,
                    user_request=request_body.user_request,
                )

                # Start workflow in background
                workflow_task = asyncio.create_task(
                    run_streaming_workflow(state, emitter)
                )

                # Stream events until complete or disconnected
                async for sse_data in emitter.events_as_sse():
                    # Check for client disconnect
                    if await request.is_disconnected():
                        logger.info(
                            "client_disconnected",
                            workflow_id=workflow_id,
                        )
                        break

                    yield sse_data

                # Wait for workflow to complete
                if workflow_task and not workflow_task.done():
                    result_state = await workflow_task

                    # Emit final completion if not already done
                    if not emitter.is_closed:
                        latency_ms = (time.monotonic() - start_time) * 1000
                        await emitter.emit_workflow_completed(
                            status=result_state.get("status", WorkflowStatus.COMPLETED.value),
                            has_errors=bool(result_state.get("errors")),
                            result={
                                "latency_ms": round(latency_ms, 2),
                                "has_research": result_state.get("research") is not None,
                                "has_analysis": result_state.get("analysis") is not None,
                                "has_recommendation": result_state.get("recommendation") is not None,
                            },
                        )

                        # Yield the final event
                        async for final_sse in emitter.events_as_sse():
                            yield final_sse

            except Exception as e:
                logger.error(
                    "streaming_error",
                    workflow_id=workflow_id,
                    error=str(e),
                )
                # Emit error event
                if not emitter.is_closed:
                    await emitter.emit_error(
                        message=str(e),
                        error_type=type(e).__name__,
                    )
                    await emitter.close()
                    # Yield error event
                    async for error_sse in emitter.events_as_sse():
                        yield error_sse

            finally:
                # Ensure cleanup
                if workflow_task and not workflow_task.done():
                    workflow_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await workflow_task

                if not emitter.is_closed:
                    await emitter.close()

                logger.info(
                    "streaming_analysis_finished",
                    workflow_id=workflow_id,
                    latency_ms=round((time.monotonic() - start_time) * 1000, 2),
                )

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )
