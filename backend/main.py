"""
FastAPI backend for paged KV cache visualization.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import logging

from backend.api_models import SimulateRequest, SimulateResponse, ModelConfig, Event, PageState
from backend.simulator import KVCacheSimulator
from backend.narration import NarrationEngine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="KV-Paged Visualizer")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize simulator once
try:
    simulator = KVCacheSimulator(device="cpu")
    logger.info("✓ Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    simulator = None

# Initialize narration engine
narration_engine = NarrationEngine()


@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "KV-Paged Visualizer API",
        "endpoints": [
            "/simulate - POST with prompt",
            "/docs - API documentation",
        ],
    }


@app.post("/simulate", response_model=SimulateResponse)
def simulate(request: SimulateRequest):
    """
    Run simulation with a prompt.
    Returns event trace for visualization.
    """
    if simulator is None:
        return {"error": "Model not loaded"}

    logger.info(f"Simulating: {request.prompt}")

    try:
        result = simulator.simulate(request.prompt)
        
        # Extract tokens from result
        tokens = result.get("tokens", [])
        narration_engine.set_tokens(tokens)

        # Build events with narrations
        events_data = result["events"]
        enriched_events = []
        
        for event_dict in events_data:
            # Add narration
            narration = narration_engine.explain_event(event_dict, events_data)
            
            # Determine what changed
            changed = []
            if event_dict["event_type"] == "page_fault":
                changed.append(f"page_{event_dict['details'].get('page_id')}")
            elif event_dict["event_type"] == "kv_write":
                changed.append(f"page_{event_dict['details'].get('page_id')}_slot_{event_dict['details'].get('slot')}")
            elif event_dict["event_type"] == "page_freed":
                changed.append(f"page_{event_dict['details'].get('page_id')}")
            elif event_dict["event_type"] == "copy_on_write":
                changed.append(f"page_{event_dict['details'].get('source_page_id')}")
                changed.append(f"page_{event_dict['details'].get('new_page_id')}")
            
            event_dict["narration"] = narration
            event_dict["changed_elements"] = changed
            
            enriched_events.append(event_dict)

        # Get token timeline at final step
        token_timeline = narration_engine.get_token_mapping(len(enriched_events) - 1, enriched_events)

        # Convert to response model
        response = SimulateResponse(
            model=ModelConfig(
                layers=result["model"]["layers"],
                heads=result["model"]["heads"],
                hidden_size=result["model"]["hidden_size"],
                head_dim=result["model"]["head_dim"],
            ),
            events=[Event(**e) for e in enriched_events],
            final_pages=result["final_pages"],
            page_states=[PageState(**ps) for ps in result["page_states"]],
            summary=result["summary"],
            tokens=tokens,
            token_timeline=token_timeline,
        )

        logger.info(f"✓ Simulation complete: {response.summary['total_events']} events")
        return response

    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
        raise


# Mount static files
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=frontend_path, html=True), name="static")


@app.get("/health")
def health():
    """Health check"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
