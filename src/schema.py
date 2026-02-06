from pydantic import BaseModel

class OTTEvent(BaseModel):
    platform: str
    genre: str
    dataset_version: str
    dialogue_density: str
    attention_required: str
    episode_duration_min: int
    pacing_score: int
    hook_strength: int
    visual_intensity: int
    avg_watch_percentage: float
    pause_count: int
    rewind_count: int
    skip_intro: int
    cognitive_load: int
    season_number: int
    episode_number: int
    release_year: int
    night_watch_safe: int