from rich.console import Console

from .graph import (
    create_deep_prep_workflow as create_deep_prep_workflow,
    create_initial_state as create_initial_state,
)
from .storage import save_final_state_to_filesystem as save_final_state_to_filesystem
from .observability import (
    get_token_usage as get_token_usage,
    reset_token_usage as reset_token_usage,
)

# Global console object for consistent output
console = Console()
