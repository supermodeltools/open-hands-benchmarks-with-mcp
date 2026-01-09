import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

def _get_sdk_sha_direct(sdk_path: Path) -> str:
    """Get SHA directly from SDK directory using git rev-parse."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=sdk_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()

def get_sdk_sha() -> str:
    """Get the SHA of the SDK submodule."""
    sdk_path = PROJECT_ROOT / "vendor" / "software-agent-sdk"
    return _get_sdk_sha_direct(sdk_path)

SDK_SHA = get_sdk_sha()
SDK_SHORT_SHA = SDK_SHA[:7]
