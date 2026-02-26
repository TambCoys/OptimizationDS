"""
Formatting helpers for benchmark/report output.
"""


def format_seconds(seconds):
	"""Format timing values with enough precision for very fast runs."""
	if seconds < 1e-3:
		return f"{seconds:.3e}"
	if seconds < 1:
		return f"{seconds:.6f}"
	return f"{seconds:.4f}"
