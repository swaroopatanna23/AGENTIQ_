"""One error type for the whole pipeline, so the frontend can show a clear message."""


class IntegrationError(Exception):
    """A problem at one stage of the pipeline.

    stage is one of: 'data', 'validation', 'input', 'ollama', 'nlp', 'adapter', 'chart'.
    The message is written for the end user (no Python jargon).
    """

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage
        self.message = message

    def __str__(self) -> str:
        return f"[{self.stage}] {self.message}"
