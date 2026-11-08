# FastAPI basics

FastAPI is a Python web framework for building APIs. Routes are plain functions decorated with the HTTP method and path, such as @app.get("/health"). Request bodies are declared as Pydantic models, which FastAPI validates automatically, returning a 422 error when the input does not match.

FastAPI generates interactive OpenAPI documentation at /docs. Dependencies are declared with Depends and can be overridden in tests through app.dependency_overrides, which is how you replace an expensive resource like a model or database with a fake.

Run a development server with `fastapi dev path/to/main.py`, which reloads on code changes. For production use `fastapi run`.
