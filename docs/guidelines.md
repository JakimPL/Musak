# Coding Guidelines

## General

1. Keep code modularized around clear ownership boundaries.
2. If a function has several meaningful steps, split them into helpers with one clear responsibility.
3. Do not abbreviate variable names. Use `note`, not `n`.
4. Avoid hardcoded semantic values. Prefer `Final` constants, and move them to a shared module when the concept is reused.
5. Use `pathlib.Path` instead of `os.path`.
6. Prefer protocols over inheritance.
7. Separate function options with `*`. Positional arguments should be intentionally chosen.
8. Prefer `match` statements over long `isinstance` chains, and for enumeration handling.
9. Do not preserve backward compatibility for internal APIs, configs, or data shapes unless the user explicitly asks for it.
10. Avoid _tramp data_ antipattern.
11. Be cautious about optional parameters. All variables upon the logic relies on cannot be optional, including configuration instances.
12. Restrict yourself from using default values for non-optional, excluding these that are not meant to be frequently changed (e.g. seed). If you do use defaults, declare a Final top-level constant for that.
13. Be explicit about type expectations. Avoid dynamic `getattr` or `hasattr`.

## Shared Ownership

1. Musical constants and reusable musical logic belong in `musak_shared`, usually `musak_shared.elements`.
2. General-purpose helpers that are not model-specific belong in shared/common modules, not inside feature modules.
3. Do not create delegated imports or re-export modules just so other modules can import through them.
4. Import shared helpers directly from the module that owns their implementation.
5. Before adding a helper, search the repository for existing logic with `rg`.
6. If new code duplicates existing logic, extract the shared rule first and make both call sites use it.

## Type Hints

1. Specify all input and return types in function signatures, including `None`.
2. Fill generic types. Use `dict[str, int]`, not `dict`.
3. Do not cast/silence type errors unless the boundary is an untyped or mistyped third-party API.
4. Avoid `Any` and `object` unless the boundary genuinely accepts arbitrary data.
5. Do not quote type names. Use `from __future__ import annotations` (only if needed), `Self`, or `TYPE_CHECKING`.
6. Validate with `mypy`.

## Error Handling

1. Let failures crash unless the code can recover meaningfully.
2. Handle errors at the execution boundary when possible.
3. Do not add `try`/`except` blocks that only repackage failures without recovery.
4. Bare `except` and `except Exception` are forbidden.
5. Error handling blocks should cover only the code that is subject to a failure, unless there is a valid reason.

## Models

1. Prefer Pydantic models for validated or serialized data.
2. Use `frozen` when instances are not meant to change.
3. Dataclasses are acceptable for small internal state objects that are not serialized or validated, including test case dataclasses.

## Documentation

1. Avoid comments and docstrings that restate code.
2. Use clear names instead of explanatory comments.
3. Comments are acceptable for tensor shapes, third-party API quirks, or non-obvious invariants.

## Tests

1. Test files should mirror the ownership of the functionality under test.
2. Shared helpers in `musak_shared` should be tested under `tests/musak_shared`.
3. Model-specific behavior in `musak_model` should be tested under the matching `tests/musak_model` subpackage.
4. When moving functionality between packages, move its direct unit tests in the same change.
5. Parametrize test functions of the same body and use test case dataclass pattern.
6. Prefer fixtures over factories.
