# Coding Standards

General development guidelines for Python/FastAPI projects.

## Pre-Commit Requirements

**Zero tolerance for preventable issues:**

1. Check IDE Problems tab - eliminate ALL errors and warnings before committing
2. Run formatters: `black .` and `ruff check --fix .`
3. Validate file properties (correct extensions, UTF-8 encoding)
4. Execute all pre-commit hooks before pushing

**Never create files that generate IDE problems or linter errors.**

## Python Standards

### Modern Syntax

- Use union operator `|` instead of `Optional` from typing module
- Maintain strict type annotations throughout
- All code must pass Black, Ruff, and mypy validation

### Error Handling

- **Never use string matching on exception messages for flow control**
- Create specific exception classes instead of parsing error strings
- Use explicit exception types, not generic `ValueError` or `Exception`
- Better to crash with clear error than continue with undefined behavior
- Avoid conditional logic based on exception message content

### Code Quality Principles

- **DRY enforcement**: Never repeat code
- **No optional fallbacks**: Never use `hasattr`, fallbacks, or default values - use assertions and error handling
- **Deterministic design**: Methods either work or fail clearly, no graceful degradation
- **Extend existing code**: Work within established patterns, don't create new classes without approval

## Codebase Analysis Before Implementation

**Mandatory before any refactoring or new features:**

1. Search for existing implementations of similar functionality
2. Examine related files and understand current patterns
3. Document what exists vs what's missing
4. Plan minimal additions that integrate with existing code

**Red Flags:**

- Creating files with names similar to existing ones
- Recreating functionality that already exists
- Writing code that doesn't follow established patterns

## Git Commit Practices

**Critical rules:**

- **Never commit without explicit user approval**
- Always display proposed changes and obtain authorization
- **Never include AI attribution** in commit messages ("Claude," "AI-generated," etc.)
- Write clear, professional messages describing what changed and why
- Maintain clean history with meaningful commits

## Testing Philosophy

Focus on **behavior verification**, not implementation details:

- Test what the system does, not how it does it internally
- When algorithms change, behavior-based tests should remain valid
- If tests break during refactoring with equivalent algorithms, they tested implementation incorrectly
- Categories: business logic, constraint validation, integration testing

**Red Flags:**

- Tests that verify specific internal structure
- Tests that check exact boundaries or algorithm-specific counts
- Implementation-focused rather than behavior-focused tests

## Markdown Standards

Maintain consistent structure to pass markdownlint:

### Lists (MD032)

**Rule: Always add a blank line before AND after lists**

**Example 1: Text ending with colon**

```markdown
❌ WRONG:
Some text with colon:
- Item 1
- Item 2

✅ CORRECT:
Some text with colon:

- Item 1
- Item 2

Next paragraph
```

**Example 2: Bold text with colon**

```markdown
❌ WRONG:
**Feature Name:**
- Sub-item 1
- Sub-item 2

✅ CORRECT:
**Feature Name:**

- Sub-item 1
- Sub-item 2
```

**Example 3: Numbered list with sub-items**

```markdown
❌ WRONG:
1. **Main Item:**
   - Sub-item 1
   - Sub-item 2

✅ CORRECT:
1. **Main Item:**

   - Sub-item 1
   - Sub-item 2
```

**Example 4: Headers followed by lists**

```markdown
❌ WRONG:
### Section Name
- Item 1
- Item 2

✅ CORRECT:
### Section Name

- Item 1
- Item 2
```

**Do NOT add blank lines between list items:**

```markdown
❌ WRONG:
- Item 1

- Item 2

✅ CORRECT:
- Item 1
- Item 2
```

### Headers

- Add blank lines before and after all headers
- Never skip heading levels (no jumping from h1 to h3)

### Other Rules

- Remove all trailing whitespace
- Never use multiple consecutive blank lines
- Use consistent indentation (spaces, not tabs)

## Code Preservation

- Never remove or modify existing functionality without explicit direction
- Produce clean code without referencing older versions
- Ask for clarification rather than making assumptions
- Think about current software design before adding functionality

## Code Style (Project-Specific)

See `pyproject.toml` for complete configuration:

- **Formatting**: Black (line length: 88)
- **Linting**: Ruff with pycodestyle, flake8-bugbear, pyupgrade rules
- **Type checking**: mypy strict mode
- **Import order**: isort with known-first-party configuration
