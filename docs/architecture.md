# Package import direction

Dependencies inside `src/hermes_realtime` point inward. The rule has one statement:
the `TIERS` table in
[`tests/test_import_direction.py`](../tests/test_import_direction.py). A module may import
package code only from its own top-level unit or from a tier strictly below it. This page
does not repeat the table; read it there.

The same test measures the real import graph from source and compares it with the table.

- **What counts as an import.** Module-level imports, imports inside a function, imports
  under `if TYPE_CHECKING:`, relative imports, and `import_module` or `__import__` calls
  with a literal name. All are dependencies of the design. A function-level import defers a
  cycle without removing it, and an annotation-only import still binds the importer to the
  imported shape. Each listing records which of the three kinds it is.
- **Existing violations are frozen, not accepted.** `KNOWN_VIOLATIONS` in the same file
  names each importing file, imported module and kind, with the move that removes it.
- **The list only shrinks.** The test fails on a violation that is not listed and on a
  listing that no longer matches an import. Remove the listing in the change that removes
  the import.
- **Every unit is placed.** A new top-level package or module fails the test until a tier
  holds it. Placing a unit, or moving one between tiers, is a design decision and is
  reviewed as one.

Limits: the measurement is static. It does not see an `import_module` call whose name is
computed, and it says nothing about third-party imports. Lazy loading of heavy provider
dependencies is a separate property, held by
[`tests/providers/test_package_import.py`](../tests/providers/test_package_import.py).
