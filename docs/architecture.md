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

A dynamic call with a literal name whose target still depends on a runtime value fails the
test instead of being guessed: a relative `import_module` name without a literal `package`,
an `__import__` with a `level`, a non-literal `fromlist` on this package, or unpacked
arguments.

## Composition roots

The top tier wires the outside together and is meant to do nothing else. The rule for
what a composition root may contain has one statement, at the top of
[`tests/test_composition_root.py`](../tests/test_composition_root.py); this page does not
repeat it. The same test counts what the two builders hold beyond construction (nested
functions and classes, lambdas, `nonlocal` declarations, statements inside closures, and
`await` and `try` in the builder itself) and binds each count to a recorded number. A
count that grows fails; a count that falls fails until the number is lowered in the same
change. The plan that takes the counts to zero is
[#525](https://github.com/sushiHex/hermes-realtime/issues/525).

Limits: the measurement is static. It does not see a dynamic import whose name is computed
or whose function is aliased, and it says nothing about third-party imports. Lazy loading of heavy provider
dependencies is a separate property, held by
[`tests/providers/test_package_import.py`](../tests/providers/test_package_import.py).
