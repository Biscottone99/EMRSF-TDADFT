"""Minimal external SOC module for ``--soc-external-module``.

Replace either Cartesian MO integral tensor before contracting it to test a
different relativistic Hamiltonian.  The callback must return matrices in
Hartree in the state-interaction basis described by ``context.basis_states``.
"""


def compute_soc(context):
    """Reproduce the built-in 1e + SOMF contraction through the public hook."""

    one = context.contract_one_body(context.integrals.one_electron_mo)
    two = context.contract_one_body(context.integrals.two_electron_mo)
    return {
        "one_electron_matrix_hartree": one,
        "two_electron_matrix_hartree": two,
        "approximation": "external example: Breit-Pauli 1e + selected SOMF",
    }
