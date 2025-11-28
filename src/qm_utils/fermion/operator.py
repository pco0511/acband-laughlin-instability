import copy
from collections import defaultdict
from functools import partial
from numbers import Number
import numpy as np

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from netket.jax import canonicalize_dtypes

OperatorTuple = tuple[int, int]
r""" Creation and annihilation operators at mode i are encoded as
:math:`\hat{a}_i^\dagger`: (i, 1)
:math:`\hat{a}`: (i, 0)
"""

OperatorTerm = tuple[OperatorTuple, ...]
r""" A term of the form :math:`\hat{a}_1^\dagger \hat{a}_2` would take the form
`((1,1), (2,0))`, where (1,1) represents :math:`\hat{a}_1^\dagger` and (2,0)
represents :math:`\hat{a}_2`."""

OperatorTermsList = list[OperatorTerm]
""" A list of operators that would e.g. describe a Hamiltonian """

OperatorWeightsList = list[Number]
""" A list of weights of corresponding terms """

OperatorDict = dict[OperatorTerm, Number]
""" A dict containing OperatorTerm as key and weights as the values """


def _parse_term_tree(terms):
    """Convert the terms tree into a canonical form of tuple tree of depth 3"""

    def _parse_branch(t):
        if hasattr(t, "__len__"):
            return tuple([_parse_branch(b) for b in t])
        else:
            return int(t)

    return _parse_branch(terms)

def _check_tree_structure(terms):
    """
    Check whether the terms structure is depth 3 everywhere
    and contains pairs of (idx, dagger) everywhere
    """

    def _descend(tree, current_depth, depths, pairs):
        if current_depth == 2 and hasattr(tree, "__len__"):
            pairs.append(len(tree) == 2)
        if hasattr(tree, "__len__"):
            for branch in tree:
                _descend(branch, current_depth + 1, depths, pairs)
        else:
            depths.append(current_depth)

    depths = []
    pairs = []
    _descend(terms, 0, depths, pairs)
    if not np.all(np.array(depths) == 3):
        raise ValueError(f"terms is not a depth 3 tree, found depths {depths}")
    if not np.all(pairs):
        raise ValueError(
            "terms should be provided in (i, dag) pairs or empty for a constant"
        )

def zero_defaultdict(dtype):
    def _dtype_init():
        return np.array(0, dtype=dtype).item()

    return defaultdict(_dtype_init)

def _remove_dict_zeros(d: dict, epsilon: float):
    return {k: v for k, v in d.items() if np.abs(v) > epsilon}

def _canonicalize_input(terms, weights, dtype, epsilon, constant):
    if terms is None:
        terms = []
    if len(terms) > 0:
        terms = _parse_term_tree(terms)
    if weights is None:
        weights = [1.0] * len(terms)
    
    weights = list(weights)

    if not np.isclose(constant, 0.0, atol=epsilon):
        terms = [()] + list(terms)
        weights = [constant] + weights
    
    dtype = canonicalize_dtypes(float, *weights, constant, dtype=dtype)

    weights = list(weights)

    if not len(weights) == len(terms):
        raise ValueError(
            f"Number of weights ({len(weights)}) must match number of terms ({len(terms)})"
        )
    
    _check_tree_structure(terms)

    operators = zero_defaultdict(dtype)
    for t, w in zip(terms, weights):
        operators[t] += w
    operators = _remove_dict_zeros(operators, epsilon)

    return operators, dtype

def _verify_input(n_modes, operators, raise_error=True) -> bool:
    """Check whether all input is valid"""
    terms = list(operators.keys())

    def _check_op(fop):
        v1 = 0 <= fop[0] < n_modes
        if not v1:
            if raise_error:
                raise ValueError(
                    f"Found invalid mode index {fop[0]} for hilbert space with {n_modes} modes."
                )
            return False
        v2 = fop[1] in (0, 1)
        if not v2:
            if raise_error:
                raise ValueError(
                    f"Found invalid character {fop[1]} for dagger, which should be 0 (no dagger) or 1 (dagger)."
                )
            return False
        return True

    def _check_term(term):
        return all(_check_op(t) for t in term)

    return all(_check_term(term) for term in terms)

def _normal_order_term(
    term: OperatorTerm, weight: Number = 1.0
) -> tuple[OperatorTermsList, OperatorWeightsList]:
    """
    Return a normal ordered single term of the fermion operator.
    Normal ordering corresponds to placing creating operators on the left
    and annihilation on the right.
    Then, it places the highest index on the left and lowest index on the right
    In this ordering, we make sure to account for the anti-commutation of operators.
    """

    parity = -1
    term = copy.deepcopy(list(term))
    weight = copy.copy(weight)

    if len(term) == 0:  # a constant
        return [term], [weight]
    ordered_terms = []
    ordered_weights = []
    # the arguments given to this function will be transformed in a normal ordered way
    # loop through all the operators in the single term from left to right and order them
    # by swapping the term operators (and transform the weights by multiplying with the parity factors)
    for i in range(1, len(term)):
        for j in range(i, 0, -1):
            right_term = term[j]
            left_term = term[j - 1]

            # exchange operators if creation operator is on the right and annihilation on the left
            if right_term[1] and not left_term[1]:
                term[j - 1] = right_term
                term[j] = left_term
                weight *= parity

                # if same indices switch order (creation on the left), remember a a^ = 1 + a^ a
                if right_term[0] == left_term[0]:
                    new_term = term[: (j - 1)] + term[(j + 1) :]

                    # ad the processed term
                    o, w = _normal_order_term(tuple(new_term), parity * weight)
                    ordered_terms += o
                    ordered_weights += w

            # if we have two creation or two annihilation operators
            elif right_term[1] == left_term[1]:
                # If same two Fermionic operators are repeated,
                # evaluate to zero.
                if parity == -1 and right_term[0] == left_term[0]:
                    return ordered_terms, ordered_weights

                # swap if same type but order is not correct
                elif right_term[0] > left_term[0]:
                    term[j - 1] = right_term
                    term[j] = left_term
                    weight *= parity

    ordered_terms.append(term)
    ordered_weights.append(weight)
    return ordered_terms, ordered_weights

def _make_tuple_tree(terms: PyTree) -> PyTree:
    """Make tuples, so terms are hashable.

    Input could be e.g. a pytree of lists of lists,
    which we convert to tuples of tuples.
    """

    def _make_tuple(branch):
        if hasattr(branch, "__len__"):
            return tuple([_make_tuple(t) for t in branch])
        else:
            return int(branch)

    return _make_tuple(terms)

def _normal_ordering(
    terms: OperatorTermsList, weights: OperatorWeightsList = 1.0
) -> tuple[OperatorTermsList, OperatorWeightsList]:
    """
    Returns the normal ordered terms and weights of the fermion operator.
    """
    ordered_terms = []
    ordered_weights = []
    # loop over all the terms and weights and order each single term with corresponding weight
    for term, weight in zip(terms, weights):
        ordered = _normal_order_term(term, weight)
        ordered_terms += ordered[0]
        ordered_weights += ordered[1]
    ordered_terms = _make_tuple_tree(ordered_terms)
    return ordered_terms, ordered_weights

def _is_diag_term(term):
    def _init_empty_arr():
        return [
            0,
            0,
        ]
    if len(term) == 0:
        return True  # constant

    ops = defaultdict(_init_empty_arr)
    for mode_idx, dagger in term:
        ops[mode_idx][int(dagger)] += 1
    return all((x[0] == x[1]) for x in ops.values())

@partial(jax.vmap, in_axes=(0, None, None))
def _flip_daggers_split_cast_term_part(term, site_dtype, dagger_dtype):
    # splits sites and daggers out of terms, casts to desired dtype
    sites, daggers = jnp.array(term).reshape([-1, 2]).T
    # we flip the daggers so that the operator returns xp s.t. <xp|O|x> != 0
    daggers = jnp.ones_like(daggers) - daggers
    return sites.astype(site_dtype), daggers.astype(dagger_dtype)

def prepare_terms_list(
    operators,
    site_dtype=np.uint32,
    dagger_dtype=np.int8,
    weight_dtype=jnp.float64,
):
    # return xp s.t. <x|O|xp> != 0
    # see https://github.com/netket/netket/issues/1385
    term_dagger_split_fn = _flip_daggers_split_cast_term_part

    # group the terms together with respect to the number of sites they act on
    terms_dicts = {}
    for t, w in operators.items():
        l = len(t)
        d = terms_dicts.get(l, {})
        d[t] = w
        terms_dicts[l] = d

    res = []
    for d in terms_dicts.values():
        w = jnp.array(list(d.values()), dtype=weight_dtype)
        t = np.array(list(d.keys()), dtype=int)
        res.append((w, *term_dagger_split_fn(t, site_dtype, dagger_dtype)))
    return res

@register_pytree_node_class
class FermionOperator:

    def __init__(
        self,
        terms=None,
        weights=None,
        constant=None,
        epsilon=1e-10,
        dtype=None,
    ):
        self._epsilon = epsilon
        _operators, dtype = _canonicalize_input(
            terms, weights, dtype, epsilon, constant
        )
        self._dtype = dtype

        self._operators = _operators
        self._initialized = False
        self._is_hermitian = None  # set when requested
        self._max_conn_size = None
        self._order = None
    
    def _reset_caches(self):
        """
        Cleans the internal caches built on the operator.
        """
        self._initialized = False
        self._is_hermitian = None
        self._max_conn_size = None

    def __repr__(self):
        rep_str = (
            f"{type(self).__name__}("
            f"n_operators={len(self._operators)}, dtype={self.dtype}"
        )
        if self.order:
            rep_str = rep_str + f", order={self.order}"
        return rep_str + ")"

    def _setup(self, force: bool=False):
        if force or not self._initialized:
            diag_operators = {
                k: v for k, v in self._operators.items() if _is_diag_term(k)
            }
            offdiag_operators = {
                k: v for k, v in self._operators.items() if not _is_diag_term(k)
            }
            self._terms_list_diag = prepare_terms_list(
                diag_operators,
                site_dtype=np.uint32,
                dagger_dtype=jnp.bool_,
                weight_dtype=self._dtype,
            )
            self._terms_list_offdiag = prepare_terms_list(
                offdiag_operators,
                site_dtype=np.uint32,
                dagger_dtype=jnp.bool_,
                weight_dtype=self._dtype,
            )
            self._max_conn_size = int(len(self._terms_list_diag) > 0) + len(
                offdiag_operators
            )
            self._initialized = True

    def reduce(self, order: bool = True, inplace: bool = True, epsilon: float = None):
        """
        Prunes the operator by removing all terms with zero weights, grouping, and normal ordering (inplace).

        Args:
            order: Whether to normal order the operator.
            inplace: Whether to change the current object in place.
            epsilon: Optional cutoff for the weights.
        """
        if epsilon is None:
            epsilon = self._epsilon

        operators = self._operators
        terms, weights = list(operators.keys()), list(operators.values())

        if order:
            terms, weights = _normal_ordering(terms, weights)

        obj = self if inplace else self.copy()
        obj._operators, _ = _canonicalize_input(terms, weights, self.dtype, epsilon)

        if order:
            obj._order = "N"
        return obj
    
    @property
    def dtype(self):
        return self._dtype
    
    @property
    def terms(self) -> OperatorTermsList:
        return list(self._operators.keys())
    
    @property
    def weights(self) -> OperatorWeightsList:
        return list(self._operators.values())
    
    @property
    def operators(self) -> OperatorDict:
        return self._operators
    
    @property
    def epsilon(self) -> float:
        return self._epsilon
    
    @property
    def order(self) -> str | None:
        return self._order


    def copy(self, *, dtype=None, epsilon=None) -> "FermionOperator":
        """
        Returns a copy of the FermionOperator.

        Args:
            dtype: Optional dtype for the new operator.
            epsilon: Optional epsilon for the new operator.
        """
        if dtype is None:
            dtype = self.dtype
        if not np.can_cast(self.dtype, dtype, casting="same_kind"):
            raise ValueError(f"Cannot cast {self.dtype} to {dtype}")
        if epsilon is None:
            epsilon = self.epsilon

        op = type(self)(epsilon=epsilon, dtype=dtype)

        if dtype == self.dtype:
            operators_new = self._operators.copy()
        else:
            operators_new = {
                k: np.array(v, dtype=dtype) for k, v in self._operators.items()
            }

        op._operators = operators_new
        op._order = self.order
        return op
    
    def _remove_zeros(self, epsilon: float=None) -> None:
        if epsilon is None:
            epsilon = self._epsilon
        op = type(self)(epsilon=epsilon, dtype=self.dtype)
        op._operators = _remove_dict_zeros(self._operators, epsilon)
        return op
    
    @property
    def max_conn_size(self) -> int:
        if self._max_conn_size is None:
            self._setup()
