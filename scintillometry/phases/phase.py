# -*- coding: utf-8 -*-
"""Provide a Phase class with integer and fractional part."""
import operator

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, Longitude
from astropy.time import Time
from astropy.utils import ShapedLikeNDArray
from astropy.time.utils import day_frac


__all__ = ['Phase', 'PhaseDelta']


class Phase(ShapedLikeNDArray):
    """Represent two-part phase.

    The phase is absolute and hence has more limited operations available to it
    than a relative phase (e.g., it cannot be multiplied).  This is analogous
    to the difference between an absolute time and a time difference.

    Parameters
    ----------
    phase1, phase2 : array or `~astropy.units.Quantity`
        Two-part phase.  If arrays, the assumed units are cycles.
    copy : bool, optional
        Make a copy of the input values

    """

    # Make sure that reverse arithmetic (e.g., PhaseDelta.__rmul__)
    # gets called over the __mul__ of Numpy arrays.
    __array_priority__ = 20000

    def __init__(self, phase1, phase2=0, copy=False):
        phase1 = u.Quantity(phase1, u.cycle, copy=copy).value
        phase2 = u.Quantity(phase2, u.cycle, copy=copy).value
        self._phase1, self._phase2 = day_frac(phase1, phase2)

    @classmethod
    def from_arrays(cls, phase1, phase2):
        self = super().__new__(cls)
        self._phase1 = phase1
        self._phase2 = phase2
        return self

    def __repr__(self):
        return "{0}({1}, {2})".format(self.__class__.__name__,
                                      self._phase1, self._phase2)

    def __str__(self):
        return str(self.value)

    @property
    def shape(self):
        """The shape of the phase instances.

        Like `~numpy.ndarray.shape`, can be set to a new shape by assigning a
        tuple.  Note that if different instances share some but not all
        underlying data, setting the shape of one instance can make the other
        instance unusable.  Hence, it is strongly recommended to get new,
        reshaped instances with the ``reshape`` method.

        Raises
        ------
        AttributeError
            If the shape of the ``phase1`` or ``phase2`` attributes cannot be
            changed without the arrays being copied.
        """
        return self._phase1.shape

    @shape.setter
    def shape(self, shape):
        old_shape = self.shape
        self._phase1.shape = shape
        try:
            self._phase2.shape = shape
        except AttributeError:
            self._phase1.shape = old_shape
            raise

    @property
    def phase1(self):
        """First of two doubles that store the phase: integer cycles."""
        return self._phase1

    @property
    def phase2(self):
        """Second of two doubles that store the phase: fractional cycles."""
        return self._phase2

    @property
    def value(self):
        """Phase value."""
        return self._phase1 + self._phase2

    @property
    def phase(self):
        """Fractional phase."""
        return Longitude(self.phase2, u.cycle, copy=False, wrap_angle=0.5*u.cy)

    @property
    def cycle(self):
        """Full cycle, including phase."""
        return Angle(self.value, u.cycle, copy=False)

    def to(self, *args, **kwargs):
        return self.cycle.to(*args, **kwargs)

    def _apply(self, method, *args, format=None, **kwargs):
        """Create a new phase object, possibly applying a method to the arrays.

        Parameters
        ----------
        method : str or callable
            If string, the name of a `~numpy.ndarray` method. Either is applied
            to the internal ``phase1`` and ``phase2`` arrays.
            If a callable, it is directly applied to the above arrays.
            Examples: 'copy', '__getitem__', 'reshape', `~numpy.broadcast_to`.
        args : tuple
            Any positional arguments for ``method``.
        kwargs : dict
            Any keyword arguments for ``method``.

        Examples
        --------
        Some ways this is used internally::

            copy : ``_apply('copy')``
            reshape : ``_apply('reshape', new_shape)``
            index or slice : ``_apply('__getitem__', item)``
            broadcast : ``_apply(np.broadcast, shape=new_shape)``
        """
        if callable(method):
            phase1 = method(self._phase1, *args, **kwargs)
            phase2 = method(self._phase2, *args, **kwargs)

        else:
            phase1 = getattr(self._phase1, method)(*args, **kwargs)
            phase2 = getattr(self._phase2, method)(*args, **kwargs)

        return self.from_arrays(phase1, phase2)

    def _advanced_index(self, indices, axis=None, keepdims=False):
        ai = Time._advanced_index(self, indices, axis=axis, keepdims=keepdims)
        return tuple(ai)

    def argmin(self, axis=None, out=None):
        """Return indices of the minimum values along the given axis."""
        phase = self.phase1 + self.phase2
        approx = np.min(phase, axis, keepdims=True)
        dt = (self.phase1 - approx) + self.phase2
        return dt.argmin(axis, out)

    def argmax(self, axis=None, out=None):
        """Return indices of the maximum values along the given axis."""
        phase = self.phase1 + self.phase2
        approx = np.max(phase, axis, keepdims=True)
        dt = (self.phase1 - approx) + self.phase2
        return dt.argmax(axis, out)

    def argsort(self, axis=-1):
        """Returns the indices that would sort the phase array."""
        phase_approx = self.value
        phase_remainder = (self - self.__class__(phase_approx)).value
        if axis is None:
            return np.lexsort((phase_remainder.ravel(), phase_approx.ravel()))
        else:
            return np.lexsort(keys=(phase_remainder, phase_approx), axis=axis)

    min = Time.min
    max = Time.max
    ptp = Time.ptp
    sort = Time.sort

    def __add__(self, other):
        if not isinstance(other, PhaseDelta):
            try:
                other = PhaseDelta(other)
            except Exception:
                return NotImplemented

        return self.from_arrays(*day_frac(self._phase1 + other._phase1,
                                          self._phase2 + other._phase2))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, PhaseDelta):
            if isinstance(other, Phase):
                return PhaseDelta.from_arrays(
                    *day_frac(self.phase1 - other.phase1,
                              self.phase2 - other.phase2))
            try:
                other = PhaseDelta(other)
            except Exception:
                return NotImplemented

        return self.from_arrays(*day_frac(self.phase1 - other.phase1,
                                          self.phase2 - other.phase2))

    def _phase_comparison(self, other, op):
        if not isinstance(other, self.__class__):
            try:
                other = self.__class__(other)
            except Exception:
                return NotImplemented

        return op((self.phase1 - other.phase1) +
                  (self.phase2 - other.phase2), 0.)

    def __lt__(self, other):
        return self._phase_comparison(other, operator.lt)

    def __le__(self, other):
        return self._phase_comparison(other, operator.le)

    def __eq__(self, other):
        return self._phase_comparison(other, operator.eq)

    def __ne__(self, other):
        return self._phase_comparison(other, operator.ne)

    def __gt__(self, other):
        return self._phase_comparison(other, operator.gt)

    def __ge__(self, other):
        return self._phase_comparison(other, operator.ge)


class PhaseDelta(Phase):
    """Represent two-part phase difference.

    Parameters
    ----------
    phase1, phase2 : array or `~astropy.units.Quantity`
        Two-part phase difference.  If arrays, the assumed units are cycles.
    copy : bool, optional
        Make a copy of the input values
    """
    def __rsub__(self, other):
        if isinstance(other, Phase):
            return NotImplemented

        out = self.__sub__(other)
        return -out if out is not NotImplemented else out

    def __neg__(self):
        return self.from_arrays(-self.phase1, -self.phase2)

    def __abs__(self):
        return self._apply(np.copysign, self.value)

    def __mul__(self, other):
        # Check needed since otherwise the self.jd1 * other multiplication
        # would enter here again (via __rmul__)
        if (not isinstance(other, PhaseDelta) and
            ((isinstance(other, u.UnitBase) and
              other == u.dimensionless_unscaled) or
             (isinstance(other, str) and other == ''))):
            return self.copy()

        # If other is something consistent with a dimensionless quantity
        # (could just be a float or an array), then we can just multiple in.
        try:
            other = u.Quantity(other, u.dimensionless_unscaled, copy=False)
        except Exception:
            # If not consistent with a dimensionless quantity, try downgrading
            # self to a quantity and see if things work.
            try:
                return self.to(u.cycle) * other
            except Exception:
                # The various ways we could multiply all failed;
                # returning NotImplemented to give other a final chance.
                return NotImplemented

        return self.from_arrays(*day_frac(self._phase1, self._phase2,
                                          factor=other.value))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # Cannot do __mul__(1./other) as that looses precision
        if ((isinstance(other, u.UnitBase) and
             other == u.dimensionless_unscaled) or
                (isinstance(other, str) and other == '')):
            return self.copy()

        # If other is something consistent with a dimensionless quantity
        # (could just be a float or an array), then we can just divide in.
        try:
            other = u.Quantity(other, u.dimensionless_unscaled, copy=False)
        except Exception:
            # If not consistent with a dimensionless quantity, try downgrading
            # self to a quantity and see if things work.
            try:
                return self.to(u.cycle) / other
            except Exception:
                # The various ways we could divide all failed;
                # returning NotImplemented to give other a final chance.
                return NotImplemented

        return self.from_arrays(*day_frac(self._phase1, self._phase2,
                                          divisor=other.value))

    def __rtruediv__(self, other):
        # Here, we do not have to worry about returning NotImplemented,
        # since other has already had a chance to look at us.
        return other / self.to(u.cycle)
