# Licensed under the GPLv3 - see LICENSE
"""Full-package tests of the Phase class."""

import operator

import pytest
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle, Longitude
from astropy.time import Time

from ..phases import Phase, PhaseDelta


def assert_equal(phase, other):
    """Check matching type, matching phase1,2 and that phase1 is integer."""
    assert type(phase) is type(other)
    assert np.all(phase == other)
    if isinstance(phase, Phase):
        assert np.all(phase.phase1 % 1 == 0)


class BaseTests:
    def setup(self):
        self.phase1 = Angle(np.array([1000., 1001., 999., 1005, 1006.]),
                            u.cycle)[:, np.newaxis]
        self.phase2 = Angle(np.arange(0., 0.99, 0.25), u.cycle)
        self.phase = self.phase_cls(self.phase1, self.phase2)

    def test_basics(self):
        assert isinstance(self.phase, Phase)
        assert np.all(self.phase.phase1 % 1 == 0)
        cycle = (self.phase1 + self.phase2).to_value(u.cycle)
        count = cycle.round()
        fraction = cycle - count
        assert_equal(self.phase.phase1, count)
        assert_equal(self.phase.phase2, fraction)
        cycle = self.phase.cycle
        assert_equal(cycle, Angle(self.phase1 + self.phase2))
        phase = self.phase.phase
        assert_equal(phase, Longitude(self.phase2, wrap_angle=0.5*u.cy))

    @pytest.mark.parametrize('in1,in2', ((1.1111111, 0),
                                         (1.5, 0.111),
                                         (0.11111111, 1),
                                         (1.*u.deg, 0),
                                         (1.*u.cycle, 1.*u.deg)))
    def test_phase1_always_integer(self, in1, in2):
        phase = self.phase_cls(in1, in2)
        assert phase.phase1 % 1. == 0
        assert phase.cycle == u.Quantity(in1 + in2, u.cycle)

    def test_conversion(self):
        degrees = self.phase.to(u.degree)
        assert_equal(degrees, Angle(self.phase1 + self.phase2))

    def test_selection(self):
        phase2 = self.phase[0]
        assert phase2.shape == self.phase.shape[1:]
        assert_equal(phase2.cycle, self.phase.cycle[0])

    @pytest.mark.parametrize('op', (operator.eq, operator.ne,
                                    operator.le, operator.lt,
                                    operator.ge, operator.ge))
    def test_comparison(self, op):
        result = op(self.phase, self.phase[0])
        assert_equal(result, op(self.phase.cycle, self.phase[0].cycle))

    @pytest.mark.parametrize('op', (operator.eq, operator.ne,
                                    operator.le, operator.lt,
                                    operator.ge, operator.ge))
    def test_comparison_quantity(self, op):
        ref = 1005. * u.cy
        result = op(self.phase, ref.to(u.deg))
        assert_equal(result, op(self.phase.cycle, ref))

    def test_comparison_invalid_quantity(self):
        with pytest.raises(TypeError):
            self.phase > 1. * u.m

        with pytest.raises(TypeError):
            self.phase <= 1. * u.m

        assert (self.phase == 1. * u.m) is False
        assert (self.phase != 1. * u.m) is True

    def test_addition_quantity(self):
        phase = self.phase + 1. * u.cycle
        assert_equal(phase, self.phase_cls(self.phase1 + 1 * u.cycle,
                                           self.phase2))
        phase2 = 360. * u.deg + self.phase
        assert_equal(phase2, phase)

    def test_subtraction_quantity(self):
        phase = self.phase - 1. * u.cycle
        assert_equal(phase, self.phase_cls(self.phase1 - 1 * u.cycle,
                                           self.phase2))

    @pytest.mark.parametrize('axis', (None, 0, 1))
    def test_min(self, axis):
        m = self.phase.min(axis=axis)
        assert_equal(m, self.phase_cls(self.phase.cycle.min(axis=axis)))

    @pytest.mark.parametrize('axis', (None, 0, 1))
    def test_max(self, axis):
        m = self.phase.max(axis=axis)
        assert_equal(m, self.phase_cls(self.phase.cycle.max(axis=axis)))

    @pytest.mark.parametrize('axis', (None, 0, 1))
    def test_ptp(self, axis):
        ptp = self.phase.ptp(axis)
        assert_equal(ptp, PhaseDelta(self.phase.cycle.ptp(axis=axis)))

    @pytest.mark.parametrize('axis', (0, 1))
    def test_sort(self, axis):
        sort = self.phase.sort(axis=axis)
        comparison = self.phase.cycle.copy()
        comparison.sort(axis=axis)
        assert_equal(sort, self.phase_cls(comparison))


class TestPhase(BaseTests):
    phase_cls = Phase

    def test_substraction(self):
        delta = self.phase[1] - self.phase[0]
        assert_equal(delta, PhaseDelta(self.phase1[1] - self.phase1[0]))
        assert np.all(delta.phase2 == 0)

    def test_not_implemented(self):
        with pytest.raises(TypeError):
            -self.phase
        with pytest.raises(TypeError):
            self.phase + self.phase
        with pytest.raises(TypeError):
            1. * u.cycle - self.phase
        with pytest.raises(TypeError):
            self.phase * 1.
        with pytest.raises(TypeError):
            1. * self.phase
        with pytest.raises(TypeError):
            self.phase / 1.
        with pytest.raises(TypeError):
            -self.phase


class TestPhaseDelta(BaseTests):
    phase_cls = PhaseDelta

    def setup(self):
        super().setup()
        self.delta = PhaseDelta(0., self.phase2)
        self.full = Phase(self.phase1, self.phase2)

    def test_negation(self):
        neg = -self.phase
        assert_equal(neg, PhaseDelta(-self.phase1, -self.phase2))

    def test_unitless_multiplication(self):
        mul = self.delta * 2
        assert_equal(mul, PhaseDelta(self.delta.phase1 * 2,
                                     self.delta.phase2 * 2))
        mul2 = self.delta * (2. * u.dimensionless_unscaled)
        assert_equal(mul2, mul)
        mul3 = self.delta * 2. * u.one
        assert_equal(mul3, mul)
        mul4 = 2. * self.delta
        assert_equal(mul4, mul)
        mul5 = self.delta * np.full(self.delta.shape, 2.)
        assert_equal(mul5, mul)

    def test_unitless_division(self):
        div = self.delta / 0.5
        assert_equal(div, PhaseDelta(self.delta.phase1 * 2,
                                     self.delta.phase2 * 2))
        div2 = self.delta / (0.5 * u.dimensionless_unscaled)
        assert_equal(div2, div)
        div3 = self.delta / 0.5 / u.one
        assert_equal(div3, div)
        div4 = self.delta / np.full(self.delta.shape, 0.5)
        assert_equal(div4, div)

    def test_unitfull_multiplication(self):
        mul = self.delta * (2 * u.Hz)
        assert_equal(mul, u.Quantity(self.delta.cycle * 2 * u.Hz))
        mul2 = self.delta * 2. * u.Hz
        assert_equal(mul2, mul)
        mul3 = 2. * u.Hz * self.delta
        assert_equal(mul3, mul)

    def test_unitfull_division(self):
        delta = self.delta[self.delta != PhaseDelta(0, 0)]
        div = delta / (0.5 * u.s)
        assert_equal(div, u.Quantity(delta.cycle * 2 / u.s))
        div2 = delta / 0.5 / u.s
        assert_equal(div2, div)
        div3 = 0.5 * u.s / delta
        assert_equal(div3, 1. / div)

    def test_addition_deltas(self):
        add = self.delta + self.delta
        assert_equal(add, PhaseDelta(2. * self.delta.phase1,
                                     2. * self.delta.phase2))
        add2 = self.delta.to(u.cycle) + self.delta
        assert_equal(add2, add)
        add3 = self.delta + self.delta.to(u.degree)
        assert_equal(add3, add)

    def test_addition_full(self):
        add = self.full + self.delta
        assert_equal(add, Phase(self.phase1, 2 * self.phase2))
        add2 = self.delta.to(u.deg) + self.full
        assert_equal(add2, add)

    def test_subtraction_deltas(self):
        half = PhaseDelta(self.delta.phase1 / 2., self.delta.phase2 / 2.)
        sub = half - self.delta
        assert_equal(sub, PhaseDelta(-self.delta.phase1 / 2.,
                                     -self.delta.phase2 / 2.))
        sub2 = self.delta.to(u.cycle) * 0.5 - self.delta
        assert_equal(sub2, sub)
        sub3 = half - self.delta.to(u.degree)
        assert_equal(sub3, sub)
