# -*- coding: utf-8 -*-
"""Comutations about two-body Orbits (v1.0.0)
This module provide computations about two-body orbits, including:
  Define the orbit by position and velocity of an object
  Define the orbit by classical orbital elements of an object
  Compute position and velocity of an object at given time
  Provide seriese of points on orbital trajectory for visualization
  Solve Lambert's problem  (From given two positions and flight time 
  between them, lambert() computes initial and terminal velocity of 
  the object)
@author: Shushi Uetsuki/whiskie14142
"""

# Issued Under the 'MIT License' 8/1/2019

import math
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import bisect, newton


class TwoBodyOrbit:
    """A class of a two-body orbit of a celestial object
    
    """

    def timeFperi(self, ta):
        """Computes time from periapsis passage for given true anomaly
        
        Args:
            ta: True Anomaly in radians
        Returns: sec_from_peri
            sec_from_peri: Time from periapsis passage (float). Unit of time
                           depends on gravitational parameter (mu)
        """
        if not self._setOrb:
            raise (
                RuntimeError(
                    "Orbit has not been defined: in TwoBodyOrbit.timeFperi"
                )
            )

        r = self.a * (1.0 - self.e ** 2) / (1.0 + self.e * np.cos(ta))
        if self.e < 1.0:
            b_over_a = np.sqrt(1.0 - self.e ** 2)
            ecc_anm = np.arctan2(
                r * np.sin(ta) / b_over_a, self.a * self.e + r * np.cos(ta)
            )
            if ecc_anm < 0.0:
                ecc_anm += math.pi * 2.0
            sec_from_peri = np.sqrt(self.a ** 3 / self.mu) * (
                ecc_anm - self.e * np.sin(ecc_anm)
            )
        elif self.e == 1.0:
            ecc_anm = np.sqrt(self.p) * np.tan(ta / 2.0)
            sec_from_peri = (
                (self.p * ecc_anm + ecc_anm ** 3 / 3.0)
                / 2.0
                / np.sqrt(self.mu)
            )
        else:
            sy = (self.e + np.cos(ta)) / (1.0 + self.e * np.cos(ta))
            lf = np.log(sy + np.sqrt(sy ** 2 - 1.0))
            if (ta < 0.0) or (ta > math.pi):
                lf = lf * (-1.0)
            sec_from_peri = np.sqrt((-1.0) * self.a ** 3 / self.mu) * (
                self.e * np.sinh(lf) - lf
            )
        return sec_from_peri

    def posvel(self, ta):
        """Comuputs position and velocity for given true anomaly
        
        Args:
            ta: True Anomaly in radians
        Returns: rv, vv
            rv: Position (x,y,z) as numpy array
            vv: Velocity (xd,yd,zd) as numpy array
                Units are depend on gravitational parameter (mu)
        """
        if not self._setOrb:
            raise (
                RuntimeError(
                    "Orbit has not been defined: in TwoBodyOrbit.posvel"
                )
            )

        PV = self.evd
        QV = np.cross(self.hv, PV) / np.sqrt(np.dot(self.hv, self.hv))
        r = self.p / (1.0 + self.e * np.cos(ta))
        rv = r * np.cos(ta) * PV + r * np.sin(ta) * QV
        vv = np.sqrt(self.mu / self.p) * (
            (-1.0) * np.sin(ta) * PV + (self.e + np.cos(ta)) * QV
        )
        return rv, vv

    def __init__(self, bname, mname="Sun", mu=1.32712440041e20):
        """
        Args:
            bname: Name of the object which orbit around the central body
            mname: Name of the central body
            mu : Gravitational parameter (mu) of the central body
                Default value is gravitational parameter of the Sun.  
                
                mu should be:
                if Mc >> Mo
                    mu = G*Mc
                else
                    mu = G*(Mc + Mo)

                where:
                    G: Newton's gravitational constant
                    Mc: mass of the central body
                    Mo: mass of the object
        """
        self._setOrb = False
        self.bodyname = bname
        self.mothername = mname
        self.mu = mu

    def setOrbCart(self, t, pos, vel):
        """Define the orbit by epoch, position, and velocity of the object
        
        Args:
            t: Epoch
            pos: Position (x,y,z), array-like object
            vel: Velocity (xd,yd,zd), array-like object
                Units are depend on gravitational parameter (mu)
                Origin of coordinates are the central body
        
        Exceptions:
            ValueError: when angular momentum is zero, the method raises
                ValueError
                        when e becomes 1.0, the method raises ValueError
        """
        self.t0 = t
        self.pos = np.array(pos)
        self.vel = np.array(vel)
        self._setOrb = True

        # Computes Classical orbital elements
        r0 = np.array(self.pos)
        r0len = np.sqrt(np.dot(r0, r0))

        rd0 = np.array(self.vel)
        rd0len2 = np.dot(rd0, rd0)

        h = np.cross(r0, rd0)
        hlen2 = np.dot(h, h)
        hlen = np.sqrt(hlen2)
        if hlen == 0.0:
            self._setOrb = False
            raise (
                ValueError(
                    "Inappropriate pos and vel in TwoBodyOrbit.setOrbCart"
                )
            )

        # eccentricity vector; it can be zero
        ev = (
            (rd0len2 - self.mu / r0len) * r0 - np.dot(r0, rd0) * rd0
        ) / self.mu
        evlen = np.sqrt(np.dot(ev, ev))  # evlen can be zero (circular orbit)

        K = np.array([0.0, 0.0, 1.0])
        n = np.cross(K, h)  # direction of the ascending node
        nlen = np.sqrt(
            np.dot(n, n)
        )  # nlen can be zero (orbital inclination is zero)

        if evlen == 0.0:
            if nlen == 0.0:
                ev_norm = np.array([1.0, 0.0, 0.0])
            else:
                ev_norm = n / nlen
        else:
            ev_norm = ev / evlen

        he = np.cross(h, ev)
        he_norm = he / np.sqrt(np.dot(he, he))

        if nlen == 0.0:
            self.lan = 0.0
            self.parg = np.arctan2(ev[1], ev[0])
            if self.parg < 0.0:
                self.parg += math.pi * 2.0
        else:
            n_norm = n / nlen
            hn = np.cross(h, n)
            hn_norm = hn / np.sqrt(np.dot(hn, hn))
            self.lan = np.arctan2(
                n[1], n[0]
            )  # longitude of ascending node (radians)
            if self.lan < 0.0:
                self.lan += math.pi * 2.0
            self.parg = np.arctan2(
                np.dot(ev, hn_norm), np.dot(ev, n_norm)
            )  # periapsis argument (radians)
            if self.parg < 0.0:
                self.parg += math.pi * 2.0

        self.hv = h  # orbital mormentum vecctor
        self.p = hlen2 / self.mu  # semi-latus rectum
        self.ev = ev  # eccentricity vector
        self.evd = ev_norm  # normalized eccentricity vector
        self.e = np.sqrt(np.dot(ev, ev))  # eccentricity
        if self.e == 1.0:
            self._setOrb = False
            raise (
                ValueError(
                    "Inappropriate pos and vel in TwoBodyOrbit.setOrbCart"
                )
            )
        self.a = self.p / (1.0 - self.e ** 2)  # semi-major axis
        self.i = np.arccos(h[2] / hlen)  # inclination (radians)
        self.ta0 = np.arctan2(
            np.dot(he_norm, r0), np.dot(ev_norm, r0)
        )  # true anomaly at epoch
        if self.ta0 < 0.0:
            self.ta0 += math.pi * 2.0

        # time from recent periapsis, mean anomaly, periapsis passage time
        timef = self.timeFperi(self.ta0)
        self.ma = None
        self.pr = None
        self.mm = None
        if self.e < 1.0:
            self.pr = (
                2.0 * math.pi * np.sqrt(self.a ** 3 / self.mu)
            )  # orbital period
            self.ma = timef / self.pr * math.pi * 2.0  # Mean anomaly (rad)
            self.mm = 2.0 * math.pi / self.pr  # mean motion (rad/time)
        self.T = self.t0 - timef  # periapsis passage time

    def setOrbKepl(self, epoch, a, e, i, LoAN, AoP, TA=None, T=None, MA=None):
        """Define the orbit by classical orbital elements
        
        Args:
            epoch:   Epoch
            a:       Semi-major axis
            e:       Eccentricity (should not be 1.0)
            i:       Inclination (degrees)
            LoAN:    longitude of ascending node (degrees)
                     If inclination is zero, this value defines reference
                     longitude of AoP
            AoP:     Argument of periapsis (degrees)
                     For a circular orbit, this value indicates a imaginary
                     periapsis.
            
            TA:      True anomaly on epoch (degrees)
                     For a circular orbit, the value defines angle from the 
                     imaginary periapsis defined by AoP
            T:       Periapsis passage time
                     for a circular orbit, the value defines passage time for
                     the imaginary periapsis defined by AoP
            MA:      Mean anomaly on epoch (degrees)
                     For a hyperbolic trajectory, you cannot specify this 
                     argument
                     For a circular orbit, the value defines anomaly from
                     the imaginary periapsis defined by AoP
                     
                 TA, T, and MA are mutually exclusive arguments. You should 
                 specify one of them.  If TA is specified, other arguments 
                 will be ignored. If T is specified, MA will be ignored.
        
        Exceptions:
            ValueError: If classical orbital element(s) are inconsistent, the
                method raises ValueError
        """
        # changed keys
        Lomega = LoAN
        Somega = AoP
        TAoE = TA
        ma = MA

        self._setOrb = False

        if e < 0.0:
            raise ValueError(
                "Invalid orbital element (e<0.0) in TwoBodyOrbit.setOrbKepl"
            )
        if e == 1.0:
            raise ValueError(
                "Invalid orbital element (e=1.0) in TwoBodyOrbit.setOrbKepl"
            )
        if (e > 1.0 and a >= 0.0) or (e < 1.0 and a <= 0.0):
            raise ValueError(
                "Invalid Orbital Element(s) (inconsistent e and a) in TwoBodyOrbit.setOrbKepl"
            )
        if e > 1.0 and TAoE is None and T is None:
            raise ValueError(
                "Missing Orbital Element (TA or T) in TwoBodyOrbit.setOrbKepl"
            )
        if TAoE is None and T is None and ma is None:
            raise ValueError(
                "Missing Orbital Elements (TA, T, or MA) in TwoBodyOrbit.setOrbKepl"
            )
        taError = False
        if TAoE is not None and e > 1.0:
            mta = math.degrees(math.acos((-1.0) / e))
            if TAoE >= mta and TAoE <= 180.0:
                taError = True
            elif TAoE >= 180.0 and TAoE <= (360.0 - mta):
                taError = True
            elif TAoE <= (-1.0) * mta:
                taError = True
            if taError:
                raise ValueError(
                    "Invalid Orbital Element (TA) in TwoBodyOrbit.setOrbKepl"
                )

        self.t0 = epoch
        self.a = a
        self.e = e
        self.i = math.radians(i)
        self.lan = math.radians(Lomega)
        self.parg = math.radians(Somega)

        self.pr = None
        self.ma = None
        self.mm = None

        self._setOrb = True

        # semi-latus rectum
        self.p = a * (1.0 - e * e)

        # orbital period and mean motion
        if e < 1.0:
            self.pr = math.pi * 2.0 / math.sqrt(self.mu) * a ** 1.5
            self.mm = math.pi * 2.0 / self.pr

        # R: rotation matrix
        R1n = np.array(
            [
                math.cos(self.lan) * math.cos(self.parg)
                - math.sin(self.lan) * math.sin(self.parg) * math.cos(self.i),
                (-1.0) * math.cos(self.lan) * math.sin(self.parg)
                - math.sin(self.lan) * math.cos(self.parg) * math.cos(self.i),
                math.sin(self.lan) * math.sin(self.i),
            ]
        )
        R2n = np.array(
            [
                math.sin(self.lan) * math.cos(self.parg)
                + math.cos(self.lan) * math.sin(self.parg) * math.cos(self.i),
                (-1.0) * math.sin(self.lan) * math.sin(self.parg)
                + math.cos(self.lan) * math.cos(self.parg) * math.cos(self.i),
                (-1.0) * math.cos(self.lan) * math.sin(self.i),
            ]
        )
        R3n = np.array(
            [
                math.sin(self.parg) * math.sin(self.i),
                math.cos(self.parg) * math.sin(self.i),
                math.cos(self.i),
            ]
        )
        R = np.array([R1n, R2n, R3n])

        # eccentricity vector
        self.evd = (np.dot(R, np.array([[1.0], [0.0], [0.0]]))).T[0]
        self.ev = self.evd * self.e
        # angular momentum vector
        h = math.sqrt(self.p * self.mu)
        self.hv = (np.dot(R, np.array([[0.0], [0.0], [1.0]]))).T[0] * h
        nv = (np.dot(R, np.array([[0.0], [1.0], [0.0]]))).T[0]

        # ta0, T, ma
        if TAoE is not None:
            # true anomaly at epoch
            self.ta0 = math.radians(TAoE)
            # periapsis passage time
            self.T = self.t0 - self.timeFperi(self.ta0)
            # mean anomaly at epoch
            if self.e < 1.0:
                self.ma = (self.t0 - self.T) / self.pr * math.pi * 2.0
            else:
                self.ma = None
        elif T is not None:
            # periapsis passage time
            self.T = T
            # position and velocity on periapsis
            self.pos, self.vel = self.posvel(0.0)
            # position and velocity at epoch
            self.t0 = self.T  # temporary setting
            pos, vel = self.posvelatt(epoch)
            # true anomaly at epoch
            ev_norm = self.evd
            nv_norm = nv / np.sqrt(np.dot(nv, nv))
            pos_norm = pos / np.sqrt(np.dot(pos, pos))
            # true anomaly at epoch
            self.ta0 = np.arctan2(
                np.dot(pos_norm, nv_norm), np.dot(pos_norm, ev_norm)
            )
            # mean anomaly at epoch
            if self.e < 1.0:
                self.ma = (epoch - self.T) / self.pr * math.pi * 2.0
                if self.ma < 0.0:
                    self.ma += math.pi * 2.0
            else:
                self.ma = None
        else:
            # mean anomaly at epoch
            self.ma = math.radians(ma)
            # periapsis passage time
            self.T = epoch - self.pr * self.ma / (math.pi * 2.0)
            # position and velocity on periapsis
            self.pos, self.vel = self.posvel(0.0)
            # position and velocity at epoch
            self.t0 = self.T  # temporary setting
            pos, vel = self.posvelatt(epoch)
            # true anomaly at epoch
            ev_norm = self.ev / np.sqrt(np.dot(self.ev, self.ev))
            nv_norm = nv / np.sqrt(np.dot(nv, nv))
            pos_norm = pos / np.sqrt(np.dot(pos, pos))
            # true anomaly at epoch
            self.ta0 = np.arctan2(
                np.dot(pos_norm, nv_norm), np.dot(pos_norm, ev_norm)
            )

        # epoch
        self.t0 = epoch
        # position and velocity at epoch
        if e != 0.0:
            self.pos, self.vel = self.posvel(self.ta0)
        else:
            r = (
                np.array([[math.cos(self.ta0)], [math.sin(self.ta0)], [0.0]])
                * self.a
            )
            self.pos = (np.dot(R, r).T)[0]
            v = np.array(
                [[(-1.0) * math.sin(self.ta0)], [math.cos(self.ta0)], [0.0]]
            ) * math.sqrt(self.mu / self.a)
            self.vel = (np.dot(R, v).T)[0]

    def points(self, ndata):
        """Returns points on orbital trajectory for visualization
        
        Args:
            ndata: Number of points
        Returns: xs, ys, zs, times
            xs: Array of x-coordinates (Numpy array)
            ys: Array of y-coordinates (Numpy array)
            zs: Array of z-coordinates (Numpy array)
            times: Array of times (Numpy array)
            
            Origin of coordinates are position of the central body
        """
        if not self._setOrb:
            raise (
                RuntimeError("Orbit has not been defined: TwoBodyOrbit.points")
            )

        times = np.zeros(ndata)
        xs = np.zeros(ndata)
        ys = np.zeros(ndata)
        zs = np.zeros(ndata)
        tas = np.zeros(ndata)

        if self.e < 1.0:
            tas = np.linspace(0.0, math.pi * 2.0, ndata)
        else:
            stop = math.pi - np.arccos(1.0 / self.e)
            start = (-1.0) * stop
            delta = (stop - start) / (ndata + 1)
            tas = np.linspace(start + delta, stop - delta, ndata)
        for j in range(ndata):
            ta = tas[j]
            times[j] = self.timeFperi(ta) + self.T
            xyz, xdydzd = self.posvel(ta)
            xs[j] = xyz[0]
            ys[j] = xyz[1]
            zs[j] = xyz[2]

        return xs, ys, zs, times

    def posvelatt(self, t):
        """Returns position and velocity of the object at given t
        
        Args:
            t: Time
        Returns: newpos, newvel
            newpos: Position of the object at t (x,y,z) (Numpy array)
            newvel: Velocity of the object at t (xd,yd,zd) (Numpy array)
        Exception:
            RuntimeError: If it failed to the computation, raises RuntimeError
            
            Origin of coordinates are position of the central body
        """

        def _Cz(z):
            if z < 0:
                return (1.0 - np.cosh(np.sqrt((-1) * z))) / z
            else:
                return (1.0 - np.cos(np.sqrt(z))) / z

        def _Sz(z):
            if z < 0:
                sqz = np.sqrt((-1) * z)
                return (np.sinh(sqz) - sqz) / sqz ** 3
            else:
                sqz = np.sqrt(z)
                return (sqz - np.sin(sqz)) / sqz ** 3

        def _func(xn, targett):
            z = xn * xn / self.a
            sr = np.sqrt(np.dot(self.pos, self.pos))
            tn = (
                np.dot(self.pos, self.vel)
                / np.sqrt(self.mu)
                * xn
                * xn
                * _Cz(z)
                + (1.0 - sr / self.a) * xn ** 3 * _Sz(z)
                + sr * xn
            ) / np.sqrt(self.mu) - targett
            return tn

        def _fprime(x, targett):
            z = x * x / self.a
            sqmu = np.sqrt(self.mu)
            sr = np.sqrt(np.dot(self.pos, self.pos))
            dtdx = (
                x * x * _Cz(z)
                + np.dot(self.pos, self.vel) / sqmu * x * (1.0 - z * _Sz(z))
                + sr * (1.0 - z * _Cz(z))
            ) / sqmu
            return dtdx

        if not self._setOrb:
            raise (
                RuntimeError(
                    "Orbit has not been defined: TwoBodyOrbit.posvelatt"
                )
            )

        delta_t = t - self.t0
        if delta_t == 0.0:
            return self.pos + 0.0, self.vel + 0.0
            # you should not return self.pos. it can cause trouble!
        x0 = np.sqrt(self.mu) * delta_t / self.a
        try:
            # compute with scipy.optimize.newton
            xn = newton(_func, x0, args=(delta_t,), fprime=_fprime)
        except RuntimeError:
            # Configure boundaries for scipy.optimize.bisect
            # b1: Lower boundary
            # b2: Upper boundary
            f0 = _func(x0, delta_t)
            if f0 < 0.0:
                b1 = x0
                found = False
                for i in range(50):
                    x1 = x0 + 10 ** (i + 1)
                    test = _func(x1, delta_t)
                    if test > 0.0:
                        found = True
                        b2 = x1
                        break
                if not found:
                    raise (
                        RuntimeError(
                            "Could not compute position and "
                            + "velocity: TwoBodyOrbit.posvelatt"
                        )
                    )
            else:
                b2 = x0
                found = False
                for i in range(50):
                    x1 = x0 - 10 ** (i + 1)
                    test = _func(x1, delta_t)
                    if test < 0.0:
                        found = True
                        b1 = x1
                        break
                if not found:
                    raise (
                        RuntimeError(
                            "Could not compute position and "
                            + "velocity: TwoBodyOrbit.posvelatt"
                        )
                    )

            # compute with scipy.optimize.bisect
            xn = bisect(_func, b1, b2, args=(delta_t,), maxiter=200)

        z = xn * xn / self.a
        sr = np.sqrt(np.dot(self.pos, self.pos))
        sqmu = np.sqrt(self.mu)
        val_f = 1.0 - xn * xn / sr * _Cz(z)
        val_g = delta_t - xn ** 3 / sqmu * _Sz(z)
        newpos = self.pos * val_f + self.vel * val_g
        newr = np.sqrt(np.dot(newpos, newpos))
        val_fd = sqmu / sr / newr * xn * (z * _Sz(z) - 1.0)
        val_gd = 1.0 - xn * xn / newr * _Cz(z)
        newvel = self.pos * val_fd + self.vel * val_gd
        return newpos, newvel

    def elmKepl(self):
        """Returns Classical orbital element
        
        Returns:
            kepl: Dictionary of orbital elements. Keys are as follows
                'epoch': Epoch
                'a': Semimajor axis
                'e': Eccentricity
                'i': Inclination in degrees
                'LoAN': Longitude of ascending node in degrees
                    If inclination is zero, LoAN yields reference longitude
                    for AoP
                'AoP': Argument of periapsis in degrees
                    If inclination is zero, AoP yields angle from reference
                    longitude (LoAN)
                    For circular orbit, AoP yields imaginary periapsis
                'TA': True anomaly at epoch in degrees
                    For circular orbit, TAoE yields angle from imaginary 
                    periapsis (AoP)
                'T': Periapsis passage time
                    For circular orbit, T yields passage time of imaginary
                    periapsis (AoP)
                'MA': Mean anomaly at epoch in degrees (elliptic orbit only)
                    For circular orbit, ma is the same to TAoE
                'n': Mean motion in degrees (elliptic orbit only)
                'P': Orbital period (elliptic orbit only)
                
                For a hyperbolic trajectory, values for keys 'MA', 'n', and 'P' 
                are None for each
        """
        if not self._setOrb:
            raise (RuntimeError("Orbit has not been defined: TwoBodyOrbit"))

        kepl = {
            "epoch": self.t0,
            "a": self.a,
            "e": self.e,
            "i": math.degrees(self.i),
            "LoAN": math.degrees(self.lan),
            "AoP": math.degrees(self.parg),
            "TA": math.degrees(self.ta0),
            "T": self.T,
        }
        if self.e < 1.0:
            kepl["MA"] = math.degrees(self.ma)
            kepl["n"] = math.degrees(self.mm)
            kepl["P"] = self.pr
        else:
            kepl["MA"] = None
            kepl["n"] = None
            kepl["P"] = None

        return kepl


def lambert(ipos, tpos, targett, mu=1.32712440041e20, ccw=True):
    """A function to solve 'Lambert's Problem'
    
    From given initial position, terminal position, and flight time, 
    compute initial velocity and terminal velocity.
    Args: ipos, tpos, targett, mu, ccw
        ipos: Initial position of the object (x,y,z) (array-like object)
        tpos: Terminal position of the object (x,y,z) (array-like object)
        targett: Flight time
        mu: Gravitational parameter of the central body (default value is for the Sun)
        ccw: Flag for orbital direction. If True, counter clockwise
    Returns: ivel, tvel
        ivel: Initial velocity of the object (xd,yd,zd) as Numpy array
        tvel: Terminal velocity of the object (xd,yd,zd) as Numpy array
    Exception:
        ValueError: When input data (ipos, tpos, targett) are inappropriate,
                    the function raises ValueError
                    
        Origin of coordinates are position of the central body
    """

    def _Cz(z):
        if z < 0:
            return (1.0 - np.cosh(np.sqrt((-1) * z))) / z
        else:
            return (1.0 - np.cos(np.sqrt(z))) / z

    def _Sz(z):
        if z < 0:
            sqz = np.sqrt((-1) * z)
            return (np.sinh(sqz) - sqz) / sqz ** 3
        else:
            sqz = np.sqrt(z)
            return (sqz - np.sin(sqz)) / sqz ** 3

    def _func(z, targett, r1pr2, A, mu):
        val_y = r1pr2 - A * (1.0 - z * _Sz(z)) / np.sqrt(_Cz(z))
        val_x = np.sqrt(val_y / _Cz(z))
        t = (val_x ** 3 * _Sz(z) + A * np.sqrt(val_y)) / np.sqrt(mu)

        return t - targett

    sipos = np.array(ipos)
    stpos = np.array(tpos)
    tsec = targett * 1.0

    r1 = np.sqrt(np.dot(sipos, sipos))
    r2 = np.sqrt(np.dot(stpos, stpos))

    r1cr2 = np.cross(sipos, stpos)
    r1dr2 = np.dot(sipos, stpos)
    sindnu = np.sqrt(np.dot(r1cr2, r1cr2)) / r1 / r2

    if r1cr2[2] < 0.0:
        sindnu = (-1) * sindnu
    if not ccw:
        sindnu = (-1) * sindnu

    cosdnu = r1dr2 / r1 / r2
    A = np.sqrt(r1 * r2) * sindnu / np.sqrt(1.0 - cosdnu)
    r1pr2 = r1 + r2

    dnu = np.arctan2(sindnu, cosdnu)
    if dnu < 0.0:
        dnu += math.pi * 2.0

    # Check difference of true anomaly of two points
    # The threshold 0.001 is an empirical value
    if dnu < 0.001 or dnu > (math.pi * 2.0 - 0.001):
        raise (
            ValueError(
                "Difference in true anomaly is too small:"
                + " pytwobodyorbit.lambert"
            )
        )

    # Check difference of true anomaly of two points
    # The threshold 0.00001 is an empirical value
    if (dnu - math.pi) ** 2 < 0.00001 ** 2:
        raise (
            ValueError(
                "Two points are placed opposite each"
                + " other: pytwobodyorbit.lambert"
            )
        )

    # Configure boundaries for scipy.optimize.bisect
    # b1: Lower boundary
    # b2: Upper boundary
    inf = float("inf")
    minb1 = (-1.0) * (math.pi * 2.0) ** 2  # minimum limit for b1

    # find b2 candidate
    found = False
    for i in range(10):
        b2 = (math.pi * 2.0) ** 2 - 1.0 / 10.0 ** i
        test = _func(b2, tsec, r1pr2, A, mu)
        if test == test and test != inf:  # if (not 'nan') and (!= 'inf')
            if test > 0.0:
                found = True
                break
    if not found:
        raise (
            ValueError(
                "Could not solve Lambert's Plobrem: pytwobodyorbit.lambert"
            )
        )

    # configure b1, and b2
    b1 = (-1.0) * dnu ** 2
    lastb1 = b2
    found = False
    for i in range(100):
        test = _func(b1, tsec, r1pr2, A, mu)
        if test == test and test != inf:  # if (not 'nan') and (!= 'inf')
            if test > 0.0:
                lastb1 = b1
                b1 = (b1 + minb1) / 2.0
            else:
                b2 = lastb1
                found = True
                break
        else:
            b1 = (b1 + lastb1) / 2.0
    if not found:
        raise (
            ValueError(
                "Could not solve Lambert's Plobrem: pytwobodyorbit.lambert"
            )
        )

    zn = bisect(_func, b1, b2, args=(tsec, r1pr2, A, mu), maxiter=100)

    val_y = r1pr2 - A * (1.0 - zn * _Sz(zn)) / np.sqrt(_Cz(zn))
    val_f = 1.0 - val_y / r1
    val_g = A * np.sqrt(val_y / mu)
    val_gd = 1.0 - val_y / r2

    ivel = (stpos - val_f * sipos) / val_g
    tvel = (val_gd * stpos - sipos) / val_g

    return ivel, tvel


if __name__ == "__main__":

    mu = 398600.4418  # Earth Gravitional Constant km^2/s^2

    # Example of Use of 2 Body Propagator and Visualize
    s = TwoBodyOrbit("RSO", mu=mu)  # create an instance
    a = random.uniform(41164, 43164)  # semi-major axis
    e = random.uniform(0, 0.1)  # eccentricity
    i = random.uniform(0, 20)  # inclination
    LoAN = random.uniform(0, 360)  # longitude of ascending node
    AoP = random.uniform(0, 360)  # argument of perigee
    MA = random.uniform(0, 360)  # mean anomaly
    s.setOrbKepl(0, a, e, i, LoAN, AoP, MA=MA)  # define the orbit

    x = []
    y = []
    z = []
    T = [q for q in range(0, 86400, 60)]
    for t in T:
        p, v = s.posvelatt(t)
        x.append(p[0])
        y.append(p[1])
        z.append(p[2])

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel="Px (km)", ylabel="Py (km)", title="Example Satellite")
    ax.grid()

    fig, ax = plt.subplots()
    ax.plot(T, z)
    ax.set(
        xlabel="Epoch (seconds)", ylabel="Pz (km)", title="Example Satellite"
    )
    ax.grid()

    plt.show()
