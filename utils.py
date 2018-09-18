# -*- coding: utf-8 -*-
"""Utility functions."""
import datetime as dt
from pytz import utc
from numpy import exp


datetime0 = dt.datetime(1900, 1, 1, 0, 0, 0, tzinfo=utc)


def zip_el(*args):
    """"Zip iterables, only if input lists have same length.

    Args:
        *args: Variable number of lists.

    Returns:
        list: Iterator that aggregates elements from each of the input lists.

    Raises:
        AssertError: If input lists do not have the same length.

    """
    lengths = [len(l) for l in args]
    assert all([l == lengths[0] for l in lengths[1:]]), "All the input lists should have the same length."
    return zip(*args)


def hour_to_date_str(hour, str_format=None):
    """"Convert hour since 1900-01-01 00:00 to string of date.

    Args:
        hour (int): Hour since 1900-01-01 00:00.
        str_format (str, optional): Explicit format string from datetime packages. Defaults to isoformat.

    Returns:
        str: String representing the timestamp.

    """
    date = hour_to_date(hour)
    if str_format is None:
        return date.isoformat()
    else:
        return date.strftime(str_format)


def hour_to_date(hour):
    """"Convert hour since 1900-01-01 00:00 to datetime object.

    Args:
        hour (int): Hour since 1900-01-01 00:00.

    Returns:
        datetime: Datetime object of timestamp.

    """
    date = (datetime0 + dt.timedelta(hours=int(hour)))
    return date


def get_density_at_altitude(altitude):
    """"Barometric altitude formula for constant temperature, source: Meteorology for Scientists and Engineers.

    Args:
        altitude (float): Height above sea level [m].

    Returns:
        float: Density at requested altitude [kg/m^3].

    """
    rho_0 = 1.225  # Standard atmospheric density at sea level at the standard temperature.
    h_p = 8.55e3  # Scale height for density.
    return exp(-altitude/h_p)*rho_0
