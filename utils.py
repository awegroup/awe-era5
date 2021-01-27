# -*- coding: utf-8 -*-
"""Utility functions."""
import datetime as dt
from pytz import utc
import numpy as np

datetime0 = dt.datetime(1900, 1, 1, 0, 0, 0, tzinfo=utc)  # Date used as starting point for counting hours.
r_d = 287.06  # Gas constant for dry air [J/K/kg]
g = 9.80665  # Gravitational acceleration [m/s^2]


def zip_el(*args):
    """Zip iterables, only if input lists have same length.

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
    """Convert hour since 1900-01-01 00:00 to string of date.

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
    """Convert hour since 1900-01-01 00:00 to datetime object.

    Args:
        hour (int): Hour since 1900-01-01 00:00.

    Returns:
        datetime: Datetime object of timestamp.

    """
    date = (datetime0 + dt.timedelta(hours=int(hour)))
    return date


def get_ph_levs(level, sp):
    """Get the half-level pressures for the requested ERA5 model level and the one after that. The a and b coefficients
    define the model levels and are provided in `L137 model level definitions`_.

    Args:
        level (int): Model level identifier.
        sp (float): Surface pressure [Pa].

    Returns:
        tuple of float: Half-level pressures for the requested ERA5 model level and the one after that [Pa].

    .. _L137 model level definitions:
        https://www.ecmwf.int/en/forecasts/documentation-and-support/137-model-levels

    """

    a_coef = [0, 2.000365, 3.102241, 4.666084, 6.827977,	9.746966, 13.605424, 18.608931, 24.985718, 32.98571,
        42.879242, 54.955463, 69.520576, 86.895882, 107.415741, 131.425507, 159.279404, 191.338562, 227.968948,
        269.539581, 316.420746, 368.982361, 427.592499, 492.616028, 564.413452, 643.339905, 729.744141, 823.967834,
        926.34491, 1037.201172, 1156.853638, 1285.610352, 1423.770142, 1571.622925, 1729.448975, 1897.519287,
        2076.095947, 2265.431641, 2465.770508, 2677.348145, 2900.391357, 3135.119385, 3381.743652, 3640.468262,
        3911.490479, 4194.930664, 4490.817383, 4799.149414, 5119.89502, 5452.990723, 5798.344727, 6156.074219,
        6526.946777, 6911.870605, 7311.869141, 7727.412109, 8159.354004, 8608.525391, 9076.400391, 9562.682617,
        10065.978516, 10584.631836, 11116.662109, 11660.067383, 12211.547852, 12766.873047, 13324.668945, 13881.331055,
        14432.139648, 14975.615234, 15508.256836, 16026.115234, 16527.322266, 17008.789063, 17467.613281, 17901.621094,
        18308.433594, 18685.71875, 19031.289063, 19343.511719, 19620.042969, 19859.390625, 20059.931641, 20219.664063,
        20337.863281, 20412.308594, 20442.078125, 20425.71875, 20361.816406, 20249.511719, 20087.085938, 19874.025391,
        19608.572266, 19290.226563, 18917.460938, 18489.707031, 18006.925781, 17471.839844, 16888.6875, 16262.046875,
        15596.695313, 14898.453125, 14173.324219, 13427.769531, 12668.257813, 11901.339844, 11133.304688, 10370.175781,
        9617.515625, 8880.453125, 8163.375, 7470.34375, 6804.421875, 6168.53125, 5564.382813, 4993.796875, 4457.375,
        3955.960938, 3489.234375, 3057.265625, 2659.140625, 2294.242188, 1961.5, 1659.476563, 1387.546875, 1143.25,
        926.507813, 734.992188, 568.0625, 424.414063, 302.476563, 202.484375, 122.101563, 62.78125, 22.835938, 3.757813,
        0, 0]

    b_coef = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        7e-06, 2.4e-05, 5.9e-05, 0.000112, 0.000199, 0.00034, 0.000562, 0.00089, 0.001353, 0.001992,
        0.002857, 0.003971, 0.005378, 0.007133, 0.009261, 0.011806, 0.014816, 0.018318, 0.022355, 0.026964,
        0.032176, 0.038026, 0.044548, 0.051773, 0.059728, 0.068448, 0.077958, 0.088286, 0.099462, 0.111505,
        0.124448, 0.138313, 0.153125, 0.16891, 0.185689, 0.203491, 0.222333, 0.242244, 0.263242, 0.285354,
        0.308598, 0.332939, 0.358254, 0.384363, 0.411125, 0.438391, 0.466003, 0.4938, 0.521619, 0.549301,
        0.576692, 0.603648, 0.630036, 0.655736, 0.680643, 0.704669, 0.727739, 0.749797, 0.770798, 0.790717,
        0.809536, 0.827256, 0.843881, 0.859432, 0.873929, 0.887408, 0.8999, 0.911448, 0.922096, 0.931881,
        0.94086, 0.949064, 0.95655, 0.963352, 0.969513, 0.975078, 0.980072, 0.984542, 0.9885, 0.991984,
        0.995003, 0.99763, 1]

    ph_lev = a_coef[level - 1] + (b_coef[level - 1] * sp)
    ph_levplusone = a_coef[level] + (b_coef[level] * sp)
    return ph_lev, ph_levplusone


def compute_level_height(t, q, level, ph_lev, ph_levplusone, h_h):
    """Compute height at half- & full-level for the requested ERA5 model level, based on temperature, humidity, and
    half-level pressures.

    Args:
        t (float): Temperature at the requested model level [K].
        q (float): Humidity at the requested model level [kg/kg].
        level (int): Model level identifier.
        ph_lev (float): Half-level pressure for the requested model level [Pa].
        ph_levplusone (float): Half-level pressures for the subsequent model level [Pa].
        h_h (float): Half-level height of previous model level [m].

    Returns:
        tuple of float: Half- and full-level heights of requested model level.

    """
    # Compute the moist temperature.
    t = t * (1. + 0.609133 * q)

    if level == 1:
        dlog_p = np.log(ph_levplusone / 0.1)
        alpha = np.log(2)
    else:
        dlog_p = np.log(ph_levplusone / ph_lev)
        alpha = 1. - ((ph_lev / (ph_levplusone - ph_lev)) * dlog_p)

    # Integrate from previous (lower) half-level h_h to the full-level.
    h_f = h_h + (t * r_d * alpha)/g

    # Integrate from previous (lower) half-level h_h to the current half-level.
    h_h = h_h + (t * r_d * dlog_p)/g

    return h_h, h_f


def compute_level_heights(levels, surface_pressure, levels_temperature, levels_humidity):
    """Compute the full-level heights and air densities for the given model levels.

    Args:
        levels (list): Identifiers of model levels to evaluate - should be consecutive and include the lower level.
        surface_pressure (ndarray): Time trace of surface pressure [Pa].
        levels_temperature (ndarray): Time traces of temperature at model levels [K].
        levels_humidity (ndarray): Time traces of humidity at model levels [kg/kg].

    Returns:
        tuple of ndarray: Time traces of Full-level heights [m] and air densities [kg/m^3] at requested model levels.

    Raises:
        AssertError: If requested model levels are not consecutive or don't include the lower level.

    """
    assert np.all(np.diff(levels) == 1) and levels[-1] == 137, "Provided levels should be consecutive."
    n_levels = len(levels)
    n_hours = levels_temperature.shape[0]
    densities = np.zeros((n_hours, n_levels))
    heights = np.zeros((n_hours, n_levels))
    for i_hr in range(n_hours):
        h_h = 0  # Half-level height at lower model level.
        for i, level in enumerate(reversed(levels)):  # Start from lower model level.
            i_level = n_levels - 1 - i  # Identifier of model level.

            # Get the half-level pressures.
            ph_lev, ph_levplusone = get_ph_levs(level, surface_pressure[i_hr])

            # Determine half- & full-level height using previous half-level height.
            h_h, h_f = compute_level_height(levels_temperature[i_hr, i_level], levels_humidity[i_hr, i_level],
                                            level, ph_lev, ph_levplusone, h_h)
            heights[i_hr, i_level] = h_f

            # Determine full-level air density.
            pf = (ph_lev+ph_levplusone)/2  # Full-level pressure.
            densities[i_hr, i_level] = pf/(r_d*levels_temperature[i_hr, i_level])

    return heights, densities


def flatten_dict(input_dict, parent_key='', sep='.'):
    """"Recursive function to convert multi-level dictionary to flat dictionary. 

    Args: 
        input_dict (dict): Dictionary to be flattened
        parent_key (str): Key under which 'input_dict' is stored in the higher-level dictionary
        sep (str): Separator used for joining together the keys pointing to the lower-level object.
    """
    items = []
    for k, v in input_dict.items():
        new_key = parent_key + sep + str(k).replace(" ", "") if parent_key else str(k).replace(" ", "")
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else: 
            items.append((new_key, v))
    return dict(items)
