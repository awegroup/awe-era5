import datetime as dt
from pytz import utc
from numpy import exp


datetime0 = dt.datetime(1900, 1, 1, 0, 0, 0, tzinfo=utc)


def zip_el(*args):
    lengths = [len(l) for l in args]
    assert all([l == lengths[0] for l in lengths[1:]]), "All the input lists should have the same length."
    return zip(*args)


def hour_to_date_str(hour, str_format=None):
    date = (datetime0 + dt.timedelta(hours=int(hour)))
    if str_format is None:
        return date.isoformat()
    else:
        return date.strftime(str_format)


def hour_to_date(hour):
    date = (datetime0 + dt.timedelta(hours=int(hour)))
    return date


def get_density_at_altitude(altitude):
    # barometric altitude formula for constant temperature, source: Meteorology for Scientists and Engineers
    rho_0 = 1.225  # standard atmospheric density at sea level at the standard temperature
    h_p = 8.55e3  # scale height for density
    return exp(-altitude/h_p)*rho_0


if __name__ == "__main__":
    speeds = [4., 8., 14., 25.]
    rho_500 = 1.2  #get_density_at_altitude(500.)
    print(rho_500)
    print(["{:.1f}".format(1./2*rho_500*v**3) for v in speeds])