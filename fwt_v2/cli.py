"""The portable distribution exposes only the reviewed training entry."""


def main(argv=None):
    raise SystemExit('Use python train.py --phase det|diff. '
                     'Calibration and held-out evaluation are not distributed commands.')
