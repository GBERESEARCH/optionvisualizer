"""
Utility functions for refreshing parameters

"""

import copy
import numpy as np
import scipy.stats as si
from optionvisualizer.visualizer_params import vis_params_dict
# pylint: disable=invalid-name

class Utils():
    """
    Utility functions for refreshing parameters

    """
    @staticmethod
    def _init_params(inputs: dict) -> dict:
        """
        Initialise parameter dictionary
        Parameters
        ----------
        inputs : Dict
            Dictionary of parameters supplied to the function.
        Returns
        -------
        params : Dict
            Dictionary of parameters.
        """
        # Copy the default parameters
        params = copy.deepcopy(vis_params_dict)

        # For all the supplied arguments
        for key, value in inputs.items():

            # Replace the default parameter with that provided
            params[key] = value

        return params


    @staticmethod
    def refresh_dist_params(
        opt_params: dict,
        params: dict) -> dict:
        """
        Calculate various parameters and distributions

        Returns
        -------
        Various
            Assigns parameters to the object

        """

        # Cost of carry as risk free rate less dividend yield
        params['b'] = opt_params['r'] - opt_params['q']

        params['carry'] = np.exp(
            (params['b'] - opt_params['r']) * opt_params['T'])
        params['discount'] = np.exp(-opt_params['r'] * opt_params['T'])

        with np.errstate(divide='ignore'):
            params['d1'] = (
                (np.log(opt_params['S'] / opt_params['K'])
                 + (params['b'] + (0.5 * opt_params['sigma'] ** 2))
                 * opt_params['T'])
                / (opt_params['sigma'] * np.sqrt(opt_params['T'])))

            params['d2'] = (
                (np.log(opt_params['S'] / opt_params['K'])
                 + (params['b'] - (0.5 * opt_params['sigma'] ** 2))
                 * opt_params['T'])
                / (opt_params['sigma'] * np.sqrt(opt_params['T'])))

            # standardised normal density function
            params['nd1'] = (
                (1 / np.sqrt(2 * np.pi)) * (np.exp(-params['d1'] ** 2 * 0.5)))

            # Cumulative normal distribution function
            params['Nd1'] = si.norm.cdf(params['d1'], 0.0, 1.0)
            params['minusNd1'] = si.norm.cdf(-params['d1'], 0.0, 1.0)
            params['Nd2'] = si.norm.cdf(params['d2'], 0.0, 1.0)
            params['minusNd2'] = si.norm.cdf(-params['d2'], 0.0, 1.0)

        return params


    @staticmethod
    def refresh_combo_params(
        params: dict,
        inputs: dict) -> dict:
        """
        Set parameters for use in various pricing functions

        Parameters
        ----------
        **kwargs : Various
                   Takes any of the arguments of the various methods
                   that use it to refresh data.

        Returns
        -------
        Various
            Runs methods to fix input parameters and reset defaults
            if no data provided

        """
        default_values = copy.deepcopy(vis_params_dict)

        # Certain combo payoffs (found in the mod_payoffs list) require
        # specific default parameters
        if params['combo_payoff'] in params['mod_payoffs']:

            # For each parameter in the combo parameters dictionary
            # corresponding to this combo payoff
            for parameter in params[
                    'combo_parameters'][params['combo_payoff']]:

                # If the parameter has been supplied as an input
                if parameter in inputs.keys():

                    # Set this value in the parameter dictionary
                    params[parameter] = inputs[parameter]

                # Otherwise if the parameter is in the mod_params list
                elif parameter in params['mod_params']:

                    # Try to extract this from the combo dict default
                    try:
                        params[parameter] = params['combo_dict'][str(
                            params['combo_payoff'])][str(parameter)]

                    # Otherwise
                    except KeyError:
                        # Set to the standard default value
                        params[parameter] = default_values[str(parameter)]
                # Otherwise
                else:
                    # Set to the standard default value
                    params[parameter] = default_values[str(parameter)]

        # For all the other combo_payoffs
        else:
            # For each parameter in the combo parameters dictionary
            # corresponding to this combo payoff
            for parameter in params[
                    'combo_parameters'][params['combo_payoff']]:

                # If the parameter has been supplied as an input
                if parameter in inputs.keys():

                    # Set this value in the parameter dictionary
                    params[parameter] = inputs[parameter]

                # Otherwise
                else:
                    # Set to the standard default value
                    params[parameter] = default_values[str(parameter)]

        return params


    @staticmethod
    def barrier_factors(params: dict) -> tuple[dict, dict]:
        """
        Calculate the barrier option specific parameters

        Returns
        -------
        Various
            Assigns parameters to the object

        """

        # Cost of carry as risk free rate less dividend yield
        params['b'] = params['r'] - params['q']

        barrier_factors = {}

        barrier_factors['mu'] = ((
            params['b'] - ((params['sigma'] ** 2) / 2))
            / (params['sigma'] ** 2))

        barrier_factors['lambda'] = (
            np.sqrt(barrier_factors['mu'] ** 2 + (
                (2 * params['r']) / params['sigma'] ** 2)))

        barrier_factors['z'] = (
            (np.log(params['H'] / params['S'])
             / (params['sigma'] * np.sqrt(params['T'])))
            + (barrier_factors['lambda']
               * params['sigma']
               * np.sqrt(params['T'])))

        barrier_factors['x1'] = (
            np.log(params['S'] / params['K'])
            / (params['sigma'] * np.sqrt(params['T']))
            + ((1 + barrier_factors['mu'])
               * params['sigma']
               * np.sqrt(params['T'])))

        barrier_factors['x2'] = (
            np.log(params['S'] / params['H'])
            / (params['sigma']
               * np.sqrt(params['T']))
            + ((1 + barrier_factors['mu'])
               * params['sigma']
               * np.sqrt(params['T'])))

        barrier_factors['y1'] = (
            np.log((params['H'] ** 2)
                   / (params['S']
                      * params['K']))
            / (params['sigma']
               * np.sqrt(params['T']))
            + ((1 + barrier_factors['mu'])
               * params['sigma']
               * np.sqrt(params['T'])))

        barrier_factors['y2'] = (
            np.log(params['H'] / params['S'])
            / (params['sigma']
               * np.sqrt(params['T']))
            + ((1 + barrier_factors['mu'])
               * params['sigma']
               * np.sqrt(params['T'])))

        params['carry'] = np.exp((params['b'] - params['r']) * params['T'])

        barrier_factors['A'] = (
            (params['phi']
             * params['S']
             * params['carry']
             * si.norm.cdf((params['phi']
                            * barrier_factors['x1']), 0.0, 1.0))
            - (params['phi']
               * params['K']
               * np.exp(-params['r']
                        * params['T'])
               * si.norm.cdf(
                   ((params['phi'] * barrier_factors['x1'])
                    - (params['phi'] * params['sigma']
                       * np.sqrt(params['T']))), 0.0, 1.0)))


        barrier_factors['B'] = (
            (params['phi']
             * params['S']
             * params['carry']
             * si.norm.cdf((params['phi']
                            * barrier_factors['x2']), 0.0, 1.0))
            - (params['phi']
               * params['K']
               * np.exp(-params['r']
                        * params['T'])
               * si.norm.cdf(
                   ((params['phi'] * barrier_factors['x2'])
                    - (params['phi'] * params['sigma']
                       * np.sqrt(params['T']))), 0.0, 1.0)))

        barrier_factors['C'] = (
            (params['phi']
             * params['S']
             * params['carry']
             * ((params['H'] / params['S'])
                ** (2 * (barrier_factors['mu'] + 1)))
             * si.norm.cdf((params['eta']
                            * barrier_factors['y1']), 0.0, 1.0))
            - (params['phi']
               * params['K']
               * np.exp(-params['r']
                        * params['T'])
               * ((params['H'] / params['S'])
                  ** (2 * barrier_factors['mu']))
               * si.norm.cdf(
                   ((params['eta'] * barrier_factors['y1'])
                    - (params['eta'] * params['sigma']
                       * np.sqrt(params['T']))), 0.0, 1.0)))

        barrier_factors['D'] = (
            (params['phi'] * params['S'] * params['carry']
             * ((params['H'] / params['S'])
                ** (2 * (barrier_factors['mu'] + 1)))
             * si.norm.cdf((params['eta']
                            * barrier_factors['y2']), 0.0, 1.0))
            - (params['phi']
               * params['K']
               * np.exp(-params['r']
                        * params['T'])
               * ((params['H'] / params['S'])
                  ** (2 * barrier_factors['mu']))
               * si.norm.cdf(
                   ((params['eta'] * barrier_factors['y2'])
                    - (params['eta'] * params['sigma']
                       * np.sqrt(params['T']))), 0.0, 1.0)))

        barrier_factors['E'] = (
            (params['R'] * np.exp(-params['r'] * params['T']))
            * (si.norm.cdf(
                ((params['eta'] * barrier_factors['x2'])
                 - (params['eta']
                    * params['sigma']
                    * np.sqrt(params['T']))), 0.0, 1.0)
                - (((params['H'] / params['S'])
                    ** (2 * barrier_factors['mu']))
                   * si.norm.cdf(
                       ((params['eta'] * barrier_factors['y2'])
                        - (params['eta'] * params['sigma']
                           * np.sqrt(params['T']))), 0.0, 1.0))))

        barrier_factors['F'] = (
            params['R'] * (((params['H'] / params['S'])
                            ** (barrier_factors['mu']
                                + barrier_factors['lambda']))
                      * (si.norm.cdf((params['eta']
                                      * barrier_factors['z']), 0.0, 1.0))
                      + (((params['H'] / params['S'])
                          ** (barrier_factors['mu']
                              - barrier_factors['lambda']))
                         * si.norm.cdf(
                             ((params['eta'] * barrier_factors['z'])
                              - (2 * params['eta']
                                 * barrier_factors['lambda']
                                 * params['sigma']
                                 * np.sqrt(params['T']))), 0.0, 1.0))))

        return barrier_factors, params
