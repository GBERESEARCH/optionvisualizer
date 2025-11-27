"""
Option Pricing and Greeks formulas

"""
import numpy as np
from optionvisualizer.utilities import Utils
# pylint: disable=invalid-name

class Option():
    """
    Calculate Black Scholes Option Price and Greeks

    """

    @staticmethod
    def price(
        opt_params: dict,
        params: dict) -> float:
        """
        Black Scholes Option Price

        Parameters
        ----------
        opt_params : Dict
            S : Float
                Underlying Stock Price. The default is 100.
            K : Float
                Strike Price. The default is 100.
            T : Float
                Time to Maturity. The default is 0.25 (3 months).
            r : Float
                Interest Rate. The default is 0.05 (5%).
            q : Float
                Dividend Yield. The default is 0.
            sigma : Float
                Implied Volatility. The default is 0.2 (20%).
            option : Str
                Option type, Put or Call. The default is 'call'
        params : Dict
            Dictionary of key parameters; used for refreshing distribution.

        Returns
        -------
        Float
            Black Scholes Option Price.

        """

        # Update distribution parameters
        params = Utils.refresh_dist_params(
            opt_params=opt_params, params=params)

        if opt_params['option'] == "call":
            opt_price = (
                (opt_params['S'] * params['carry'] * params['Nd1'])
                - (opt_params['K'] * np.exp(-opt_params['r'] * opt_params['T'])
                   * params['Nd2']))
        elif opt_params['option'] == "put":
            opt_price = (
                (opt_params['K'] * np.exp(-opt_params['r'] * opt_params['T'])
                 * params['minusNd2'])
                - (opt_params['S'] * params['carry'] * params['minusNd1']))

        else:
            raise ValueError("Please supply an option type, 'put' or 'call'")

        opt_price = np.nan_to_num(opt_price)

        return opt_price


    @staticmethod
    def delta(
        opt_params: dict,
        params: dict) -> float:
        """
        Sensitivity of the option price to changes in asset price

        Parameters
        ----------
        opt_params : Dict
            S : Float
                Underlying Stock Price. The default is 100.
            K : Float
                Strike Price. The default is 100.
            T : Float
                Time to Maturity. The default is 0.25 (3 months).
            r : Float
                Interest Rate. The default is 0.05 (5%).
            q : Float
                Dividend Yield. The default is 0.
            sigma : Float
                Implied Volatility. The default is 0.2 (20%).
            option : Str
                Option type, Put or Call. The default is 'call'
        params : Dict
            Dictionary of key parameters; used for refreshing distribution.

        Returns
        -------
        Float
            Option Delta.

        """

        # Update distribution parameters
        params = Utils.refresh_dist_params(
            opt_params=opt_params, params=params)

        if opt_params['option'] == 'call':
            opt_delta = params['carry'] * params['Nd1']
        elif opt_params['option'] == 'put':
            opt_delta = params['carry'] * (params['Nd1'] - 1)

        else:
            raise ValueError(f"Choose option type put or call: {opt_params['option']}")

        return opt_delta


    @staticmethod
    def theta(
        opt_params: dict,
        params: dict) -> float:
        """
        Sensitivity of the option price to changes in time to maturity

        Parameters
        ----------
        opt_params : Dict
            S : Float
                Underlying Stock Price. The default is 100.
            K : Float
                Strike Price. The default is 100.
            T : Float
                Time to Maturity. The default is 0.25 (3 months).
            r : Float
                Interest Rate. The default is 0.05 (5%).
            q : Float
                Dividend Yield. The default is 0.
            sigma : Float
                Implied Volatility. The default is 0.2 (20%).
            option : Str
                Option type, Put or Call. The default is 'call'
        params : Dict
            Dictionary of key parameters; used for refreshing distribution.

        Returns
        -------
        Float
            Option Theta.

        """

        # Update distribution parameters
        params = Utils.refresh_dist_params(
            opt_params=opt_params, params=params)

        if opt_params['option'] == 'call':
            opt_theta = (
                ((-opt_params['S']
                  * params['carry']
                  * params['nd1']
                  * opt_params['sigma'])
                 / (2 * np.sqrt(opt_params['T']))
                - (params['b'] - opt_params['r'])
                * (opt_params['S']
                   * params['carry']
                   * params['Nd1'])
                - (opt_params['r'] * opt_params['K'])
                * np.exp(-opt_params['r'] * opt_params['T'])
                * params['Nd2'])
                / 100)

        elif opt_params['option'] == 'put':
            opt_theta = (
                ((-opt_params['S']
                  * params['carry']
                  * params['nd1']
                  * opt_params['sigma'] )
                 / (2 * np.sqrt(opt_params['T']))
                + (params['b'] - opt_params['r'])
                * (opt_params['S']
                   * params['carry']
                   * params['minusNd1'])
                + (opt_params['r'] * opt_params['K'])
                * np.exp(-opt_params['r'] * opt_params['T'])
                * params['minusNd2'])
                / 100)

        else:
            raise ValueError(f"Choose option type put or call: {opt_params['option']}")

        return opt_theta


    @staticmethod
    def gamma(
        opt_params: dict,
        params: dict) -> float:
        """
        Sensitivity of delta to changes in the underlying asset price

        Parameters
        ----------
        opt_params : Dict
            S : Float
                Underlying Stock Price. The default is 100.
            K : Float
                Strike Price. The default is 100.
            T : Float
                Time to Maturity. The default is 0.25 (3 months).
            r : Float
                Interest Rate. The default is 0.05 (5%).
            q : Float
                Dividend Yield. The default is 0.
            sigma : Float
                Implied Volatility. The default is 0.2 (20%).
            option : Str
                Option type, Put or Call. The default is 'call'
        params : Dict
            Dictionary of key parameters; used for refreshing distribution.

        Returns
        -------
        Float
            Option Gamma.

        """

        # Update distribution parameters
        params = Utils.refresh_dist_params(
            opt_params=opt_params, params=params)

        opt_gamma = ((params['nd1'] * params['carry'])
                     / (opt_params['S'] * opt_params['sigma']
                        * np.sqrt(opt_params['T'])))

        return opt_gamma


    @staticmethod
    def vega(
        opt_params: dict,
        params: dict) -> float:
        """
        Sensitivity of the option price to changes in volatility

        Parameters
        ----------
        opt_params : Dict
            S : Float
                Underlying Stock Price. The default is 100.
            K : Float
                Strike Price. The default is 100.
            T : Float
                Time to Maturity. The default is 0.25 (3 months).
            r : Float
                Interest Rate. The default is 0.05 (5%).
            q : Float
                Dividend Yield. The default is 0.
            sigma : Float
                Implied Volatility. The default is 0.2 (20%).
            option : Str
                Option type, Put or Call. The default is 'call'
        params : Dict
            Dictionary of key parameters; used for refreshing distribution.

        Returns
        -------
        Float
            Option Vega.

        """

        # Update distribution parameters
        params = Utils.refresh_dist_params(
            opt_params=opt_params, params=params)

        opt_vega = ((opt_params['S'] * params['carry']
                     * params['nd1'] * np.sqrt(opt_params['T'])) / 100)

        return opt_vega


    @staticmethod
    def rho(
        opt_params: dict,
        params: dict) -> float:
        """
        Sensitivity of the option price to changes in the risk free rate

        Parameters
        ----------
        opt_params : Dict
            S : Float
                Underlying Stock Price. The default is 100.
            K : Float
                Strike Price. The default is 100.
            T : Float
                Time to Maturity. The default is 0.25 (3 months).
            r : Float
                Interest Rate. The default is 0.05 (5%).
            q : Float
                Dividend Yield. The default is 0.
            sigma : Float
                Implied Volatility. The default is 0.2 (20%).
            option : Str
                Option type, Put or Call. The default is 'call'
        params : Dict
            Dictionary of key parameters; used for refreshing distribution.

        Returns
        -------
        Float
            Option Rho.

        """

        # Update distribution parameters
        params = Utils.refresh_dist_params(
            opt_params=opt_params, params=params)

        if opt_params['option'] == 'call':
            opt_rho = (
                (opt_params['T'] * opt_params['K']
                 * np.exp(-opt_params['r'] * opt_params['T']) * params['Nd2'])
                / 10000)
        elif opt_params['option'] == 'put':
            opt_rho = (
                (-opt_params['T'] * opt_params['K']
                 * np.exp(-opt_params['r'] * opt_params['T'])
                 * params['minusNd2'])
                / 10000)

        else:
            raise ValueError(f"Choose option type put or call: {opt_params['option']}")

        return opt_rho


    @staticmethod
    def vanna(
        opt_params: dict,
        params: dict) -> float:
        """
        DdeltaDvol, DvegaDspot
        Sensitivity of delta to changes in volatility
        Sensitivity of vega to changes in the asset price

        Parameters
        ----------
        opt_params : Dict
            S : Float
                Underlying Stock Price. The default is 100.
            K : Float
                Strike Price. The default is 100.
            T : Float
                Time to Maturity. The default is 0.25 (3 months).
            r : Float
                Interest Rate. The default is 0.05 (5%).
            q : Float
                Dividend Yield. The default is 0.
            sigma : Float
                Implied Volatility. The default is 0.2 (20%).
            option : Str
                Option type, Put or Call. The default is 'call'
        params : Dict
            Dictionary of key parameters; used for refreshing distribution.

        Returns
        -------
        Float
            Option Vanna.

        """

        # Update distribution parameters
        params = Utils.refresh_dist_params(
            opt_params=opt_params, params=params)

        opt_vanna = (
            (((-params['carry'] * params['d2'])
              / opt_params['sigma']) * params['nd1']) / 100)

        return opt_vanna


    @classmethod
    def vomma(
        cls,
        opt_params: dict,
        params: dict) -> float:
        """
        DvegaDvol, Vega Convexity, Volga, Vol Gamma
        Sensitivity of vega to changes in volatility

        Parameters
        ----------
        opt_params : Dict
            S : Float
                Underlying Stock Price. The default is 100.
            K : Float
                Strike Price. The default is 100.
            T : Float
                Time to Maturity. The default is 0.25 (3 months).
            r : Float
                Interest Rate. The default is 0.05 (5%).
            q : Float
                Dividend Yield. The default is 0.
            sigma : Float
                Implied Volatility. The default is 0.2 (20%).
            option : Str
                Option type, Put or Call. The default is 'call'
        params : Dict
            Dictionary of key parameters; used for refreshing distribution.

        Returns
        -------
        Float
            Option Vomma.

        """

        # Update distribution parameters
        params = Utils.refresh_dist_params(
            opt_params=opt_params, params=params)

        opt_vomma = (
            (cls.vega(opt_params, params) * (
                (params['d1'] * params['d2']) / (opt_params['sigma']))) / 100)

        return opt_vomma


    @staticmethod
    def charm(
        opt_params: dict,
        params: dict) -> float:
        """
        DdeltaDtime, Delta Bleed
        Sensitivity of delta to changes in time to maturity

        Parameters
        ----------
        opt_params : Dict
            S : Float
                Underlying Stock Price. The default is 100.
            K : Float
                Strike Price. The default is 100.
            T : Float
                Time to Maturity. The default is 0.25 (3 months).
            r : Float
                Interest Rate. The default is 0.05 (5%).
            q : Float
                Dividend Yield. The default is 0.
            sigma : Float
                Implied Volatility. The default is 0.2 (20%).
            option : Str
                Option type, Put or Call. The default is 'call'
        params : Dict
            Dictionary of key parameters; used for refreshing distribution.

        Returns
        -------
        Float
            Option Charm.

        """

        # Update distribution parameters
        params = Utils.refresh_dist_params(
            opt_params=opt_params, params=params)

        if opt_params['option'] == 'call':
            opt_charm = (
                (-params['carry'] * ((params['nd1'] * (
                    (params['b'] / (opt_params['sigma']
                                    * np.sqrt(opt_params['T'])))
                    - (params['d2'] / (2 * opt_params['T']))))
                    + ((params['b'] - opt_params['r']) * params['Nd1'])))
                / 100)
        elif opt_params['option'] == 'put':
            opt_charm = (
                (-params['carry'] * (
                    (params['nd1'] * (
                        (params['b']
                         / (opt_params['sigma'] * np.sqrt(opt_params['T'])))
                        - (params['d2'] / (2 * opt_params['T']))))
                    - ((params['b'] - opt_params['r']) * params['minusNd1'])))
                / 100)

        else:
            raise ValueError(f"Choose option type put or call: {opt_params['option']}")

        return opt_charm


    @classmethod
    def zomma(
        cls,
        opt_params: dict,
        params: dict) -> float:
        """
        DgammaDvol
        Sensitivity of gamma to changes in volatility

        Parameters
        ----------
        opt_params : Dict
            S : Float
                Underlying Stock Price. The default is 100.
            K : Float
                Strike Price. The default is 100.
            T : Float
                Time to Maturity. The default is 0.25 (3 months).
            r : Float
                Interest Rate. The default is 0.05 (5%).
            q : Float
                Dividend Yield. The default is 0.
            sigma : Float
                Implied Volatility. The default is 0.2 (20%).
            option : Str
                Option type, Put or Call. The default is 'call'
        params : Dict
            Dictionary of key parameters; used for refreshing distribution.

        Returns
        -------
        Float
            Option Zomma.

        """

        # Update distribution parameters
        params = Utils.refresh_dist_params(
            opt_params=opt_params, params=params)

        opt_zomma = (
            (cls.gamma(opt_params, params) * (
                (params['d1'] * params['d2'] - 1)
                / opt_params['sigma'])) / 100)

        return opt_zomma


    @classmethod
    def speed(
        cls,
        opt_params: dict,
        params: dict) -> float:
        """
        DgammaDspot
        Sensitivity of gamma to changes in asset price
        3rd derivative of option price with respect to spot

        Parameters
        ----------
        opt_params : Dict
            S : Float
                Underlying Stock Price. The default is 100.
            K : Float
                Strike Price. The default is 100.
            T : Float
                Time to Maturity. The default is 0.25 (3 months).
            r : Float
                Interest Rate. The default is 0.05 (5%).
            q : Float
                Dividend Yield. The default is 0.
            sigma : Float
                Implied Volatility. The default is 0.2 (20%).
            option : Str
                Option type, Put or Call. The default is 'call'
        params : Dict
            Dictionary of key parameters; used for refreshing distribution.

        Returns
        -------
        Float
            Option Speed.

        """

        # Update distribution parameters
        params = Utils.refresh_dist_params(
            opt_params=opt_params, params=params)

        opt_speed = -(
            cls.gamma(opt_params, params) * (1 + (
                params['d1'] / (opt_params['sigma']
                                * np.sqrt(opt_params['T']))))
            / opt_params['S'])

        return opt_speed


    @classmethod
    def color(
        cls,
        opt_params: dict,
        params: dict) -> float:
        """
        DgammaDtime, Gamma Bleed, Gamma Theta
        Sensitivity of gamma to changes in time to maturity

        Parameters
        ----------
        opt_params : Dict
            S : Float
                Underlying Stock Price. The default is 100.
            K : Float
                Strike Price. The default is 100.
            T : Float
                Time to Maturity. The default is 0.25 (3 months).
            r : Float
                Interest Rate. The default is 0.05 (5%).
            q : Float
                Dividend Yield. The default is 0.
            sigma : Float
                Implied Volatility. The default is 0.2 (20%).
            option : Str
                Option type, Put or Call. The default is 'call'
        params : Dict
            Dictionary of key parameters; used for refreshing distribution.

        Returns
        -------
        Float
            Option Color.

        """

        # Update distribution parameters
        params = Utils.refresh_dist_params(
            opt_params=opt_params, params=params)

        opt_color = (
            (cls.gamma(opt_params, params) * (
                (opt_params['r'] - params['b']) + (
                    (params['b'] * params['d1'])
                    / (opt_params['sigma'] * np.sqrt(opt_params['T'])))
                + ((1 - params['d1'] * params['d2']) / (2 * opt_params['T']))))
            / 100)

        return opt_color


    @classmethod
    def ultima(
        cls,
        opt_params: dict,
        params: dict) -> float:
        """
        DvommaDvol
        Sensitivity of vomma to changes in volatility
        3rd derivative of option price wrt volatility

        Parameters
        ----------
        opt_params : Dict
            S : Float
                Underlying Stock Price. The default is 100.
            K : Float
                Strike Price. The default is 100.
            T : Float
                Time to Maturity. The default is 0.25 (3 months).
            r : Float
                Interest Rate. The default is 0.05 (5%).
            q : Float
                Dividend Yield. The default is 0.
            sigma : Float
                Implied Volatility. The default is 0.2 (20%).
            option : Str
                Option type, Put or Call. The default is 'call'
        params : Dict
            Dictionary of key parameters; used for refreshing distribution.

        Returns
        -------
        Float
            Option Ultima.

        """

        # Update distribution parameters
        params = Utils.refresh_dist_params(
            opt_params=opt_params, params=params)

        opt_ultima = (
            (cls.vomma(opt_params, params) * (
                (1 / opt_params['sigma']) * (params['d1'] * params['d2']
                                         - (params['d1'] / params['d2'])
                                         - (params['d2'] / params['d1']) - 1)))
            / 100)

        return opt_ultima


    @classmethod
    def vega_bleed(
        cls,
        opt_params: dict,
        params: dict) -> float:
        """
        DvegaDtime
        Sensitivity of vega to changes in time to maturity.

        Parameters
        ----------
        opt_params : Dict
            S : Float
                Underlying Stock Price. The default is 100.
            K : Float
                Strike Price. The default is 100.
            T : Float
                Time to Maturity. The default is 0.25 (3 months).
            r : Float
                Interest Rate. The default is 0.05 (5%).
            q : Float
                Dividend Yield. The default is 0.
            sigma : Float
                Implied Volatility. The default is 0.2 (20%).
            option : Str
                Option type, Put or Call. The default is 'call'
        params : Dict
            Dictionary of key parameters; used for refreshing distribution.

        Returns
        -------
        Float
            Option Vega Bleed.

        """

        # Update distribution parameters
        params = Utils.refresh_dist_params(
            opt_params=opt_params, params=params)

        opt_vega_bleed = (
            (cls.vega(opt_params, params)
             * (opt_params['r']
                - params['b']
                + ((params['b'] * params['d1'])
                   / (opt_params['sigma'] * np.sqrt(opt_params['T'])))
                - ((1 + (params['d1'] * params['d2']) )
                   / (2 * opt_params['T']))))
            / 100)

        return opt_vega_bleed


    @classmethod
    def return_options(
        cls,
        opt_dict: dict,
        params: dict) -> dict:

        """
        Calculate option prices to be used in payoff diagrams.

        Parameters
        ----------
        opt_dict : Dict
            Dictionary of option pricing parameters
        params : Dict
            Dictionary of key parameters

        Returns
        -------
        From 1 to 4 sets of option values:
            Cx_0: Current option price; Float.
            Cx: Terminal Option payoff, varying by strike; Array
            Cx_G: Current option value, varying by strike; Array

        """

        # Dictionary to store option legs
        option_legs = {}

        # create array of 1000 equally spaced points between 75% of
        # initial underlying price and 125%
        option_legs['SA'] = np.linspace(
            0.75 * opt_dict['S'], 1.25 * opt_dict['S'], 1000)

        opt_params = {
            'S':opt_dict['S'],
            'K':opt_dict['K1'],
            'T':opt_dict['T1'],
            'r':opt_dict['r'],
            'q':opt_dict['q'],
            'sigma':opt_dict['sigma'],
            'option':opt_dict['option1'],
            }

        # Calculate the current price of option 1
        option_legs['C1_0'] = cls.price(opt_params=opt_params, params=params)

        # Calculate the prices at maturity for the range of strikes
        # in SA of option 1
        change_params = {'S':option_legs['SA'], 'T':0}
        opt_params.update(change_params)

        option_legs['C1'] = cls.price(opt_params=opt_params, params=params)

        # Calculate the current prices for the range of strikes
        # in SA of option 1
        change_params = {'T':opt_dict['T1']}
        opt_params.update(change_params)

        option_legs['C1_G'] = cls.price(opt_params=opt_params, params=params)

        if opt_dict['legs'] > 1:
            # Calculate the current price of option 2
            change_params = {'S':opt_dict['S'],
                             'K':opt_dict['K2'],
                             'T':opt_dict['T2'],
                             'option':opt_dict['option2']}
            opt_params.update(change_params)

            option_legs['C2_0'] = cls.price(
                opt_params=opt_params, params=params)

            # Calculate the prices at maturity for the range of strikes
            # in SA of option 2
            change_params = {'S':option_legs['SA'], 'T':0}
            opt_params.update(change_params)

            option_legs['C2'] = cls.price(opt_params=opt_params, params=params)

            # Calculate the current prices for the range of strikes
            # in SA of option 2
            change_params = {'T':opt_dict['T2']}
            opt_params.update(change_params)

            option_legs['C2_G'] = cls.price(
                opt_params=opt_params, params=params)

        if opt_dict['legs'] > 2:
            # Calculate the current price of option 3
            change_params = {'S':opt_dict['S'],
                             'K':opt_dict['K3'],
                             'T':opt_dict['T3'],
                             'option':opt_dict['option3']}
            opt_params.update(change_params)

            option_legs['C3_0'] = cls.price(
                opt_params=opt_params, params=params)

            # Calculate the prices at maturity for the range of strikes
            # in SA of option 3
            change_params = {'S':option_legs['SA'], 'T':0}
            opt_params.update(change_params)

            option_legs['C3'] = cls.price(opt_params=opt_params, params=params)

            # Calculate the current prices for the range of strikes
            # in SA of option 3
            change_params = {'T':opt_dict['T3']}
            opt_params.update(change_params)

            option_legs['C3_G'] = cls.price(
                opt_params=opt_params, params=params)

        if opt_dict['legs'] > 3:
            # Calculate the current price of option 4
            change_params = {'S':opt_dict['S'],
                             'K':opt_dict['K4'],
                             'T':opt_dict['T4'],
                             'option':opt_dict['option4']}
            opt_params.update(change_params)

            option_legs['C4_0'] = cls.price(
                opt_params=opt_params, params=params)

            # Calculate the prices at maturity for the range of strikes
            # in SA of option 4
            change_params = {'S':option_legs['SA'], 'T':0}
            opt_params.update(change_params)

            option_legs['C4'] = cls.price(
                opt_params=opt_params, params=params)

            # Calculate the current prices for the range of strikes
            # in SA of option 4
            change_params = {'T':opt_dict['T4']}
            opt_params.update(change_params)

            option_legs['C4_G'] = cls.price(
                opt_params=opt_params, params=params)

        return option_legs
