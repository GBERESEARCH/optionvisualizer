"""
One and Two Option Payoffs

"""
import numpy as np
from optionvisualizer.multi_payoffs import MultiPayoff
from optionvisualizer.option_formulas import Option
# pylint: disable=invalid-name

class SimplePayoff():
    """
    Calculate One and Two Option payoffs - Call / Put / Spread / Straddle etc.

    """

    @staticmethod
    def call(params):
        """
        Displays the graph of the call.

        Parameters
        ----------
        params : Dict
            S : Float
                 Underlying Stock Price. The default is 100.
            K : Float
                Strike Price of option 1. The default is 100.
            T : Float
                Time to Maturity. The default is 0.25 (3 months).
            r : Float
                Interest Rate. The default is 0.05 (5%).
            q : Float
                Dividend Yield. The default is 0.
            sigma : Float
                Implied Volatility. The default is 0.2 (20%).
            direction : Str
                Whether the payoff is long or short. The default is 'long'.
            value : Bool
                Whether to show the current value as well as the terminal
                payoff. The default is False.
            mpl_style : Str
                Matplotlib style template for 2D risk charts and payoffs.
                The default is 'seaborn-darkgrid'.

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.

        """

        params['opt_dict'] = {
            'legs':1,
            'S':params['S'],
            'K1':params['K'],
            'T1':params['T'],
            'r':params['r'],
            'q':params['q'],
            'sigma':params['sigma'],
            'option1':'call'
            }

        # Calculate option prices
        option_legs = Option.return_options(
            opt_dict=params['opt_dict'], params=params)

        # Create payoff based on direction
        if params['direction'] == 'long':
            payoff = option_legs['C1'] - option_legs['C1_0']
            title = 'Long Call'
            if params['value']:
                payoff2 = option_legs['C1_G'] - option_legs['C1_0']
            else:
                payoff2 = None

        if params['direction'] == 'short':
            payoff = -option_legs['C1'] + option_legs['C1_0']
            title = 'Short Call'
            if params['value']:
                payoff2 = -option_legs['C1_G'] + option_legs['C1_0']
            else:
                payoff2 = None

        params['payoff_dict'] = {
            'S':params['S'],
            'SA':option_legs['SA'],
            'payoff':payoff,
            'title':title,
            'payoff2':payoff2,
            'size2d':params['size2d'],
            'mpl_style':params['mpl_style']
            }

        # Visualize payoff
        return MultiPayoff.vis_payoff(
            payoff_dict=params['payoff_dict'], params=params)


    @staticmethod
    def put(params):
        """
        Displays the graph of the put.

        Parameters
        ----------
        params : Dict
            S : Float
                 Underlying Stock Price. The default is 100.
            K : Float
                Strike Price of option 1. The default is 100.
            T : Float
                Time to Maturity. The default is 0.25 (3 months).
            r : Float
                Interest Rate. The default is 0.05 (5%).
            q : Float
                Dividend Yield. The default is 0.
            sigma : Float
                Implied Volatility. The default is 0.2 (20%).
            direction : Str
                Whether the payoff is long or short. The default is 'long'.
            value : Bool
                Whether to show the current value as well as the terminal
                payoff. The default is False.
            mpl_style : Str
                Matplotlib style template for 2D risk charts and payoffs.
                The default is 'seaborn-darkgrid'.

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.

        """

        params['opt_dict'] = {
            'legs':1,
            'S':params['S'],
            'K1':params['K'],
            'T1':params['T'],
            'r':params['r'],
            'q':params['q'],
            'sigma':params['sigma'],
            'option1':'put'
            }

        # Calculate option prices
        option_legs = Option.return_options(
            opt_dict=params['opt_dict'], params=params)

        # Create payoff based on direction
        if params['direction'] == 'long':
            payoff = option_legs['C1'] - option_legs['C1_0']
            title = 'Long Put'
            if params['value']:
                payoff2 = option_legs['C1_G'] - option_legs['C1_0']
            else:
                payoff2 = None

        if params['direction'] == 'short':
            payoff = -option_legs['C1'] + option_legs['C1_0']
            title = 'Short Put'
            if params['value']:
                payoff2 = -option_legs['C1_G'] + option_legs['C1_0']
            else:
                payoff2 = None

        params['payoff_dict'] = {
            'S':params['S'],
            'SA':option_legs['SA'],
            'payoff':payoff,
            'title':title,
            'payoff2':payoff2,
            'size2d':params['size2d'],
            'mpl_style':params['mpl_style']
            }

        # Visualize payoff
        return MultiPayoff.vis_payoff(
            payoff_dict=params['payoff_dict'], params=params)


    @staticmethod
    def stock(params):
        """
        Displays the graph of the underlying.

        Parameters
        ----------
        S : Float
             Underlying Stock Price. The default is 100.
        direction : Str
            Whether the payoff is long or short. The default is 'long'.
        mpl_style : Str
            Matplotlib style template for 2D risk charts and payoffs.
            The default is 'seaborn-darkgrid'.

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.

        """

        # Define strike range
        SA = np.linspace(0.75 * params['S'], 1.25 * params['S'], 1000)

        # Create payoff based on option type
        if params['direction'] == 'long':
            payoff = SA - params['S']
            title = 'Long Stock'

        if params['direction'] == 'short':
            payoff = params['S'] - SA
            title = 'Short Stock'

        params['payoff_dict'] = {
            'S':params['S'],
            'SA':SA,
            'payoff':payoff,
            'title':title,
            'payoff2':None,
            'size2d':params['size2d'],
            'mpl_style':params['mpl_style']
            }

        # Visualize payoff
        return MultiPayoff.vis_payoff(
            payoff_dict=params['payoff_dict'], params=params)


    @staticmethod
    def forward(params):
        """
        Displays the graph of the synthetic forward strategy:
            Long one ATM call
            Short one ATM put

        Parameters
        ----------
        S : Float
             Underlying Stock Price. The default is 100.
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        direction : Str
            Whether the payoff is long or short. The default is 'long'.
        cash : Bool
            Whether to discount to present value. The default is False.
        mpl_style : Str
            Matplotlib style template for 2D risk charts and payoffs.
            The default is 'seaborn-darkgrid'.

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.

        """

        params['opt_dict'] = {
            'legs':2,
            'S':params['S'],
            'K1':params['S'],
            'T1':params['T'],
            'K2':params['S'],
            'T2':params['T'],
            'r':params['r'],
            'q':params['q'],
            'sigma':params['sigma'],
            'option1':'call',
            'option2':'put'
            }

        # Calculate option prices
        option_legs = Option.return_options(
            opt_dict=params['opt_dict'], params=params)

        # Whether to discount the payoff
        if params['cash']:
            pv = np.exp(-params['r'] * params['T'])
        else:
            pv = 1

        # Create payoff based on option type
        if params['direction'] == 'long':
            payoff = ((option_legs['C1'] - option_legs['C2']
                      - option_legs['C1_0'] + option_legs['C2_0'])
                      * pv)
            title = 'Long Forward'

        if params['direction'] == 'short':
            payoff = ((-option_legs['C1'] + option_legs['C2']
                       + option_legs['C1_0'] - option_legs['C2_0'])
                      * pv)
            title = 'Short Forward'

        params['payoff_dict'] = {
            'S':params['S'],
            'SA':option_legs['SA'],
            'payoff':payoff,
            'title':title,
            'payoff2':None,
            'size2d':params['size2d'],
            'mpl_style':params['mpl_style']
            }

        # Visualize payoff
        return MultiPayoff.vis_payoff(
            payoff_dict=params['payoff_dict'], params=params)


    @staticmethod
    def collar(params):
        """
        Displays the graph of the collar strategy:
            Long one OTM put
            Short one OTM call

        Parameters
        ----------
        S : Float
             Underlying Stock Price. The default is 100.
        K1 : Float
             Strike Price of option 1. The default is 98.
        K2 : Float
             Strike Price of option 2. The default is 102.
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        direction : Str
            Whether the payoff is long or short. The default is 'long'.
        value : Bool
            Whether to show the current value as well as the terminal
            payoff. The default is False.
        mpl_style : Str
            Matplotlib style template for 2D risk charts and payoffs.
            The default is 'seaborn-darkgrid'.

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.

        """

        params['opt_dict'] = {
            'legs':2,
            'S':params['S'],
            'K1':params['K1'],
            'T1':params['T'],
            'K2':params['K2'],
            'T2':params['T'],
            'r':params['r'],
            'q':params['q'],
            'sigma':params['sigma'],
            'option1':'put',
            'option2':'call'
            }

        # Calculate option prices
        option_legs = Option.return_options(
            opt_dict=params['opt_dict'], params=params)

        # Create payoff based on option type
        if params['direction'] == 'long':
            payoff = (option_legs['SA'] - params['S']
                      + option_legs['C1'] - option_legs['C2']
                      - option_legs['C1_0'] + option_legs['C2_0'])
            title = 'Long Collar'

            if params['value']:
                payoff2 = (option_legs['SA'] - params['S']
                           + option_legs['C1_G'] - option_legs['C2_G']
                           - option_legs['C1_0'] + option_legs['C2_0'])
            else:
                payoff2 = None

        if params['direction'] == 'short':
            payoff = (-option_legs['SA'] + params['S']
                      - option_legs['C1'] + option_legs['C2']
                      + option_legs['C1_0'] - option_legs['C2_0'])
            title = 'Short Collar'

            if params['value']:
                payoff2 = (-option_legs['SA'] + params['S']
                           - option_legs['C1_G'] + option_legs['C2_G']
                           + option_legs['C1_0'] - option_legs['C2_0'])
            else:
                payoff2 = None

        params['payoff_dict'] = {
            'S':params['S'],
            'SA':option_legs['SA'],
            'payoff':payoff,
            'title':title,
            'payoff2':payoff2,
            'size2d':params['size2d'],
            'mpl_style':params['mpl_style']
            }

        # Visualize payoff
        return MultiPayoff.vis_payoff(
            payoff_dict=params['payoff_dict'], params=params)


    @staticmethod
    def spread(params):
        """
        Displays the graph of the spread strategy:
            Long one ITM option
            Short one OTM option

        Parameters
        ----------
        S : Float
             Underlying Stock Price. The default is 100.
        K1 : Float
             Strike Price of option 1. The default is 95.
        K2 : Float
             Strike Price of option 2. The default is 105.
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
        direction : Str
            Whether the payoff is long or short. The default is 'long'.
        value : Bool
            Whether to show the current value as well as the terminal
            payoff. The default is False.
        mpl_style : Str
            Matplotlib style template for 2D risk charts and payoffs.
            The default is 'seaborn-darkgrid'.

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.


        """

        params['opt_dict'] = {
            'legs':2,
            'S':params['S'],
            'K1':params['K1'],
            'T1':params['T'],
            'K2':params['K2'],
            'T2':params['T'],
            'r':params['r'],
            'q':params['q'],
            'sigma':params['sigma'],
            'option1':params['option'],
            'option2':params['option']
            }

        # Calculate option prices
        option_legs = Option.return_options(
            opt_dict=params['opt_dict'], params=params)

        # Create payoff based on option type
        if params['direction'] == 'long':
            payoff = (option_legs['C1'] - option_legs['C2']
                      - option_legs['C1_0'] + option_legs['C2_0'])
            if params['value']:
                payoff2 = (option_legs['C1_G'] - option_legs['C2_G']
                           - option_legs['C1_0'] + option_legs['C2_0'])
            else:
                payoff2 = None

        if params['direction'] == 'short':
            payoff = (-option_legs['C1'] + option_legs['C2']
                      + option_legs['C1_0'] - option_legs['C2_0'])
            if params['value']:
                payoff2 = (-option_legs['C1_G'] + option_legs['C2_G']
                           + option_legs['C1_0'] - option_legs['C2_0'])
            else:
                payoff2 = None

        # Create title based on option type and direction
        if params['option'] == 'call' and params['direction'] == 'long':
            title = 'Bull Call Spread'
        if params['option'] == 'put' and params['direction'] == 'long':
            title = 'Bull Put Spread'
        if params['option'] == 'call' and params['direction'] == 'short':
            title = 'Bear Call Spread'
        if params['option'] == 'put' and params['direction'] == 'short':
            title = 'Bear Put Spread'

        params['payoff_dict'] = {
            'S':params['S'],
            'SA':option_legs['SA'],
            'payoff':payoff,
            'title':title,
            'payoff2':payoff2,
            'size2d':params['size2d'],
            'mpl_style':params['mpl_style']
            }

        # Visualize payoff
        return MultiPayoff.vis_payoff(
            payoff_dict=params['payoff_dict'], params=params)


    @staticmethod
    def backspread(params):
        """
        Displays the graph of the backspread strategy:
            Short one ITM option
            Long ratio * OTM options

        Parameters
        ----------
        S : Float
             Underlying Stock Price. The default is 100.
        K1 : Float
             Strike Price of option 1. The default is 95.
        K2 : Float
             Strike Price of option 2. The default is 105.
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
        ratio : Int
            Multiple of OTM options to be sold for ITM purchased. The
            default is 2.
        value : Bool
            Whether to show the current value as well as the terminal
            payoff. The default is False.
        mpl_style : Str
            Matplotlib style template for 2D risk charts and payoffs.
            The default is 'seaborn-darkgrid'.

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.

        """

        params['opt_dict'] = {
            'legs':2,
            'S':params['S'],
            'K1':params['K1'],
            'T1':params['T'],
            'K2':params['K2'],
            'T2':params['T'],
            'r':params['r'],
            'q':params['q'],
            'sigma':params['sigma'],
            'option1':params['option'],
            'option2':params['option']
            }

        # Calculate option prices
        option_legs = Option.return_options(
            opt_dict=params['opt_dict'], params=params)

        # Create payoff based on option type
        if params['option'] == 'call':
            title = 'Call Backspread'
            payoff = (-option_legs['C1'] + (params['ratio']
                                            * option_legs['C2'])
                      + option_legs['C1_0'] - (params['ratio']
                                               * option_legs['C2_0']))
            if params['value']:
                payoff2 = (-option_legs['C1_G'] + (params['ratio']
                                                   * option_legs['C2_G'])
                           + option_legs['C1_0'] - (params['ratio']
                                                    * option_legs['C2_0']))
            else:
                payoff2 = None

        if params['option'] == 'put':
            payoff = (params['ratio'] * option_legs['C1']
                      - option_legs['C2']
                      - params['ratio'] * option_legs['C1_0']
                      + option_legs['C2_0'])
            title = 'Put Backspread'
            if params['value']:
                payoff2 = (params['ratio'] * option_legs['C1_G']
                           - option_legs['C2_G']
                           - params['ratio'] * option_legs['C1_0']
                           + option_legs['C2_0'])
            else:
                payoff2 = None

        params['payoff_dict'] = {
            'S':params['S'],
            'SA':option_legs['SA'],
            'payoff':payoff,
            'title':title,
            'payoff2':payoff2,
            'size2d':params['size2d'],
            'mpl_style':params['mpl_style']
            }

        # Visualize payoff
        return MultiPayoff.vis_payoff(
            payoff_dict=params['payoff_dict'], params=params)


    @staticmethod
    def ratio_vertical_spread(params):
        """
        Displays the graph of the ratio vertical spread strategy:
            Long one ITM option
            Short ratio * OTM options

        Parameters
        ----------
        S : Float
             Underlying Stock Price. The default is 100.
        K1 : Float
             Strike Price of option 1. The default is 95.
        K2 : Float
             Strike Price of option 2. The default is 105.
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
        ratio : Int
            Multiple of OTM options to be sold for ITM purchased. The
            default is 2.
        direction : Str
            Whether the payoff is long or short. The default is 'long'.
        value : Bool
            Whether to show the current value as well as the terminal
            payoff. The default is False.
        mpl_style : Str
            Matplotlib style template for 2D risk charts and payoffs.
            The default is 'seaborn-darkgrid'.

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.

        """

        params['opt_dict'] = {
            'legs':2,
            'S':params['S'],
            'K1':params['K1'],
            'T1':params['T'],
            'K2':params['K2'],
            'T2':params['T'],
            'r':params['r'],
            'q':params['q'],
            'sigma':params['sigma'],
            'option1':params['option'],
            'option2':params['option']
            }

        # Calculate option prices
        option_legs = Option.return_options(
            opt_dict=params['opt_dict'], params=params)

        # Create payoff based on option type
        if params['option'] == 'call':
            title = 'Call Ratio Vertical Spread'
            payoff = (option_legs['C1']
                      - params['ratio'] * option_legs['C2']
                      - option_legs['C1_0']
                      + params['ratio'] * option_legs['C2_0'])
            if params['value']:
                payoff2 = (option_legs['C1_G']
                           - params['ratio'] * option_legs['C2_G']
                           - option_legs['C1_0']
                           + params['ratio'] * option_legs['C2_0'])
            else:
                payoff2 = None

        if params['option'] == 'put':
            title = 'Put Ratio Vertical Spread'
            payoff = (-params['ratio'] * option_legs['C1']
                      + option_legs['C2']
                      + params['ratio'] * option_legs['C1_0']
                      - option_legs['C2_0'])
            if params['value']:
                payoff2 = (-params['ratio'] * option_legs['C1_G']
                           + option_legs['C2_G']
                           + params['ratio'] * option_legs['C1_0']
                           - option_legs['C2_0'])
            else:
                payoff2 = None

        params['payoff_dict'] = {
            'S':params['S'],
            'SA':option_legs['SA'],
            'payoff':payoff,
            'title':title,
            'payoff2':payoff2,
            'size2d':params['size2d'],
            'mpl_style':params['mpl_style']
            }

        # Visualize payoff
        return MultiPayoff.vis_payoff(
            payoff_dict=params['payoff_dict'], params=params)


    @staticmethod
    def straddle(params):
        """
        Displays the graph of the straddle strategy:
            Long one ATM put
            Long one ATM call

        Parameters
        ----------
        S : Float
             Underlying Stock Price. The default is 100.
        K : Float
            Strike Price of options 1 and 2. The default is 100.
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        direction : Str
            Whether the payoff is long or short. The default is 'long'.
        value : Bool
            Whether to show the current value as well as the terminal
            payoff. The default is False.
        mpl_style : Str
            Matplotlib style template for 2D risk charts and payoffs.
            The default is 'seaborn-darkgrid'.

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.

        """

        params['opt_dict'] = {
            'legs':2,
            'S':params['S'],
            'K1':params['K'],
            'T1':params['T'],
            'K2':params['K'],
            'T2':params['T'],
            'r':params['r'],
            'q':params['q'],
            'sigma':params['sigma'],
            'option1':'put',
            'option2':'call'
            }

        # Calculate option prices
        option_legs = Option.return_options(
            opt_dict=params['opt_dict'], params=params)

        # Create payoff based on direction
        if params['direction'] == 'long':
            payoff = (option_legs['C1'] + option_legs['C2']
                      - option_legs['C1_0'] - option_legs['C2_0'])
            title = 'Long Straddle'
            if params['value']:
                payoff2 = (option_legs['C1_G'] + option_legs['C2_G']
                           - option_legs['C1_0'] - option_legs['C2_0'])
            else:
                payoff2 = None

        if params['direction'] == 'short':
            payoff = (-option_legs['C1'] - option_legs['C2']
                      + option_legs['C1_0'] + option_legs['C2_0'])
            title = 'Short Straddle'
            if params['value']:
                payoff2 = (-option_legs['C1_G'] - option_legs['C2_G']
                           + option_legs['C1_0'] + option_legs['C2_0'])
            else:
                payoff2 = None

        params['payoff_dict'] = {
            'S':params['S'],
            'SA':option_legs['SA'],
            'payoff':payoff,
            'title':title,
            'payoff2':payoff2,
            'size2d':params['size2d'],
            'mpl_style':params['mpl_style']
            }

        # Visualize payoff
        return MultiPayoff.vis_payoff(
            payoff_dict=params['payoff_dict'], params=params)


    @staticmethod
    def strangle(params):
        """
        Displays the graph of the strangle strategy:
            Long one OTM put
            Long one OTM call

        Parameters
        ----------
        S : Float
             Underlying Stock Price. The default is 100.
        K1 : Float
             Strike Price of option 1. The default is 95.
        K2 : Float
             Strike Price of option 2. The default is 105.
        T : Float
            Time to Maturity. The default is 0.25 (3 months).
        r : Float
            Interest Rate. The default is 0.05 (5%).
        q : Float
            Dividend Yield. The default is 0.
        sigma : Float
            Implied Volatility. The default is 0.2 (20%).
        direction : Str
            Whether the payoff is long or short. The default is 'long'.
        value : Bool
            Whether to show the current value as well as the terminal
            payoff. The default is False.
        mpl_style : Str
            Matplotlib style template for 2D risk charts and payoffs.
            The default is 'seaborn-darkgrid'.

        Returns
        -------
        payoff: Terminal payoff value less initial cost; Array
        payoff2: Current payoff value less initial cost (optional); Array
        title: Description of payoff; Str
        Runs method to graph using Matplotlib.

        """

        params['opt_dict'] = {
            'legs':2,
            'S':params['S'],
            'K1':params['K1'],
            'T1':params['T'],
            'K2':params['K2'],
            'T2':params['T'],
            'r':params['r'],
            'q':params['q'],
            'sigma':params['sigma'],
            'option1':'put',
            'option2':'call'
            }

        # Calculate option prices
        option_legs = Option.return_options(
            opt_dict=params['opt_dict'], params=params)

        # Create payoff based on direction
        if params['direction'] == 'long':
            payoff = (option_legs['C1'] + option_legs['C2']
                      - option_legs['C1_0'] - option_legs['C2_0'])
            title = 'Long Strangle'
            if params['value']:
                payoff2 = (option_legs['C1_G'] + option_legs['C2_G']
                           - option_legs['C1_0'] - option_legs['C2_0'])
            else:
                payoff2 = None

        if params['direction'] == 'short':
            payoff = (-option_legs['C1'] - option_legs['C2']
                      + option_legs['C1_0'] + option_legs['C2_0'])
            title = 'Short Strangle'
            if params['value']:
                payoff2 = (-option_legs['C1_G'] - option_legs['C2_G']
                           + option_legs['C1_0'] + option_legs['C2_0'])
            else:
                payoff2 = None

        params['payoff_dict'] = {
            'S':params['S'],
            'SA':option_legs['SA'],
            'payoff':payoff,
            'title':title,
            'payoff2':payoff2,
            'size2d':params['size2d'],
            'mpl_style':params['mpl_style']
            }

        # Visualize payoff
        return MultiPayoff.vis_payoff(
            payoff_dict=params['payoff_dict'], params=params)
