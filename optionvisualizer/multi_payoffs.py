"""
Three and Four Option Payoffs

"""
from matplotlib import gridspec
#from matplotlib import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.offline import plot
from optionvisualizer.option_formulas import Option
# pylint: disable=invalid-name

class MultiPayoff():
    """
    Calculate Three and Four Option payoffs - Butterfly / Christmas Tree /
    CondorPut etc.

    """
    @classmethod
    def butterfly(
        cls,
        params: dict) -> go.Figure | None:
        """
        Displays the graph of the butterfly strategy:
            Long one ITM option
            Short two ATM options
            Long one OTM option

        Parameters
        ----------
        S : Float
             Underlying Stock Price. The default is 100.
        K1 : Float
             Strike Price of option 1. The default is 95.
        K2 : Float
             Strike Price of option 2. The default is 100.
        K3 : Float
             Strike Price of option 3. The default is 105.
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
            'legs':3,
            'S':params['S'],
            'K1':params['K1'],
            'T1':params['T'],
            'K2':params['K2'],
            'T2':params['T'],
            'K3':params['K3'],
            'T3':params['T'],
            'r':params['r'],
            'q':params['q'],
            'sigma':params['sigma'],
            'option1':params['option'],
            'option2':params['option'],
            'option3':params['option'],
            }

        # Calculate option prices
        option_legs = Option.return_options(
            opt_dict=params['opt_dict'], params=params)

        # Create payoff based on direction
        if params['direction'] == 'long':
            payoff = (option_legs['C1'] - 2 * option_legs['C2']
                      + option_legs['C3'] - option_legs['C1_0']
                      + 2 * option_legs['C2_0'] - option_legs['C3_0'])
            if params['value']:
                payoff2 = (option_legs['C1_G'] - 2 * option_legs['C2_G']
                           + option_legs['C3_G'] - option_legs['C1_0']
                           + 2 * option_legs['C2_0'] - option_legs['C3_0'])
            else:
                payoff2 = None

        if params['direction'] == 'short':
            payoff = (-option_legs['C1'] + 2 * option_legs['C2']
                      - option_legs['C3'] + option_legs['C1_0']
                      - 2 * option_legs['C2_0'] + option_legs['C3_0'])
            if params['value']:
                payoff2 = (-option_legs['C1_G'] + 2 * option_legs['C2_G']
                           - option_legs['C3_G'] + option_legs['C1_0']
                           - 2 * option_legs['C2_0'] + option_legs['C3_0'])
            else:
                payoff2 = None

        # Create title based on option type and direction
        if params['option'] == 'call' and params['direction'] == 'long':
            title = 'Long Butterfly with Calls'
        if params['option'] == 'put' and params['direction'] == 'long':
            title = 'Long Butterfly with Puts'
        if params['option'] == 'call' and params['direction'] == 'short':
            title = 'Short Butterfly with Calls'
        if params['option'] == 'put' and params['direction'] == 'short':
            title = 'Short Butterfly with Puts'

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
        if (
            params['interactive'] and 
            params['notebook'] and 
            params['web_graph']
            ):
            fig = cls.vis_payoff(
            payoff_dict=params['payoff_dict'], params=params)
            return fig

        return cls.vis_payoff(
            payoff_dict=params['payoff_dict'], params=params)

    @classmethod
    def christmas_tree(
        cls,
        params: dict) -> go.Figure | None:
        """
        Displays the graph of the christmas tree strategy:
            Long one ITM option
            Short one ATM option
            Short one OTM option

        Parameters
        ----------
        S : Float
             Underlying Stock Price. The default is 100.
        K1 : Float
             Strike Price of option 1. The default is 95.
        K2 : Float
             Strike Price of option 2. The default is 100.
        K3 : Float
             Strike Price of option 3. The default is 105.
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
            'legs':3,
            'S':params['S'],
            'K1':params['K1'],
            'T1':params['T'],
            'K2':params['K2'],
            'T2':params['T'],
            'K3':params['K3'],
            'T3':params['T'],
            'r':params['r'],
            'q':params['q'],
            'sigma':params['sigma'],
            'option1':params['option'],
            'option2':params['option'],
            'option3':params['option'],
            }

        # Calculate option prices
        option_legs = Option.return_options(
            opt_dict=params['opt_dict'], params=params)

        # Create payoff based on option type and direction
        if params['option'] == 'call' and params['direction'] == 'long':
            payoff = (option_legs['C1'] - option_legs['C1_0']
                      - option_legs['C2'] + option_legs['C2_0']
                      - option_legs['C3'] + option_legs['C3_0'])
            title = 'Long Christmas Tree with Calls'
            if params['value']:
                payoff2 = (option_legs['C1_G'] - option_legs['C1_0']
                           - option_legs['C2_G'] + option_legs['C2_0']
                           - option_legs['C3_G'] + option_legs['C3_0'])
            else:
                payoff2 = None

        if params['option'] == 'put' and params['direction'] == 'long':
            payoff = (-option_legs['C1'] + option_legs['C1_0']
                      - option_legs['C2'] + option_legs['C2_0']
                      + option_legs['C3'] - option_legs['C3_0'])
            title = 'Long Christmas Tree with Puts'
            if params['value']:
                payoff2 = (-option_legs['C1_G'] + option_legs['C1_0']
                           - option_legs['C2_G'] + option_legs['C2_0']
                           + option_legs['C3_G'] - option_legs['C3_0'])
            else:
                payoff2 = None

        if params['option'] == 'call' and params['direction'] == 'short':
            payoff = (-option_legs['C1'] + option_legs['C1_0']
                      + option_legs['C2'] - option_legs['C2_0']
                      + option_legs['C3'] - option_legs['C3_0'])
            title = 'Short Christmas Tree with Calls'
            if params['value']:
                payoff2 = (-option_legs['C1_G'] + option_legs['C1_0']
                           + option_legs['C2_G'] - option_legs['C2_0']
                           + option_legs['C3_G'] - option_legs['C3_0'])
            else:
                payoff2 = None

        if params['option'] == 'put' and params['direction'] == 'short':
            payoff = (option_legs['C1'] - option_legs['C1_0']
                      + option_legs['C2'] - option_legs['C2_0']
                      - option_legs['C3'] + option_legs['C3_0'])
            title = 'Short Christmas Tree with Puts'
            if params['value']:
                payoff2 = (option_legs['C1_G'] - option_legs['C1_0']
                           + option_legs['C2_G'] - option_legs['C2_0']
                           - option_legs['C3_G'] + option_legs['C3_0'])
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
        if (
            params['interactive'] and 
            params['notebook'] and 
            params['web_graph']
            ):
            fig = cls.vis_payoff(
            payoff_dict=params['payoff_dict'], params=params)
            return fig

        return cls.vis_payoff(
            payoff_dict=params['payoff_dict'], params=params)


    @classmethod
    def condor(
        cls,
        params: dict) -> go.Figure | None:
        """
        Displays the graph of the condor strategy:
            Long one low strike option
            Short one option with a higher strike
            Short one option with a higher strike
            Long one option with a higher strike

        Parameters
        ----------
        S : Float
             Underlying Stock Price. The default is 100.
        K1 : Float
             Strike Price of option 1. The default is 90.
        K2 : Float
             Strike Price of option 2. The default is 95.
        K3 : Float
             Strike Price of option 3. The default is 100.
        K4 : Float
             Strike Price of option 4. The default is 105.
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
            'legs':4,
            'S':params['S'],
            'K1':params['K1'],
            'T1':params['T'],
            'K2':params['K2'],
            'T2':params['T'],
            'K3':params['K3'],
            'T3':params['T'],
            'K4':params['K4'],
            'T4':params['T'],
            'r':params['r'],
            'q':params['q'],
            'sigma':params['sigma'],
            'option1':params['option'],
            'option2':params['option'],
            'option3':params['option'],
            'option4':params['option'],
            }

        # Calculate option prices
        option_legs = Option.return_options(
            opt_dict=params['opt_dict'], params=params)

        # Create payoff based on direction
        if params['direction'] == 'long':
            payoff = (option_legs['C1'] - option_legs['C1_0']
                      - option_legs['C2'] + option_legs['C2_0']
                      - option_legs['C3'] + option_legs['C3_0']
                      + option_legs['C4'] - option_legs['C4_0'])
            if params['value']:
                payoff2 = (option_legs['C1_G'] - option_legs['C1_0']
                           - option_legs['C2_G'] + option_legs['C2_0']
                           - option_legs['C3_G'] + option_legs['C3_0']
                           + option_legs['C4_G'] - option_legs['C4_0'])
            else:
                payoff2 = None

        if params['direction'] == 'short':
            payoff = (-option_legs['C1'] + option_legs['C1_0']
                      + option_legs['C2'] - option_legs['C2_0']
                      + option_legs['C3'] - option_legs['C3_0']
                      - option_legs['C4'] + option_legs['C4_0'])
            if params['value']:
                payoff2 = (-option_legs['C1_G'] + option_legs['C1_0']
                           + option_legs['C2_G'] - option_legs['C2_0']
                           + option_legs['C3_G'] - option_legs['C3_0']
                           - option_legs['C4_G'] + option_legs['C4_0'])
            else:
                payoff2 = None

        # Create title based on option type and direction
        if params['option'] == 'call' and params['direction'] == 'long':
            title = 'Long Condor with Calls'
        if params['option'] == 'put' and params['direction'] == 'long':
            title = 'Long Condor with Puts'
        if params['option'] == 'call' and params['direction'] == 'short':
            title = 'Short Condor with Calls'
        if params['option'] == 'put' and params['direction'] == 'short':
            title = 'Short Condor with Puts'

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
        if (
            params['interactive'] and 
            params['notebook'] and 
            params['web_graph']
            ):
            fig = cls.vis_payoff(
            payoff_dict=params['payoff_dict'], params=params)
            return fig

        return cls.vis_payoff(
            payoff_dict=params['payoff_dict'], params=params)


    @classmethod
    def iron_butterfly(
        cls,
        params: dict) -> go.Figure | None:
        """
        Displays the graph of the iron butterfly strategy:
            Short one OTM put
            Long one ATM put
            Long one ATM call
            Short one OTM call
        Akin to having a long straddle inside a larger short strangle
        (or vice-versa)

        Parameters
        ----------
        S : Float
             Underlying Stock Price. The default is 100.
        K1 : Float
             Strike Price of option 1. The default is 95.
        K2 : Float
             Strike Price of option 2. The default is 100.
        K3 : Float
             Strike Price of option 3. The default is 100.
        K4 : Float
             Strike Price of option 4. The default is 105.
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
            'legs':4,
            'S':params['S'],
            'K1':params['K1'],
            'T1':params['T'],
            'K2':params['K2'],
            'T2':params['T'],
            'K3':params['K3'],
            'T3':params['T'],
            'K4':params['K4'],
            'T4':params['T'],
            'r':params['r'],
            'q':params['q'],
            'sigma':params['sigma'],
            'option1':'put',
            'option2':'call',
            'option3':'put',
            'option4':'call',
            }

        # Calculate option prices
        option_legs = Option.return_options(
            opt_dict=params['opt_dict'], params=params)

        # Create payoff based on direction
        if params['direction'] == 'long':
            payoff = (-option_legs['C1'] + option_legs['C1_0']
                      + option_legs['C2'] - option_legs['C2_0']
                      + option_legs['C3'] - option_legs['C3_0']
                      - option_legs['C4'] + option_legs['C4_0'])
            title = 'Long Iron Butterfly'
            if params['value']:
                payoff2 = (-option_legs['C1_G'] + option_legs['C1_0']
                           + option_legs['C2_G'] - option_legs['C2_0']
                           + option_legs['C3_G'] - option_legs['C3_0']
                           - option_legs['C4_G'] + option_legs['C4_0'])
            else:
                payoff2 = None

        if params['direction'] == 'short':
            payoff = (option_legs['C1'] - option_legs['C1_0']
                      - option_legs['C2'] + option_legs['C2_0']
                      - option_legs['C3'] + option_legs['C3_0']
                      + option_legs['C4'] - option_legs['C4_0'])
            title = 'Short Iron Butterfly'
            if params['value']:
                payoff2 = (option_legs['C1_G'] - option_legs['C1_0']
                           - option_legs['C2_G'] + option_legs['C2_0']
                           - option_legs['C3_G'] + option_legs['C3_0']
                           + option_legs['C4_G'] - option_legs['C4_0'])
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
        if (
            params['interactive'] and 
            params['notebook'] and 
            params['web_graph']
            ):
            fig = cls.vis_payoff(
            payoff_dict=params['payoff_dict'], params=params)
            return fig

        return cls.vis_payoff(
            payoff_dict=params['payoff_dict'], params=params)


    @classmethod
    def iron_condor(
        cls,
        params: dict) -> go.Figure | None:
        """
        Displays the graph of the iron condor strategy:
            Long one OTM put
            Short one OTM put with a higher strike
            Short one OTM call
            Long one OTM call with a higher strike
        Akin to having a long strangle inside a larger short strangle
        (or vice-versa)

        Parameters
        ----------
        S : Float
             Underlying Stock Price. The default is 100.
        K1 : Float
             Strike Price of option 1. The default is 90.
        K2 : Float
             Strike Price of option 2. The default is 95.
        K3 : Float
             Strike Price of option 3. The default is 100.
        K4 : Float
             Strike Price of option 4. The default is 105.
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
            'legs':4,
            'S':params['S'],
            'K1':params['K1'],
            'T1':params['T'],
            'K2':params['K2'],
            'T2':params['T'],
            'K3':params['K3'],
            'T3':params['T'],
            'K4':params['K4'],
            'T4':params['T'],
            'r':params['r'],
            'q':params['q'],
            'sigma':params['sigma'],
            'option1':'put',
            'option2':'put',
            'option3':'call',
            'option4':'call',
            }

        # Calculate option prices
        option_legs = Option.return_options(
            opt_dict=params['opt_dict'], params=params)

        # Create payoff based on direction and value flag
        if params['direction'] == 'long':
            payoff = (option_legs['C1'] - option_legs['C1_0']
                      - option_legs['C2'] + option_legs['C2_0']
                      - option_legs['C3'] + option_legs['C3_0']
                      + option_legs['C4'] - option_legs['C4_0'])
            if params['value']:
                payoff2 = (option_legs['C1_G'] - option_legs['C1_0']
                           - option_legs['C2_G'] + option_legs['C2_0']
                           - option_legs['C3_G'] + option_legs['C3_0']
                           + option_legs['C4_G'] - option_legs['C4_0'])
            else:
                payoff2 = None

        elif params['direction'] == 'short':
            payoff = (-option_legs['C1'] + option_legs['C1_0']
                      + option_legs['C2'] - option_legs['C2_0']
                      + option_legs['C3'] - option_legs['C3_0']
                      - option_legs['C4'] + option_legs['C4_0'])
            if params['value']:
                payoff2 = (-option_legs['C1_G'] + option_legs['C1_0']
                           + option_legs['C2_G'] - option_legs['C2_0']
                           + option_legs['C3_G'] - option_legs['C3_0']
                           - option_legs['C4_G'] + option_legs['C4_0'])
            else:
                payoff2 = None

        # Create graph title based on direction
        if params['direction'] == 'long':
            title = 'Long Iron Condor'

        if params['direction'] == 'short':
            title = 'Short Iron Condor'

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
        if (
            params['interactive'] and 
            params['notebook'] and 
            params['web_graph']
            ):
            fig = cls.vis_payoff(
            payoff_dict=params['payoff_dict'], params=params)
            return fig

        return cls.vis_payoff(
            payoff_dict=params['payoff_dict'], params=params)


    @classmethod
    def vis_payoff(
        cls,
        payoff_dict: dict,
        params: dict) -> go.Figure | None:
        """
        Display the payoff diagrams

        Parameters
        ----------
        payoff_dict : Dict
            S : Float
                 Underlying Stock Price. The default is 100.
            SA : Array
                 Range of Strikes to provide x-axis values. The default
                 is 75% to 125%.
            payoff : Array
                Terminal payoff value less initial cost.
            label : Str
                Label for terminal payoff. The default is 'Payoff'
            title : Str
                Chart title giving name of combo. The default
                is 'Option Payoff'
            payoff2 : Array
                Current payoff value less initial cost.
            label2 : Str
                Label for current payoff value.
        params : Dict
            Dictionary of key parameters

        Returns
        -------
        2D Payoff Graph.

        """

        if params['interactive']: 
            if(params['notebook'] and params['web_graph']):
                fig = cls._vis_payoff_plotly(
                    payoff_dict=payoff_dict, params=params)
                return fig
            
            return cls._vis_payoff_plotly(
                payoff_dict=payoff_dict, params=params)

        return cls._vis_payoff_mpl(payoff_dict=payoff_dict, params=params)


    @staticmethod
    def _vis_payoff_mpl(
        payoff_dict: dict,
        params: dict) -> None:
        """
        Display the payoff diagrams using matplotlib

        Parameters
        ----------
        payoff_dict : Dict
            S : Float
                 Underlying Stock Price. The default is 100.
            SA : Array
                 Range of Strikes to provide x-axis values. The default
                 is 75% to 125%.
            payoff : Array
                Terminal payoff value less initial cost.
            label : Str
                Label for terminal payoff. The default is 'Payoff'
            title : Str
                Chart title giving name of combo. The default
                is 'Option Payoff'
            payoff2 : Array
                Current payoff value less initial cost.
            label2 : Str
                Label for current payoff value.
        params : Dict
            Dictionary of key parameters

        Returns
        -------
        2D Payoff Graph.

        """

        # Use seaborn darkgrid style
        plt.style.use(payoff_dict['mpl_style'])

        # Update chart parameters
        #pylab.rcParams.update(params['mpl_params'])
        mpl.rcParams.update(params['mpl_params'])

        # Create the figure and axes objects
        fig = plt.figure(figsize=payoff_dict['size2d'])

        # Use gridspec to allow modification of bounding box
        gs1 = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs1[0])

        # Plot the terminal payoff
        ax.plot(payoff_dict['SA'],
                payoff_dict['payoff'],
                color='blue',
                label='Payoff')

        # If the value flag is selected, plot the payoff with the
        # current time to maturity
        if payoff_dict['payoff2'] is not None:
            ax.plot(payoff_dict['SA'],
                    payoff_dict['payoff2'],
                    color='red',
                    label='Value')

        # Set a horizontal line at zero P&L
        ax.axhline(y=0, linewidth=0.5, color='k')

        #Set a vertical line at ATM strike
        ax.axvline(x=payoff_dict['S'], linewidth=0.5, color='k')

        # Apply a black border to the chart
        #ax.patch.set_edgecolor('black')
        #ax.patch.set_linewidth('1')

        fig.patch.set(linewidth=1, edgecolor='black')

        # Apply a grid
        plt.grid(True)

        # Set x and y axis labels and title
        ax.set(xlabel='Stock Price', ylabel='P&L', title=payoff_dict['title'])

        # Create a legend
        ax.legend(loc=0, fontsize=10)

        # Apply tight layout
        gs1.tight_layout(fig, rect=[0, 0, 1, 1])

        # Display the chart
        plt.show()


    @staticmethod
    def _vis_payoff_plotly(
        payoff_dict: dict,
        params: dict) -> go.Figure | None:
        """
        Display the payoff diagrams using plotly

        Parameters
        ----------
        payoff_dict : Dict
            S : Float
                 Underlying Stock Price. The default is 100.
            SA : Array
                 Range of Strikes to provide x-axis values. The default
                 is 75% to 125%.
            payoff : Array
                Terminal payoff value less initial cost.
            label : Str
                Label for terminal payoff. The default is 'Payoff'
            title : Str
                Chart title giving name of combo. The default
                is 'Option Payoff'
            payoff2 : Array
                Current payoff value less initial cost.
            label2 : Str
                Label for current payoff value.
        params : Dict
            Dictionary of key parameters

        Returns
        -------
        Payoff chart.

        """

        # Create the figure
        fig = go.Figure()

        # Plot the terminal payoff
        fig.add_trace(go.Scatter(x=payoff_dict['SA'],
                                 y=payoff_dict['payoff'],
                                 line=dict(color='blue'),
                                 name='Payoff'))

        # If the value flag is selected, plot the payoff with the
        # current time to maturity
        if payoff_dict['payoff2'] is not None:
            fig.add_trace(go.Scatter(x=payoff_dict['SA'],
                                     y=payoff_dict['payoff2'],
                                     line=dict(color='red'),
                                     name='Value'))


        fig.update_layout(
            title={
                'text': payoff_dict['title'],
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font':{
                    'size': 20,
                    'color': "#f2f5fa"
                }
            },
            xaxis_title={
                'text': 'Underlying Price',
                'font':{
                    'size': 15,
                    'color': "#f2f5fa"
                }
            },
            yaxis_title={
                'text': 'P&L',
                'font':{
                    'size': 15,
                    'color': "#f2f5fa"
                }
            },
            font={'color': '#f2f5fa'},
            paper_bgcolor='black',
            plot_bgcolor='black',
            legend={
                'x': 0.05,
                'y': 0.95,
                'traceorder': "normal",
                'bgcolor': 'rgba(0, 0, 0, 0)',
                'font': {
                    'family': "sans-serif",
                    'size': 12,
                    'color': "#f2f5fa"
                },
            },
        )

        if params['web_graph'] is False:
            fig.update_layout(
                autosize=False,
                width=800,
                height=600
                )

        fig.add_vline(x=payoff_dict['S'], line_width=0.5, line_color="white")
        fig.add_hline(y=0, line_width=0.5, line_color="white")

        fig.update_xaxes(showline=True,
                         linewidth=2,
                         linecolor='#2a3f5f',
                         mirror=True,
                         #range = [xmin, xmax],
                         gridwidth=1,
                         gridcolor='#2a3f5f',
                         zeroline=False)

        fig.update_yaxes(showline=True,
                         linewidth=2,
                         linecolor='#2a3f5f',
                         mirror=True,
                         #range = [ymin, ymax],
                         gridwidth=1,
                         gridcolor='#2a3f5f',
                         zeroline=False)

        # If running in an iPython notebook the chart will display
        # in line
        if params['notebook']:
            # If output is sent to Dash
            if params['web_graph']:
                return fig

            fig.show()
            return

        # Otherwise create an HTML file that opens in a new window
        plot(fig, auto_open=True)
        return
