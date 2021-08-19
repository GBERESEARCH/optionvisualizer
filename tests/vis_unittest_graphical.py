"""
Unit tests for graphical output functions

"""

import unittest
import optionvisualizer.visualizer as vis

class OptionGraphicalTestCase(unittest.TestCase):
    """
    Unit tests for graphical output functions

    """

    @staticmethod
    def test_visualize():
        """
        Unit test for visualize function.

        Returns
        -------
        Pass / Fail.

        """
        vis.Visualizer().visualize()

        vis.Visualizer().visualize(
            risk=False, combo_payoff='backspread', S=50, K1=48, K2=52, T=1,
            r=0.05, q=0.01, sigma=0.3, option='put', ratio=3, value=True,
            mpl_style='fivethirtyeight', size2d=(6, 4))

        vis.Visualizer().visualize(
            risk=True, graphtype='2D', x_plot='vol', y_plot='vega', S=50,
            G1=45, G2=50, G3=55, T=0.5, T1=0.5, T2=0.5, T3=0.5, time_shift=0.5,
            r=0.05, q=0.01, sigma=0.3, option='put', direction='short',
            size2d=(6, 4), mpl_style='fivethirtyeight', num_sens=True)

        vis.Visualizer().visualize(
            risk=True, graphtype='3D', interactive=False, S=50, r=0.05, q=0.01,
            sigma=0.3, option='put', notebook=False, colorscheme='plasma',
            colorintensity=0.8, size3d=(12, 8), direction='short', axis='vol',
            spacegrain=150, azim=-50, elev=20, greek='vega', num_sens=True)

        vis.Visualizer().visualize(
            risk=True, graphtype='3D', interactive=True, S=50, r=0.05, q=0.01,
            sigma=0.3, option='put', notebook=False, colorscheme='plasma',
            colorintensity=0.8, size3d=(12, 8), direction='short', axis='vol',
            spacegrain=150, azim=-50, elev=20, greek='vega', num_sens=True)


    @staticmethod
    def test_greeks():
        """
        Unit test for greeks function.

        Returns
        -------
        Pass / Fail.

        """
        vis.Visualizer().greeks()

        vis.Visualizer().greeks(
            graphtype='2D', x_plot='vol', y_plot='vega', S=50, G1=45, G2=50,
            G3=55, T=0.5, T1=0.5, T2=0.5, T3=0.5, time_shift=0.5, r=0.05,
            q=0.01, sigma=0.3, option='put', direction='short', size2d=(6, 4),
            mpl_style='fivethirtyeight', num_sens=True)

        vis.Visualizer().greeks(
            graphtype='3D', interactive=False, S=50, r=0.05, q=0.01, sigma=0.3,
            option='put', notebook=False, colorscheme='plasma',
            colorintensity=0.8, size3d=(12, 8), direction='short', axis='vol',
            spacegrain=150, azim=-50, elev=20, greek='vega', num_sens=True)

        vis.Visualizer().greeks(
            graphtype='3D', interactive=True, S=50, r=0.05, q=0.01, sigma=0.3,
            option='put', notebook=False, colorscheme='plasma',
            colorintensity=0.8, size3d=(12, 8), direction='short', axis='vol',
            spacegrain=150, azim=-50, elev=20, greek='vega', num_sens=True)


    @staticmethod
    def test_greeks_graphs_2d():
        """
        Unit test for 2d greeks function.

        Returns
        -------
        Pass / Fail.

        """
        vis.Visualizer().greeks(graphtype='2D')

        vis.Visualizer().greeks(graphtype='2D',
            x_plot='vol', y_plot='vega', S=50, G1=45, G2=50, G3=55, T=0.5,
            T1=0.5, T2=0.5, T3=0.5, time_shift=0.5, r=0.05, q=0.01, sigma=0.3,
            option='put', direction='short', size2d=(6, 4),
            mpl_style='fivethirtyeight', num_sens=True)

        vis.Visualizer().greeks(graphtype='2D',
            x_plot='vol', y_plot='rho', S=50, G1=45, G2=50, G3=55, T=0.5,
            time_shift=0.5, r=0.05, q=0.01, sigma=0.3, option='put',
            direction='short', size2d=(6, 4), mpl_style='fivethirtyeight',
            num_sens=True)


    @staticmethod
    def test_greeks_graphs_3d():
        """
        Unit test for 3d greeks function.

        Returns
        -------
        Pass / Fail.

        """
        vis.Visualizer().greeks(graphtype='3D')

        vis.Visualizer().greeks(graphtype='3D',
            interactive=False, S=50, r=0.05, q=0.01, sigma=0.3, option='put',
            notebook=False, colorscheme='plasma', colorintensity=0.8,
            size3d=(12, 8), direction='short', axis='vol', spacegrain=150,
            azim=-50, elev=20, greek='vega', num_sens=True)

        vis.Visualizer().greeks(graphtype='3D',
            interactive=True, S=50, r=0.05, q=0.01, sigma=0.3, option='put',
            notebook=False, colorscheme='plasma', colorintensity=0.8,
            size3d=(12, 8), direction='short', axis='vol', spacegrain=150,
            azim=-50, elev=20, greek='vega', num_sens=True)


    @staticmethod
    def test_animated_gif_2d():
        """
        Unit test for animated 2d gif function.

        Returns
        -------
        Pass / Fail.

        """
        vis.Visualizer().animated_gif(graphtype='2D')

        vis.Visualizer().animated_gif(graphtype='2D',
            gif_folder='images/test2d', gif_filename='test2d', T=1, steps=60,
            x_plot='vol', y_plot='vega', S=50, G1=45, G2=50, G3=55,
            r=0.05, q=0.01, sigma=0.3, option='put', direction='short',
            size2d=(6, 4), mpl_style='fivethirtyeight', num_sens=True)


    @staticmethod
    def test_animated_gif_3d():
        """
        Unit test for animated 3d gif function.

        Returns
        -------
        Pass / Fail.

        """
        vis.Visualizer().animated_gif(graphtype='3D')

        vis.Visualizer().animated_gif(graphtype='3D',
            S=50, r=0.05, q=0.01, sigma=0.3, option='put', direction='short',
            notebook=False, colorscheme='plasma', colorintensity=0.8,
            size3d=(12, 8), axis='vol', spacegrain=150, azim=-50, elev=20,
            greek='vega', num_sens=True, gif_folder='images/test3d',
            gif_filename='test3d', gif_frame_update=2, gif_min_dist=9.2,
            gif_max_dist=9.8, gif_min_elev=20.0, gif_max_elev=50.0,
            gif_start_azim=90, gif_end_azim=270, gif_dpi=40, gif_ms=150)


    @staticmethod
    def test_payoffs():
        """
        Unit test for payoffs function.

        Returns
        -------
        Pass / Fail.

        """
        vis.Visualizer().payoffs(payoff_type='straddle')

        vis.Visualizer().payoffs(
            payoff_type='backspread', S=50, K1=48, K2=52, T=1, r=0.05, q=0.01,
            sigma=0.3, option='put', ratio=3, value=True,
            mpl_style='fivethirtyeight', size2d=(6, 4))


    @staticmethod
    def test_call():
        """
        Unit test for call function.

        Returns
        -------
        Pass / Fail.

        """
        vis.Visualizer().payoffs(payoff_type='call')

        vis.Visualizer().payoffs(payoff_type='call',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, direction='short',
            value=True, mpl_style='fivethirtyeight', size2d=(6, 4))


    @staticmethod
    def test_put():
        """
        Unit test for put function.

        Returns
        -------
        Pass / Fail.

        """
        vis.Visualizer().payoffs(payoff_type='put')

        vis.Visualizer().payoffs(payoff_type='put',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, direction='short',
            value=True, mpl_style='fivethirtyeight', size2d=(6, 4))


    @staticmethod
    def test_stock():
        """
        Unit test for stock function.

        Returns
        -------
        Pass / Fail.

        """
        vis.Visualizer().payoffs(payoff_type='stock')

        vis.Visualizer().payoffs(payoff_type='stock', S=50, direction='short',
                           mpl_style='fivethirtyeight', size2d=(6, 4))

    @staticmethod
    def test_forward():
        """
        Unit test for forward function.

        Returns
        -------
        Pass / Fail.

        """
        vis.Visualizer().payoffs(payoff_type='forward')

        vis.Visualizer().payoffs(payoff_type='forward',
            S=50, T=1, r=0.05, q=0.01, sigma=0.3, direction='short', cash=True,
            mpl_style='fivethirtyeight', size2d=(6, 4))


    @staticmethod
    def test_collar():
        """
        Unit test for collar function.

        Returns
        -------
        Pass / Fail.

        """
        vis.Visualizer().payoffs(payoff_type='collar')

        vis.Visualizer().payoffs(
            payoff_type='collar', S=50, K1=48, K2=52, T=1, r=0.05, q=0.01,
            sigma=0.3, direction='short', value=True,
            mpl_style='fivethirtyeight', size2d=(6, 4))


    @staticmethod
    def test_spread():
        """
        Unit test for spread function.

        Returns
        -------
        Pass / Fail.

        """
        vis.Visualizer().payoffs(payoff_type='spread')

        vis.Visualizer().payoffs(payoff_type='spread',
            S=50, K1=48, K2=52, T=1, r=0.05, q=0.01, sigma=0.3, option='put',
            direction='short', value=True, mpl_style='fivethirtyeight',
            size2d=(6, 4))


    @staticmethod
    def test_backspread():
        """
        Unit test for backspread function.

        Returns
        -------
        Pass / Fail.

        """
        vis.Visualizer().payoffs(payoff_type='backspread')

        vis.Visualizer().payoffs(payoff_type='backspread',
            S=50, K1=48, K2=52, T=1, r=0.05, q=0.01, sigma=0.3, option='put',
            ratio=3, value=True, mpl_style='fivethirtyeight', size2d=(6, 4))


    @staticmethod
    def test_ratio_vertical_spread():
        """
        Unit test for ratio vertical spread function.

        Returns
        -------
        Pass / Fail.

        """
        vis.Visualizer().payoffs(payoff_type='ratio vertical spread')

        vis.Visualizer().payoffs(payoff_type='ratio vertical spread',
            S=50, K1=48, K2=52, T=1, r=0.05, q=0.01, sigma=0.3, option='put',
            ratio=3, value=True, mpl_style='fivethirtyeight', size2d=(6, 4))


    @staticmethod
    def test_straddle():
        """
        Unit test for straddle function.

        Returns
        -------
        Pass / Fail.

        """
        vis.Visualizer().payoffs(payoff_type='straddle')

        vis.Visualizer().payoffs(payoff_type='straddle',
            S=50, K=50, T=1, r=0.05, q=0.01, sigma=0.3, direction='short',
            value=True, mpl_style='fivethirtyeight', size2d=(6, 4))


    @staticmethod
    def test_strangle():
        """
        Unit test for strangle function.

        Returns
        -------
        Pass / Fail.

        """
        vis.Visualizer().payoffs(payoff_type='strangle')

        vis.Visualizer().payoffs(payoff_type='strangle',
            S=50, K1=45, K2=55, T=1, r=0.05, q=0.01, sigma=0.3,
            direction='short', value=True, mpl_style='fivethirtyeight',
            size2d=(6, 4))


    @staticmethod
    def test_butterfly():
        """
        Unit test for butterfly function.

        Returns
        -------
        Pass / Fail.

        """
        vis.Visualizer().payoffs(payoff_type='butterfly')

        vis.Visualizer().payoffs(payoff_type='butterfly',
            S=50, K1=45, K2=50, K3=55, T=1, r=0.05, q=0.01, sigma=0.3,
            option='put', direction='short', value=True,
            mpl_style='fivethirtyeight', size2d=(6, 4))


    @staticmethod
    def test_christmas_tree():
        """
        Unit test for christmas tree function.

        Returns
        -------
        Pass / Fail.

        """
        vis.Visualizer().payoffs(payoff_type='christmas tree')

        vis.Visualizer().payoffs(payoff_type='christmas tree',
            S=50, K1=45, K2=50, K3=55, T=1, r=0.05, q=0.01, sigma=0.3,
            option='put', direction='short', value=True,
            mpl_style='fivethirtyeight', size2d=(6, 4))


    @staticmethod
    def test_condor():
        """
        Unit test for condor function.

        Returns
        -------
        Pass / Fail.

        """
        vis.Visualizer().payoffs(payoff_type='condor')

        vis.Visualizer().payoffs(payoff_type='condor',
            S=50, K1=45, K2=50, K3=55, K4=60, T=1, r=0.05, q=0.01, sigma=0.3,
            option='put', direction='short', value=True,
            mpl_style='fivethirtyeight', size2d=(6, 4))


    @staticmethod
    def test_iron_butterfly():
        """
        Unit test for iron butterfly function.

        Returns
        -------
        Pass / Fail.

        """
        vis.Visualizer().payoffs(payoff_type='iron butterfly')

        vis.Visualizer().payoffs(payoff_type='iron butterfly',
            S=50, K1=45, K2=50, K3=55, K4=60, T=1, r=0.05, q=0.01, sigma=0.3,
            direction='short', value=True,
            mpl_style='fivethirtyeight', size2d=(6, 4))


    @staticmethod
    def test_iron_condor():
        """
        Unit test for condor function.

        Returns
        -------
        Pass / Fail.

        """
        vis.Visualizer().payoffs(payoff_type='iron condor')

        vis.Visualizer().payoffs(payoff_type='iron condor',
            S=50, K1=45, K2=50, K3=55, K4=60, T=1, r=0.05, q=0.01, sigma=0.3,
            direction='short', value=True,
            mpl_style='fivethirtyeight', size2d=(6, 4))

if __name__ == '__main__':
    unittest.main()
