import unittest
import optionvisualizer.visualizer as vis

class OptionGraphicalTestCase(unittest.TestCase):
    

    def test_visualize(self):

        vis.Option().visualize()

        vis.Option().visualize(
            risk=False, combo_payoff='backspread', S=50, K1=48, K2=52, T=1, 
            r=0.05, q=0.01, sigma=0.3, option='put', ratio=3, value=True, 
            mpl_style='fivethirtyeight', size2d=(6, 4))

        vis.Option().visualize(
            risk=True, graphtype='2D', x_plot='vol', y_plot='vega', S=50, 
            G1=45, G2=50, G3=55, T=0.5, T1=0.5, T2=0.5, T3=0.5, time_shift=0.5, 
            r=0.05, q=0.01, sigma=0.3, option='put', direction='short', 
            size2d=(6, 4), mpl_style='fivethirtyeight', num_sens=True)

        vis.Option().visualize(
            risk=True, graphtype='3D', interactive=False, S=50, r=0.05, q=0.01, 
            sigma=0.3, option='put', notebook=False, colorscheme='plasma', 
            colorintensity=0.8, size3d=(12, 8), direction='short', axis='vol', 
            spacegrain=150, azim=-50, elev=20, greek='vega', num_sens=True)

        vis.Option().visualize(
            risk=True, graphtype='3D', interactive=True, S=50, r=0.05, q=0.01, 
            sigma=0.3, option='put', notebook=False, colorscheme='plasma', 
            colorintensity=0.8, size3d=(12, 8), direction='short', axis='vol', 
            spacegrain=150, azim=-50, elev=20, greek='vega', num_sens=True)

        
    def test_greeks(self):

        vis.Option().greeks()

        vis.Option().greeks(
            graphtype='2D', x_plot='vol', y_plot='vega', S=50, G1=45, G2=50, 
            G3=55, T=0.5, T1=0.5, T2=0.5, T3=0.5, time_shift=0.5, r=0.05, 
            q=0.01, sigma=0.3, option='put', direction='short', size2d=(6, 4), 
            mpl_style='fivethirtyeight', num_sens=True)

        vis.Option().greeks(
            graphtype='3D', interactive=False, S=50, r=0.05, q=0.01, sigma=0.3, 
            option='put', notebook=False, colorscheme='plasma', 
            colorintensity=0.8, size3d=(12, 8), direction='short', axis='vol', 
            spacegrain=150, azim=-50, elev=20, greek='vega', num_sens=True)

        vis.Option().greeks(
            graphtype='3D', interactive=True, S=50, r=0.05, q=0.01, sigma=0.3, 
            option='put', notebook=False, colorscheme='plasma', 
            colorintensity=0.8, size3d=(12, 8), direction='short', axis='vol', 
            spacegrain=150, azim=-50, elev=20, greek='vega', num_sens=True)

        
    def test_greeks_graphs_2D(self):

        vis.Option().greeks_graphs_2D()

        vis.Option().greeks_graphs_2D(
            x_plot='vol', y_plot='vega', S=50, G1=45, G2=50, G3=55, T=0.5, 
            T1=0.5, T2=0.5, T3=0.5, time_shift=0.5, r=0.05, q=0.01, sigma=0.3, 
            option='put', direction='short', size2d=(6, 4), 
            mpl_style='fivethirtyeight', num_sens=True)

        vis.Option().greeks_graphs_2D(
            x_plot='vol', y_plot='rho', S=50, G1=45, G2=50, G3=55, T=0.5, 
            time_shift=0.5, r=0.05, q=0.01, sigma=0.3, option='put', 
            direction='short', size2d=(6, 4), mpl_style='fivethirtyeight', 
            num_sens=True)


    def test_greeks_graphs_3D(self):

        vis.Option().greeks_graphs_3D()

        vis.Option().greeks_graphs_3D(
            interactive=False, S=50, r=0.05, q=0.01, sigma=0.3, option='put', 
            notebook=False, colorscheme='plasma', colorintensity=0.8, 
            size3d=(12, 8), direction='short', axis='vol', spacegrain=150, 
            azim=-50, elev=20, greek='vega', num_sens=True)

        vis.Option().greeks_graphs_3D(
            interactive=True, S=50, r=0.05, q=0.01, sigma=0.3, option='put', 
            notebook=False, colorscheme='plasma', colorintensity=0.8, 
            size3d=(12, 8), direction='short', axis='vol', spacegrain=150, 
            azim=-50, elev=20, greek='vega', num_sens=True)


    def test_animated_2D_gif(self):

        vis.Option().animated_2D_gif()

        vis.Option().animated_2D_gif(
            gif_folder='images/test2d', gif_filename='test2d', T=1, steps=60, 
            x_plot='vol', y_plot='vega', S=50, G1=45, G2=50, G3=55, 
            r=0.05, q=0.01, sigma=0.3, option='put', direction='short', 
            size2d=(6, 4), mpl_style='fivethirtyeight', num_sens=True)        


    def test_animated_3D_gif(self):
        vis.Option().animated_3D_gif()

        vis.Option().animated_3D_gif(
            S=50, r=0.05, q=0.01, sigma=0.3, option='put', direction='short', 
            notebook=False, colorscheme='plasma', colorintensity=0.8, 
            size3d=(12, 8), axis='vol', spacegrain=150, azim=-50, elev=20, 
            greek='vega', num_sens=True, gif_folder='images/test3d', 
            gif_filename='test3d', gif_frame_update=2, gif_min_dist=9.2, 
            gif_max_dist=9.8, gif_min_elev=20.0, gif_max_elev=50.0, 
            gif_start_azim=90, gif_end_azim=270, gif_dpi=40, gif_ms=150)        


    def test_payoffs(self):

        vis.Option().payoffs()

        vis.Option().payoffs(
            combo_payoff='backspread', S=50, K1=48, K2=52, T=1, r=0.05, q=0.01, 
            sigma=0.3, option='put', ratio=3, value=True, 
            mpl_style='fivethirtyeight', size2d=(6, 4))


    def test_call(self):

        vis.Option().call()

        vis.Option().call(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, direction='short', 
            value=True, mpl_style='fivethirtyeight', size2d=(6, 4))

        
    def test_put(self):

        vis.Option().put()

        vis.Option().put(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, direction='short', 
            value=True, mpl_style='fivethirtyeight', size2d=(6, 4))


    def test_stock(self):

        vis.Option().stock()

        vis.Option().stock(S=50, direction='short', 
                           mpl_style='fivethirtyeight', size2d=(6, 4))


    def test_forward(self):

        vis.Option().forward()

        vis.Option().forward(
            S=50, T=1, r=0.05, q=0.01, sigma=0.3, direction='short', cash=True, 
            mpl_style='fivethirtyeight', size2d=(6, 4))


    def test_collar(self):

        vis.Option().collar()

        vis.Option().collar(S=50, K1=48, K2=52, T=1, r=0.05, q=0.01, sigma=0.3, 
           direction='short', value=True, mpl_style='fivethirtyeight', 
           size2d=(6, 4))


    def test_spread(self):

        vis.Option().spread()

        vis.Option().spread(
            S=50, K1=48, K2=52, T=1, r=0.05, q=0.01, sigma=0.3, option='put',
            direction='short', value=True, mpl_style='fivethirtyeight', 
            size2d=(6, 4))


    def test_backspread(self):

        vis.Option().backspread()

        vis.Option().backspread(
            S=50, K1=48, K2=52, T=1, r=0.05, q=0.01, sigma=0.3, option='put', 
            ratio=3, value=True, mpl_style='fivethirtyeight', size2d=(6, 4))


    def test_ratio_vertical_spread(self):

        vis.Option().ratio_vertical_spread()

        vis.Option().ratio_vertical_spread(
            S=50, K1=48, K2=52, T=1, r=0.05, q=0.01, sigma=0.3, option='put', 
            ratio=3, value=True, mpl_style='fivethirtyeight', size2d=(6, 4))


    def test_straddle(self):

        vis.Option().straddle()

        vis.Option().straddle(
            S=50, K=50, T=1, r=0.05, q=0.01, sigma=0.3, direction='short', 
            value=True, mpl_style='fivethirtyeight', size2d=(6, 4))


    def test_strangle(self):

        vis.Option().strangle()

        vis.Option().strangle(
            S=50, K1=45, K2=55, T=1, r=0.05, q=0.01, sigma=0.3, 
            direction='short', value=True, mpl_style='fivethirtyeight', 
            size2d=(6, 4))


    def test_butterfly(self):

        vis.Option().butterfly()

        vis.Option().butterfly(
            S=50, K1=45, K2=50, K3=55, T=1, r=0.05, q=0.01, sigma=0.3, 
            option='put', direction='short', value=True, 
            mpl_style='fivethirtyeight', size2d=(6, 4))


    def test_christmas_tree(self):

        vis.Option().christmas_tree()

        vis.Option().christmas_tree(
            S=50, K1=45, K2=50, K3=55, T=1, r=0.05, q=0.01, sigma=0.3, 
            option='put', direction='short', value=True, 
            mpl_style='fivethirtyeight', size2d=(6, 4))


    def test_condor(self):

        vis.Option().condor()

        vis.Option().condor(
            S=50, K1=45, K2=50, K3=55, K4=60, T=1, r=0.05, q=0.01, sigma=0.3, 
            option='put', direction='short', value=True, 
            mpl_style='fivethirtyeight', size2d=(6, 4))


    def test_iron_butterfly(self):

        vis.Option().iron_butterfly()

        vis.Option().iron_butterfly(
            S=50, K1=45, K2=50, K3=55, K4=60, T=1, r=0.05, q=0.01, sigma=0.3, 
            direction='short', value=True, 
            mpl_style='fivethirtyeight', size2d=(6, 4))


    def test_iron_condor(self):

        vis.Option().iron_condor()

        vis.Option().iron_condor(
            S=50, K1=45, K2=50, K3=55, K4=60, T=1, r=0.05, q=0.01, sigma=0.3, 
            direction='short', value=True, 
            mpl_style='fivethirtyeight', size2d=(6, 4))


if __name__ == '__main__':
    unittest.main()
        



