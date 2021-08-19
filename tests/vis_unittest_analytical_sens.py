"""
Unit tests for numerical output functions

"""

import unittest
import optionvisualizer.visualizer as vis

class OptionNumericalTestCase(unittest.TestCase):
    """
    Unit tests for numerical output functions

    """

    def test_price(self):
        """
        Unit test for option pricing function.

        Returns
        -------
        Pass / Fail.

        """

        self.assertGreater(vis.Visualizer().option_data(
            option_value='price'), 0)
        print("Default price: ", vis.Visualizer().option_data(
            option_value='price'))

        self.assertIsInstance(vis.Visualizer().option_data(
            option_value='price'), float)

        self.assertGreater(vis.Visualizer().option_data(
            option_value='price', S=50, K=55, T=1, r=0.05, q=0.01,
            sigma=0.3, option='put'), 0)
        print("Revalued price: ", vis.Visualizer().option_data(
            option_value='price', S=50, K=55, T=1, r=0.05, q=0.01,
            sigma=0.3, option='put'))

        self.assertIsInstance(vis.Visualizer().option_data(
            option_value='price', S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3,
            option='put'), float)


    def test_delta(self):
        """
        Unit test for delta function.

        Returns
        -------
        Pass / Fail.

        """
        self.assertGreater(abs(vis.Visualizer().option_data(
            option_value='delta')), 0)
        print("Default delta: ", vis.Visualizer().option_data(
            option_value='delta'))

        self.assertIsInstance(vis.Visualizer().option_data(
            option_value='delta'), float)

        self.assertGreater(abs(vis.Visualizer().option_data(
            option_value='delta',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued delta: ", vis.Visualizer().option_data(
            option_value='delta',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Visualizer().option_data(
            option_value='delta',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)


    def test_theta(self):
        """
        Unit test for theta function.

        Returns
        -------
        Pass / Fail.

        """
        self.assertGreater(abs(vis.Visualizer().option_data(
            option_value='theta')), 0)
        print("Default theta: ", vis.Visualizer().option_data(
            option_value='theta'))

        self.assertIsInstance(vis.Visualizer().option_data(
            option_value='theta'), float)

        self.assertGreater(abs(vis.Visualizer().option_data(
            option_value='theta',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued theta: ", vis.Visualizer().option_data(
            option_value='theta',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Visualizer().option_data(
            option_value='theta',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)


    def test_gamma(self):
        """
        Unit test for gamma function.

        Returns
        -------
        Pass / Fail.

        """
        self.assertGreater(abs(vis.Visualizer().option_data(
            option_value='gamma')), 0)
        print("Default gamma: ", vis.Visualizer().option_data(
            option_value='gamma'))

        self.assertIsInstance(vis.Visualizer().option_data(
            option_value='gamma'), float)

        self.assertGreater(abs(vis.Visualizer().option_data(
            option_value='gamma',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued gamma: ", vis.Visualizer().option_data(
            option_value='gamma',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Visualizer().option_data(
            option_value='gamma',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)


    def test_vega(self):
        """
        Unit test for vega function.

        Returns
        -------
        Pass / Fail.

        """
        self.assertGreater(abs(vis.Visualizer().option_data(
            option_value='vega')), 0)
        print("Default vega: ", vis.Visualizer().option_data(
            option_value='vega'))

        self.assertIsInstance(vis.Visualizer().option_data(
            option_value='vega'), float)

        self.assertGreater(abs(vis.Visualizer().option_data(
            option_value='vega',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued vega: ", vis.Visualizer().option_data(
            option_value='vega',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Visualizer().option_data(
            option_value='vega',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)


    def test_rho(self):
        """
        Unit test for rho function.

        Returns
        -------
        Pass / Fail.

        """
        self.assertGreater(abs(vis.Visualizer().option_data(
            option_value='rho')), 0)
        print("Default rho: ", vis.Visualizer().option_data(
            option_value='rho'))

        self.assertIsInstance(vis.Visualizer().option_data(
            option_value='rho'), float)

        self.assertGreater(abs(vis.Visualizer().option_data(
            option_value='rho',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued rho: ", vis.Visualizer().option_data(
            option_value='rho',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Visualizer().option_data(
            option_value='rho',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)


    def test_vanna(self):
        """
        Unit test for vanna function.

        Returns
        -------
        Pass / Fail.

        """
        self.assertGreater(abs(vis.Visualizer().option_data(
            option_value='vanna')), 0)
        print("Default vanna: ", vis.Visualizer().option_data(
            option_value='vanna'))

        self.assertIsInstance(vis.Visualizer().option_data(
            option_value='vanna'), float)

        self.assertGreater(abs(vis.Visualizer().option_data(
            option_value='vanna',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued vanna: ", vis.Visualizer().option_data(
            option_value='vanna',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Visualizer().option_data(
            option_value='vanna',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)


    def test_vomma(self):
        """
        Unit test for vomma function.

        Returns
        -------
        Pass / Fail.

        """
        self.assertGreater(abs(vis.Visualizer().option_data(
            option_value='vomma')), 0)
        print("Default vomma: ", vis.Visualizer().option_data(
            option_value='vomma'))

        self.assertIsInstance(vis.Visualizer().option_data(
            option_value='vomma'), float)

        self.assertGreater(abs(vis.Visualizer().option_data(
            option_value='vomma',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued vomma: ", vis.Visualizer().option_data(
            option_value='vomma',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Visualizer().option_data(
            option_value='vomma',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)


    def test_charm(self):
        """
        Unit test for charm function.

        Returns
        -------
        Pass / Fail.

        """
        self.assertGreater(abs(vis.Visualizer().option_data(
            option_value='charm')), 0)
        print("Default charm: ", vis.Visualizer().option_data(
            option_value='charm'))

        self.assertIsInstance(vis.Visualizer().option_data(
            option_value='charm'), float)

        self.assertGreater(abs(vis.Visualizer().option_data(
            option_value='charm',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued charm: ", vis.Visualizer().option_data(
            option_value='charm',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Visualizer().option_data(
            option_value='charm',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)


    def test_zomma(self):
        """
        Unit test for zomma function.

        Returns
        -------
        Pass / Fail.

        """
        self.assertGreater(abs(vis.Visualizer().option_data(
            option_value='zomma')), 0)
        print("Default zomma: ", vis.Visualizer().option_data(
            option_value='zomma'))

        self.assertIsInstance(vis.Visualizer().option_data(
            option_value='zomma'), float)

        self.assertGreater(abs(vis.Visualizer().option_data(
            option_value='zomma',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued zomma: ", vis.Visualizer().option_data(
            option_value='zomma',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Visualizer().option_data(
            option_value='zomma',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)


    def test_speed(self):
        """
        Unit test for speed function.

        Returns
        -------
        Pass / Fail.

        """
        self.assertGreater(abs(vis.Visualizer().option_data(
            option_value='speed')), 0)
        print("Default speed: ", vis.Visualizer().option_data(
            option_value='speed'))

        self.assertIsInstance(vis.Visualizer().option_data(
            option_value='speed'), float)

        self.assertGreater(abs(vis.Visualizer().option_data(
            option_value='speed',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued speed: ", vis.Visualizer().option_data(
            option_value='speed',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Visualizer().option_data(
            option_value='speed',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)


    def test_color(self):
        """
        Unit test for color function.

        Returns
        -------
        Pass / Fail.

        """
        self.assertGreater(abs(vis.Visualizer().option_data(
            option_value='color')), 0)
        print("Default color: ", vis.Visualizer().option_data(
            option_value='color'))

        self.assertIsInstance(vis.Visualizer().option_data(
            option_value='color'), float)

        self.assertGreater(abs(vis.Visualizer().option_data(
            option_value='color',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued color: ", vis.Visualizer().option_data(
            option_value='color',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Visualizer().option_data(
            option_value='color',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)


    def test_ultima(self):
        """
        Unit test for ultima function.

        Returns
        -------
        Pass / Fail.

        """
        self.assertGreater(abs(vis.Visualizer().option_data(
            option_value='ultima')), 0)
        print("Default ultima: ", vis.Visualizer().option_data(
            option_value='ultima'))

        self.assertIsInstance(vis.Visualizer().option_data(
            option_value='ultima'), float)

        self.assertGreater(abs(vis.Visualizer().option_data(
            option_value='ultima',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued ultima: ", vis.Visualizer().option_data(
            option_value='ultima',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Visualizer().option_data(
            option_value='ultima',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)


    def test_vega_bleed(self):
        """
        Unit test for vega bleed function.

        Returns
        -------
        Pass / Fail.

        """
        self.assertGreater(abs(vis.Visualizer().option_data(
            option_value='vega bleed')), 0)
        print("Default vega_bleed: ", vis.Visualizer().option_data(
            option_value='vega bleed'))

        self.assertIsInstance(vis.Visualizer().option_data(
            option_value='vega bleed'), float)

        self.assertGreater(abs(vis.Visualizer().option_data(
            option_value='vega bleed',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued vega_bleed: ", vis.Visualizer().option_data(
            option_value='vega bleed',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Visualizer().option_data(
            option_value='vega bleed',
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)


    def test_analytical_sensitivities(self):
        """
        Unit test for analytical sensitivities function.

        Returns
        -------
        Pass / Fail.

        """
        self.assertGreater(
            abs(vis.Visualizer().sensitivities(num_sens=False)), 0)
        print("Default analytical_sensitivities: ",
              vis.Visualizer().sensitivities(num_sens=False))

        self.assertIsInstance(
            vis.Visualizer().sensitivities(num_sens=False), float)

        self.assertGreater(abs(vis.Visualizer().sensitivities(num_sens=False,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued analytical_sensitivities: ",
              vis.Visualizer().sensitivities(num_sens=False,
                  S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Visualizer().sensitivities(num_sens=False,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)


    def test_numerical_sensitivities(self):
        """
        Unit test for numerical sensitivities function.

        Returns
        -------
        Pass / Fail.

        """
        self.assertGreater(
            abs(vis.Visualizer().sensitivities(num_sens=True)), 0)
        print("Default numerical_sensitivities: ",
              vis.Visualizer().sensitivities(num_sens=True))

        self.assertIsInstance(
            vis.Visualizer().sensitivities(num_sens=True), float)

        self.assertGreater(abs(vis.Visualizer().sensitivities(num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued numerical_sensitivities: ",
              vis.Visualizer().sensitivities(num_sens=True,
                  S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Visualizer().sensitivities(num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)


    def test_barrier(self):
        """
        Unit test for barrier pricing function.

        Returns
        -------
        Pass / Fail.

        """
        self.assertGreater(abs(vis.Visualizer().barrier()), 0)
        print("Default barrier_price: ", vis.Visualizer().barrier())

        self.assertIsInstance(vis.Visualizer().barrier(), float)

        self.assertGreater(abs(vis.Visualizer().barrier(
            S=50, K=55, H=60, R=0.1, T=1, r=0.05, q=0.01, sigma=0.3,
            option='put', barrier_direction='up', knock='out')), 0)
        print("Revalued barrier_price: ", vis.Visualizer().barrier(
            S=50, K=55, H=60, R=0.1, T=1, r=0.05, q=0.01, sigma=0.3,
            option='put', barrier_direction='up', knock='out'))

        self.assertIsInstance(vis.Visualizer().barrier(
            S=50, K=55, H=60, R=0.1, T=1, r=0.05, q=0.01, sigma=0.3,
            option='put', barrier_direction='up', knock='out'), float)


if __name__ == '__main__':
    unittest.main()
