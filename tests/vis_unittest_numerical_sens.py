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

        self.assertGreater(vis.Visualizer().sensitivities(
            greek='price', num_sens=True), 0)
        print("Default price: ", vis.Visualizer().sensitivities(
            greek='price', num_sens=True))

        self.assertIsInstance(vis.Visualizer().sensitivities(
            greek='price', num_sens=True), float)

        self.assertGreater(vis.Visualizer().sensitivities(
            greek='price', S=50, K=55, T=1, r=0.05, q=0.01,
            sigma=0.3, option='put', num_sens=True), 0)
        print("Revalued price: ", vis.Visualizer().sensitivities(
            greek='price', S=50, K=55, T=1, r=0.05, q=0.01,
            sigma=0.3, option='put', num_sens=True))

        self.assertIsInstance(vis.Visualizer().sensitivities(
            greek='price', S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3,
            option='put', num_sens=True), float)


    def test_delta(self):
        """
        Unit test for delta function.

        Returns
        -------
        Pass / Fail.

        """
        self.assertGreater(abs(vis.Visualizer().sensitivities(
            greek='delta', num_sens=True)), 0)
        print("Default delta: ", vis.Visualizer().sensitivities(
            greek='delta', num_sens=True))

        self.assertIsInstance(vis.Visualizer().sensitivities(
            greek='delta', num_sens=True), float)

        self.assertGreater(abs(vis.Visualizer().sensitivities(
            greek='delta', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued delta: ", vis.Visualizer().sensitivities(
            greek='delta', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Visualizer().sensitivities(
            greek='delta', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)


    def test_theta(self):
        """
        Unit test for theta function.

        Returns
        -------
        Pass / Fail.

        """
        self.assertGreater(abs(vis.Visualizer().sensitivities(
            greek='theta', num_sens=True)), 0)
        print("Default theta: ", vis.Visualizer().sensitivities(
            greek='theta', num_sens=True))

        self.assertIsInstance(vis.Visualizer().sensitivities(
            greek='theta', num_sens=True), float)

        self.assertGreater(abs(vis.Visualizer().sensitivities(
            greek='theta', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued theta: ", vis.Visualizer().sensitivities(
            greek='theta', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Visualizer().sensitivities(
            greek='theta', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)


    def test_gamma(self):
        """
        Unit test for gamma function.

        Returns
        -------
        Pass / Fail.

        """
        self.assertGreater(abs(vis.Visualizer().sensitivities(
            greek='gamma', num_sens=True)), 0)
        print("Default gamma: ", vis.Visualizer().sensitivities(
            greek='gamma', num_sens=True))

        self.assertIsInstance(vis.Visualizer().sensitivities(
            greek='gamma', num_sens=True), float)

        self.assertGreater(abs(vis.Visualizer().sensitivities(
            greek='gamma', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued gamma: ", vis.Visualizer().sensitivities(
            greek='gamma', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Visualizer().sensitivities(
            greek='gamma', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)


    def test_vega(self):
        """
        Unit test for vega function.

        Returns
        -------
        Pass / Fail.

        """
        self.assertGreater(abs(vis.Visualizer().sensitivities(
            greek='vega', num_sens=True)), 0)
        print("Default vega: ", vis.Visualizer().sensitivities(
            greek='vega', num_sens=True))

        self.assertIsInstance(vis.Visualizer().sensitivities(
            greek='vega', num_sens=True), float)

        self.assertGreater(abs(vis.Visualizer().sensitivities(
            greek='vega', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued vega: ", vis.Visualizer().sensitivities(
            greek='vega', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Visualizer().sensitivities(
            greek='vega', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)


    def test_rho(self):
        """
        Unit test for rho function.

        Returns
        -------
        Pass / Fail.

        """
        self.assertGreater(abs(vis.Visualizer().sensitivities(
            greek='rho', num_sens=True)), 0)
        print("Default rho: ", vis.Visualizer().sensitivities(
            greek='rho', num_sens=True))

        self.assertIsInstance(vis.Visualizer().sensitivities(
            greek='rho', num_sens=True), float)

        self.assertGreater(abs(vis.Visualizer().sensitivities(
            greek='rho', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued rho: ", vis.Visualizer().sensitivities(
            greek='rho', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Visualizer().sensitivities(
            greek='rho', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)


    def test_vanna(self):
        """
        Unit test for vanna function.

        Returns
        -------
        Pass / Fail.

        """
        self.assertGreater(abs(vis.Visualizer().sensitivities(
            greek='vanna', num_sens=True)), 0)
        print("Default vanna: ", vis.Visualizer().sensitivities(
            greek='vanna', num_sens=True))

        self.assertIsInstance(vis.Visualizer().sensitivities(
            greek='vanna', num_sens=True), float)

        self.assertGreater(abs(vis.Visualizer().sensitivities(
            greek='vanna', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued vanna: ", vis.Visualizer().sensitivities(
            greek='vanna', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Visualizer().sensitivities(
            greek='vanna', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)


    def test_vomma(self):
        """
        Unit test for vomma function.

        Returns
        -------
        Pass / Fail.

        """
        self.assertGreater(abs(vis.Visualizer().sensitivities(
            greek='vomma', num_sens=True)), 0)
        print("Default vomma: ", vis.Visualizer().sensitivities(
            greek='vomma', num_sens=True))

        self.assertIsInstance(vis.Visualizer().sensitivities(
            greek='vomma', num_sens=True), float)

        self.assertGreater(abs(vis.Visualizer().sensitivities(
            greek='vomma', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued vomma: ", vis.Visualizer().sensitivities(
            greek='vomma', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Visualizer().sensitivities(
            greek='vomma', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)


    def test_charm(self):
        """
        Unit test for charm function.

        Returns
        -------
        Pass / Fail.

        """
        self.assertGreater(abs(vis.Visualizer().sensitivities(
            greek='charm', num_sens=True)), 0)
        print("Default charm: ", vis.Visualizer().sensitivities(
            greek='charm', num_sens=True))

        self.assertIsInstance(vis.Visualizer().sensitivities(
            greek='charm', num_sens=True), float)

        self.assertGreater(abs(vis.Visualizer().sensitivities(
            greek='charm', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued charm: ", vis.Visualizer().sensitivities(
            greek='charm', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Visualizer().sensitivities(
            greek='charm', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)


    def test_zomma(self):
        """
        Unit test for zomma function.

        Returns
        -------
        Pass / Fail.

        """
        self.assertGreater(abs(vis.Visualizer().sensitivities(
            greek='zomma', num_sens=True)), 0)
        print("Default zomma: ", vis.Visualizer().sensitivities(
            greek='zomma', num_sens=True))

        self.assertIsInstance(vis.Visualizer().sensitivities(
            greek='zomma', num_sens=True), float)

        self.assertGreater(abs(vis.Visualizer().sensitivities(
            greek='zomma', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued zomma: ", vis.Visualizer().sensitivities(
            greek='zomma', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Visualizer().sensitivities(
            greek='zomma', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)


    def test_speed(self):
        """
        Unit test for speed function.

        Returns
        -------
        Pass / Fail.

        """
        self.assertGreater(abs(vis.Visualizer().sensitivities(
            greek='speed', num_sens=True)), 0)
        print("Default speed: ", vis.Visualizer().sensitivities(
            greek='speed', num_sens=True))

        self.assertIsInstance(vis.Visualizer().sensitivities(
            greek='speed', num_sens=True), float)

        self.assertGreater(abs(vis.Visualizer().sensitivities(
            greek='speed', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued speed: ", vis.Visualizer().sensitivities(
            greek='speed', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Visualizer().sensitivities(
            greek='speed', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)


    def test_color(self):
        """
        Unit test for color function.

        Returns
        -------
        Pass / Fail.

        """
        self.assertGreater(abs(vis.Visualizer().sensitivities(
            greek='color', num_sens=True)), 0)
        print("Default color: ", vis.Visualizer().sensitivities(
            greek='color', num_sens=True))

        self.assertIsInstance(vis.Visualizer().sensitivities(
            greek='color', num_sens=True), float)

        self.assertGreater(abs(vis.Visualizer().sensitivities(
            greek='color', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued color: ", vis.Visualizer().sensitivities(
            greek='color', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Visualizer().sensitivities(
            greek='color', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)


    def test_ultima(self):
        """
        Unit test for ultima function.

        Returns
        -------
        Pass / Fail.

        """
        self.assertGreater(abs(vis.Visualizer().sensitivities(
            greek='ultima', num_sens=True)), 0)
        print("Default ultima: ", vis.Visualizer().sensitivities(
            greek='ultima', num_sens=True))

        self.assertIsInstance(vis.Visualizer().sensitivities(
            greek='ultima', num_sens=True), float)

        self.assertGreater(abs(vis.Visualizer().sensitivities(
            greek='ultima', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued ultima: ", vis.Visualizer().sensitivities(
            greek='ultima', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Visualizer().sensitivities(
            greek='ultima', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)


    def test_vega_bleed(self):
        """
        Unit test for vega bleed function.

        Returns
        -------
        Pass / Fail.

        """
        self.assertGreater(abs(vis.Visualizer().sensitivities(
            greek='vega bleed', num_sens=True)), 0)
        print("Default vega_bleed: ", vis.Visualizer().sensitivities(
            greek='vega bleed', num_sens=True))

        self.assertIsInstance(vis.Visualizer().sensitivities(
            greek='vega bleed', num_sens=True), float)

        self.assertGreater(abs(vis.Visualizer().sensitivities(
            greek='vega bleed', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued vega_bleed: ", vis.Visualizer().sensitivities(
            greek='vega bleed', num_sens=True,
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Visualizer().sensitivities(
            greek='vega bleed', num_sens=True,
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
