import unittest
import optionvisualizer.visualizer as vis

class OptionNumericalTestCase(unittest.TestCase):
    
    
    def test_price(self):
        
        self.assertGreater(vis.Option().price(), 0)
        print("Default price: ", vis.Option().price())
        
        self.assertIsInstance(vis.Option().price(), float)
        
        self.assertGreater(vis.Option().price(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), 0)
        print("Revalued price: ", vis.Option().price(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))
        
        self.assertIsInstance(vis.Option().price(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)        
    
        
    def test_delta(self):
        
        self.assertGreater(abs(vis.Option().delta()), 0)
        print("Default delta: ", vis.Option().delta())
        
        self.assertIsInstance(vis.Option().delta(), float)
        
        self.assertGreater(abs(vis.Option().delta(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued delta: ", vis.Option().delta(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))
        
        self.assertIsInstance(vis.Option().delta(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)
        
        
    def test_theta(self):
        
        self.assertGreater(abs(vis.Option().theta()), 0)
        print("Default theta: ", vis.Option().theta())
        
        self.assertIsInstance(vis.Option().theta(), float)
        
        self.assertGreater(abs(vis.Option().theta(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued theta: ", vis.Option().theta(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))
        
        self.assertIsInstance(vis.Option().theta(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)  
        
        
    def test_gamma(self):
        
        self.assertGreater(abs(vis.Option().gamma()), 0)
        print("Default gamma: ", vis.Option().gamma())
        
        self.assertIsInstance(vis.Option().gamma(), float)
        
        self.assertGreater(abs(vis.Option().gamma(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued gamma: ", vis.Option().gamma(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))
        
        self.assertIsInstance(vis.Option().gamma(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)   
        
        
    def test_vega(self):
        
        self.assertGreater(abs(vis.Option().vega()), 0)
        print("Default vega: ", vis.Option().vega())
        
        self.assertIsInstance(vis.Option().vega(), float)
        
        self.assertGreater(abs(vis.Option().vega(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued vega: ", vis.Option().vega(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))
        
        self.assertIsInstance(vis.Option().vega(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)   
        
        
    def test_rho(self):
        
        self.assertGreater(abs(vis.Option().rho()), 0)
        print("Default rho: ", vis.Option().rho())
        
        self.assertIsInstance(vis.Option().rho(), float)
        
        self.assertGreater(abs(vis.Option().rho(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued rho: ", vis.Option().rho(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))
        
        self.assertIsInstance(vis.Option().rho(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)   
        
        
    def test_vanna(self):
        
        self.assertGreater(abs(vis.Option().vanna()), 0)
        print("Default vanna: ", vis.Option().vanna())
        
        self.assertIsInstance(vis.Option().vanna(), float)
        
        self.assertGreater(abs(vis.Option().vanna(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued vanna: ", vis.Option().vanna(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))
        
        self.assertIsInstance(vis.Option().vanna(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)    
        
        
    def test_vomma(self):
        
        self.assertGreater(abs(vis.Option().vomma()), 0)
        print("Default vomma: ", vis.Option().vomma())
        
        self.assertIsInstance(vis.Option().vomma(), float)
        
        self.assertGreater(abs(vis.Option().vomma(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued vomma: ", vis.Option().vomma(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))
        
        self.assertIsInstance(vis.Option().vomma(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)     


    def test_charm(self):
        
        self.assertGreater(abs(vis.Option().charm()), 0)
        print("Default charm: ", vis.Option().charm())
        
        self.assertIsInstance(vis.Option().charm(), float)
        
        self.assertGreater(abs(vis.Option().charm(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued charm: ", vis.Option().charm(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))
        
        self.assertIsInstance(vis.Option().charm(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)   


    def test_zomma(self):
        
        self.assertGreater(abs(vis.Option().zomma()), 0)
        print("Default zomma: ", vis.Option().zomma())
        
        self.assertIsInstance(vis.Option().zomma(), float)
        
        self.assertGreater(abs(vis.Option().zomma(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued zomma: ", vis.Option().zomma(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))
        
        self.assertIsInstance(vis.Option().zomma(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)    


    def test_speed(self):

        self.assertGreater(abs(vis.Option().speed()), 0)
        print("Default speed: ", vis.Option().speed())

        self.assertIsInstance(vis.Option().speed(), float)

        self.assertGreater(abs(vis.Option().speed(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued speed: ", vis.Option().speed(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Option().speed(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)          

        
    def test_color(self):

        self.assertGreater(abs(vis.Option().color()), 0)
        print("Default color: ", vis.Option().color())

        self.assertIsInstance(vis.Option().color(), float)

        self.assertGreater(abs(vis.Option().color(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued color: ", vis.Option().color(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Option().color(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)   

        
    def test_ultima(self):

        self.assertGreater(abs(vis.Option().ultima()), 0)
        print("Default ultima: ", vis.Option().ultima())

        self.assertIsInstance(vis.Option().ultima(), float)

        self.assertGreater(abs(vis.Option().ultima(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued ultima: ", vis.Option().ultima(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Option().ultima(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)     


    def test_vega_bleed(self):

        self.assertGreater(abs(vis.Option().vega_bleed()), 0)
        print("Default vega_bleed: ", vis.Option().vega_bleed())

        self.assertIsInstance(vis.Option().vega_bleed(), float)

        self.assertGreater(abs(vis.Option().vega_bleed(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued vega_bleed: ", vis.Option().vega_bleed(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Option().vega_bleed(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)    


    def test_analytical_sensitivities(self):

        self.assertGreater(abs(vis.Option().analytical_sensitivities()), 0)
        print("Default analytical_sensitivities: ", 
              vis.Option().analytical_sensitivities())

        self.assertIsInstance(vis.Option().analytical_sensitivities(), float)

        self.assertGreater(abs(vis.Option().analytical_sensitivities(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued analytical_sensitivities: ", 
              vis.Option().analytical_sensitivities(
                  S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Option().analytical_sensitivities(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)    


    def test_numerical_delta(self):

        self.assertGreater(abs(vis.Option().numerical_delta()), 0)
        print("Default numerical_delta: ", vis.Option().numerical_delta())

        self.assertIsInstance(vis.Option().numerical_delta(), float)

        self.assertGreater(abs(vis.Option().numerical_delta(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued numerical_delta: ", vis.Option().numerical_delta(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Option().numerical_delta(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)           


    def test_numerical_sensitivities(self):

        self.assertGreater(abs(vis.Option().numerical_sensitivities()), 0)
        print("Default numerical_sensitivities: ", 
              vis.Option().numerical_sensitivities())

        self.assertIsInstance(vis.Option().numerical_sensitivities(), float)

        self.assertGreater(abs(vis.Option().numerical_sensitivities(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued numerical_sensitivities: ", 
              vis.Option().numerical_sensitivities(
                  S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Option().numerical_sensitivities(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)    

        
    def test_sensitivities(self):

        self.assertGreater(abs(vis.Option().sensitivities()), 0)
        print("Default sensitivities: ", vis.Option().sensitivities())

        self.assertIsInstance(vis.Option().sensitivities(), float)

        self.assertGreater(abs(vis.Option().sensitivities(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put')), 0)
        print("Revalued sensitivities: ", vis.Option().sensitivities(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'))

        self.assertIsInstance(vis.Option().sensitivities(
            S=50, K=55, T=1, r=0.05, q=0.01, sigma=0.3, option='put'), float)      


    def test_barrier_price(self):

        self.assertGreater(abs(vis.Option().barrier_price()), 0)
        print("Default barrier_price: ", vis.Option().barrier_price())

        self.assertIsInstance(vis.Option().barrier_price(), float)

        self.assertGreater(abs(vis.Option().barrier_price(
            S=50, K=55, H=60, R=0.1, T=1, r=0.05, q=0.01, sigma=0.3, 
            option='put', barrier_direction='up', knock='out')), 0)
        print("Revalued barrier_price: ", vis.Option().barrier_price(
            S=50, K=55, H=60, R=0.1, T=1, r=0.05, q=0.01, sigma=0.3, 
            option='put', barrier_direction='up', knock='out'))

        self.assertIsInstance(vis.Option().barrier_price(
            S=50, K=55, H=60, R=0.1, T=1, r=0.05, q=0.01, sigma=0.3, 
            option='put', barrier_direction='up', knock='out'), float)    


if __name__ == '__main__':
    unittest.main()
        



