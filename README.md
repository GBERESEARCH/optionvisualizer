# optionvisualizer
## Visualize option prices and sensitivities

&nbsp;

Each of the functions comes with default parameters and can be run without arguments for ease of illustration or they can be specified individually. Sensitivities can be calculated both analytically and numerically.

&nbsp;

### Installation
Install from PyPI:
```
$ pip install optionvisualizer
```

&nbsp;

Or to install in new environment using anaconda:
```
$ conda create --name optvis
```
Activate new environment
```
$ activate optvis
```
Install Python
```
(optvis) $ conda install python
```
Install Spyder
```
(optvis) $ conda install spyder=4
```
Install package
```
(optvis) $ pip install optionvisualizer
```

&nbsp;

### Setup
Import visualizer and initialise an Option object

```
import optionvisualizer.visualizer as vis
opt = vis.Option()
```

&nbsp;

### Black-Scholes option pricing and sensitivities:
  - Price: option price 
  - Delta: sensitivity of option price to changes in asset price
  - Gamma: sensitivity of delta to changes in asset price
  - Vega: sensitivity of option price to changes in volatility
  - Theta: sensitivity of option price to changes in time to maturity
  - Rho: sensitivity of option price to changes in the risk free rate
  - Vomma: sensitivity of vega to changes in volatility; Volga
  - Vanna: sensitivity of delta to changes in volatility / of vega to changes in asset price
  - Charm: sensitivity of delta to changes in time to maturity aka Delta Bleed
  - Zomma: sensitivity of gamma to changes in volatility
  - Speed: sensitivity of gamma to changes in asset price; 3rd derivative of option price wrt spot
  - Color: sensitivity of gamma to changes in time to maturity; GammaTheta
  - Ultima: sensitivity of vomma to changes in volatility; 3rd derivative of option price wrt volatility
  - Vega Bleed: sensitivity of vega to changes in time to maturity

```
opt.price(S=3477, K=3400, T=0.5, r=0.005, q=0, sigma=0.3, option='put')
```

```
opt.sensitivities(greek='delta, S=3477, K=3400, T=0.5, r=0.005, q=0, sigma=0.3, option='put')
```

&nbsp;

### 2D greeks graphs:
#### Charts of 3 options showing price, vol or time against:
  - option value
  - delta
  - gamma
  - vega
  - theta

&nbsp;

#### Long Call Delta vs Price
```
opt.visualize(risk=True, x_plot='price', y_plot='delta', G1=100, G2=100, G3=100, T1=0.05, T2=0.15, T3=0.25)
```
![call_delta_price](images/call_delta_price.png)

&nbsp;

#### Long Call Theta vs Price
```
opt.visualize(risk=True, x_plot='price', y_plot='theta', G1=100, G2=100, G3=100, T1=0.05, T2=0.15, T3=0.25)
```
![long_call_theta_price](images/long_call_theta_price.png)

&nbsp;

#### Charts of 4 options showing price, strike and vol against rho
#### Short Rho vs Strike
```
opt.visualize(risk=True, x_plot='strike', y_plot='rho', direction='short')
```
![short_rho_strike](images/short_rho_strike.png)

&nbsp;

### 3D greeks graphs:
#### Each of the greeks above can be plotted showing Time to Expiration against Strike or Volatility

&nbsp;

#### Using matplotlib: 

#### Long Vega
```
opt.visualize(risk=True, graphtype='3D', greek='vega', S=50)
```
![long_vega_static](images/long_vega_static.png)

&nbsp;

#### Short Gamma
```
opt.visualize(risk=True, graphtype='3D', greek='gamma', direction='short')
```

![short_gamma_static](images/short_gamma_static.png)

&nbsp;

#### Or using plotly display a graph that can be rotated and zoomed:
#### Long Call Price
```
opt.visualize(risk=True, graphtype='3D', greek='price', colorscheme='Plasma', interactive=True)
```
![long_call_price_interactive](images/long_call_price_interactive.png)

&nbsp;

#### Long Put Price against Volatility
```
opt.visualize(risk=True, graphtype='3D', greek='price', axis='vol', option='put', interactive=True) 
```

![long_put_price_volatility](images/long_put_price_volatility.png)

&nbsp;

#### Long Vanna
```
opt.visualize(risk=True, graphtype='3D', greek='vanna', sigma=0.4, interactive=True)
```
![long_vanna_interactive](images/long_vanna_interactive.png)

&nbsp;

#### Short Zomma
```
opt.visualize(risk=True, graphtype='3D', greek='zomma', direction='short', interactive=True)
```
![short_zomma_interactive](images/short_zomma_interactive.png) 

&nbsp;

### Option strategy Payoff graphs:
  - call / put
  - stock
  - forward
  - collar 
  - call / put spread
  - backspread
  - ratio vertical spread
  - straddle
  - strangle
  - butterfly
  - christmas tree
  - iron butterfly
  - iron condor

&nbsp;

#### Short Call:
```
opt.visualize(risk=False, combo_payoff='call', S=90, K=95, T=0.75, r=0.05, q=0, sigma=0.3, direction='short', value=True)
```
![short_call](images/short_call.png)

&nbsp;

#### Long Straddle:
```
opt.visualize(risk=False, combo_payoff='straddle', mpl_style='ggplot')
```
![straddle](images/straddle.png)

&nbsp;

#### Short Christmas Tree:
```
opt.visualize(risk=False, combo_payoff='christmas tree', value=True, direction='short')
```
![christmas_tree](images/christmas_tree.png)

&nbsp;

The following volumes served as a reference for the formulas and charts:
* [The Complete Guide to Option Pricing Formulas, 2nd Ed, E. G. Haug]
* [Option Volatility & Pricing, S. Natenburg]
  
[The Complete Guide to Option Pricing Formulas, 2nd Ed, E. G. Haug]:<https://www.amazon.co.uk/Complete-Guide-Option-Pricing-Formulas/dp/0071389970/>
[Option Volatility & Pricing, S. Natenburg]:<https://www.amazon.co.uk/Option-Volatility-Pricing-Strategies-Techniques/dp/155738486X/>
