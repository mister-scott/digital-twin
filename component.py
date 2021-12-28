import numpy as np

def sub_vth(w_l,vgs,vth,temp=300):
    """
    Helper function to calculate sub-vth current analytically.
    
    Uses randomized parameters to mimic 
    measurement noise and manufacturing/material variability
    """
    # Electron charge
    q = 1.60218e-19
    # Boltzman constant
    k = 1.3806e-23
    # Capacitance factor (randomized to mimic manufacturing variability)
    eta = 1.2+0.01*np.random.normal()
    # Mobility factor/coefficient (randomized to mimic material and manufacturing variability)
    w_l = w_l*(1+0.01*np.random.normal())
    
    return w_l*np.exp(q*(vgs-vth)/(eta*k*temp))

class MOSFET:
    def __init__(self,params=None,terminals=None):
        
        # Params
        if params is None:
            self._params_ = {'BV':20,
                             'Vth':1.0,
                             'gm':1e-2}
        else:
            self._params_ = params
        
        # Terminals
        if terminals is None:
            self._terminals_ = {'source':0.0,
                        'drain':0.0,
                        'gate':0.0}
        else:
            self._terminals_ = terminals
        
        # Determine state
        self._state_ = self.determine_state()
        
        # Leakage model trained?
        self._leakage_ = False
        self.leakage_model = None
    
    def __repr__(self):
        return "Digital Twin of a MOSFET"
    
    def determine_state(self,vgs=None):
        """
        """
        if vgs is None:
            vgs = self._terminals_['gate'] - self._terminals_['source']
        else:
            vgs = vgs
        if vgs > self._params_['Vth']:
            return 'ON'
        else:
            return 'OFF'
    
    def id_vd(self,vgs=None,vds=None,rounding=True):
        """
        Calculates drain-source current from terminal voltages and gm 
        """        
        if vds is None:
            vds = self._terminals_['drain'] - self._terminals_['source']
        else:
            vds = vds
        if vgs is None:
            vgs = self._terminals_['gate'] - self._terminals_['source']
        else:
            vgs = vgs
        
        vth = self._params_['Vth']
        state = self.determine_state(vgs=vgs)
        self._state_ = state
        
        if state=='ON':
            if vds <= vgs - vth:
                ids = self._params_['gm']*(vgs - vth - (vds/2))*vds
            else:
                ids = (self._params_['gm']/2)*(vgs-vth)**2
            if rounding:
                return round(ids,3)
            else:
                return ids
        else:
            return sub_vth(w_l=self._params_['gm'],
                           vgs=vgs,
                           vth=vth)
            #return 0.0
        
    def rdson(self,vgs=None,vds=None):
        """
        Calculates Rdson i.e. on-state resistance of the MOSFET
        """
        if vds is None:
            vds = self._terminals_['drain '] - self._terminals_['source']
        else:
            vds = vds
        if vgs is None:
            vgs = self._terminals_['gate'] - self._terminals_['source']
        else:
            vgs = vgs
        
        ids = self.id_vd(vgs=vgs,vds=vds,rounding=False)
        vds_delta = vds+0.001
        ids_delta = self.id_vd(vgs=vgs,vds=vds_delta,rounding=False)
        rdson = 0.001/(ids_delta-ids)
        
        return round(rdson,3)
        
    def train_leakage(self,data=None,
                      batch_size=5,
                      epochs=20,
                      learning_rate=2e-5,
                      verbose=1):
        """
        Trains the digital twin for leakage current model with experimental data
        Args:
            data: The training data, expected as a Pandas DataFrame
            batch_size (int): Training batch size
            epochs (int): Number of epochs for training
            learning_rate (float): Learning rate for training
            verbose (0 or 1): Verbosity of display while training
        """
        if data is None:
            return "No data to train with"
        X_train_scaled, X_test_scaled, \
        y_train_scaled, y_test_scaled = prepare_data(data,
                                             input_cols=['w_l','vgs','vth'],
                                             output_var='log-leakage',
                                                     scaley=False)
        # Deep-learning model
        model = build_model(num_layers=3,
                            architecture=[32,32,32],
                            input_dim=3)
        # Compile and train
        model_trained = compile_train_model(model,
                                            X_train_scaled,
                                            y_train_scaled,
                                            batch_size=batch_size,
                                            epochs=epochs,
                                            learning_rate=learning_rate,
                                            verbose=verbose)
        self.leakage_model = model_trained
        self._leakage_ = True
    
    def leakage(self,
                w_l=1e-2,
                vgs=None,
                vth=None):
        """
        Calculates leakage current using the deep learning model
        """
        if not self._leakage_:
            return "Leakage model is not trained yet"
        # Vgs
        if vgs is None:
            vgs = self._terminals_['gate'] - self._terminals_['source']
        else:
            vgs = vgs
        # Vth
        if vth is None:
            vth = self._params_['Vth']
        else:
            vth = vth
        # Predict
        x = np.array([w_l,vgs,vth])
        ip = x.reshape(-1,3)
        result = float(10**(-self.leakage_model.predict(ip)))
        
        return result