# IMPORTS
import numpy as np 
import pandas as pd; pd.set_option('future.no_silent_downcasting', True)
import datetime as dt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)


class get_swaption_data:
    pass

oneM = get_swaption_data(startdate="04JAN2016", enddate="16MAY2025",freq="1B",swapstart="1Y",swapend="5Y",read_strikes=True)
oneM.get_all_metrics()
oneM.get_pv_swaprate()

class get_RNDs:
    def __init__(self,
                swaption_data: get_swaption_data,
                scenario: np.arange = np.arange(-200,200.1,0.1),
                processes: int= None
                ):
        """
        Class that uses swaption_data to perform further analysis and return RNDs.
        """
        self.dates = swaption_data.dates
        self.swaption_data = swaption_data
        self.strikes = self.swaption_data.strikes
        self.scenario = scenario
        self.processes = processes
        
        # Always compute these attributes
        # self.define_payoff()

    def define_payoff(self,date,np):
        """
        Defines the payoff as the difference be in each scenario for the ITM or ATM swaptions
        """

        # Getting relevant fwd price
        fwd_price = self.swaption_data.fwd_price[date]
        fwd_price_notna = (~fwd_price.isna()).to_list()
        self.fwd_price = fwd_price[fwd_price_notna]
        
        strikes = self.strikes[fwd_price_notna]

        strike = strikes[strikes<0]
        payoff_pay = np.maximum(self.scenario[:, None] - strike[None, :], 0)
        strike = strikes[strikes>=0]
        payoff_rec = np.maximum(strike[None, :] - self.scenario[:, None], 0)

        P = np.hstack([
            payoff_rec,
            payoff_pay
        ])

        self.payoff = P
        self.num_rec = len(payoff_rec[0])
        self.num_pay = len(payoff_pay[0])

    def calc_MEM(self,date):
        import numpy as np
        import cvxpy as cp
        import pandas as pd

        self.define_payoff(date,np)

        payoff = self.payoff
        m, n = payoff.shape

        d = np.concatenate([
            self.fwd_price.iloc[-self.num_rec:].values,
            self.fwd_price.iloc[:self.num_pay].values
        ],dtype='object')

        q_var = cp.Variable(m)
        entr = cp.sum(cp.entr(q_var))    
        cons = [payoff.T @ q_var == d, q_var >= 1e-12] #### Check >= vs. ==
        # cons = [payoff.T @ q_var == d, cp.sum(q_var) == 1, q_var >= 0]
        
        problem = cp.Problem(cp.Maximize(entr), cons)
        
        try:
            problem.solve(solver=cp.MOSEK)
        except:
            pass

        # print(payoff.shape, strikes, problem.status, d.shape)

        return {
            "date": date,
            "status": problem.status,
            "optimal_value": problem.value if problem.status == "optimal" else None,
            "solution": q_var.value.tolist() / q_var.value.sum() if problem.status == "optimal" else None
        }
    
    def get_all_metrics(self):
        if self.processes:
            import sys
            sys.path.append(r'H:\Python\Bachelorprojekt\Testing')  # Add module directory to sys.path
            import multiprocessing_module as mp

            result = mp.run_parallel(self.calc_MEM, self.dates,processes=self.processes)

        else:
            from tqdm import tqdm
            result = [self.calc_MEM(date) for date in tqdm(self.dates, desc="Processing")]

        self.distr = pd.DataFrame.from_dict(result)
        self.distr.index = self.dates
        
        self.failed = self.distr[self.distr.status != 'optimal']
        print(f"\n Failed: {len(self.failed)}")

rndoneM = get_RNDs(swaption_data=oneM,scenario=np.arange(-500,500,0.5),processes=None)
rndoneM.get_all_metrics()

class get_stats:
    def __init__(self,
                 RND_data: get_RNDs,
                 ):
        self.dates = RND_data.dates
        self.RND_data = RND_data
        self.scenario = RND_data.scenario

        self.stats = self.calc_stats(data=self.RND_data.distr)
        self.aggregated_statistics = self.agg_stats(data=self.stats)

    def calc_stats(self, data: pd.DataFrame) -> pd.DataFrame:
        stats_list = []
        for idx, row in data.iterrows():
            # Make sure that row.solution is a numpy array.
            P = np.array(row.solution)
            
            # Calculate the statistics
            mean_val = np.sum(P * self.scenario)
            volatility = np.sqrt(np.sum(P * (self.scenario - mean_val) ** 2))
            skewness = np.sum(P * (self.scenario - mean_val) ** 3) / (volatility ** 3)
            kurtosis = np.sum(P * (self.scenario - mean_val) ** 4) / (volatility ** 4)
            
            stats_list.append({
                "date": idx,
                "mean": mean_val,
                "volatility": volatility,
                "skewness": skewness,
                "kurtosis": kurtosis
            })
        return pd.DataFrame(stats_list)

    def agg_stats(self, data: pd.DataFrame) -> pd.DataFrame:
        metrics = data.columns.drop("date")

        return data[metrics].agg(["mean", "std"])

statsoneM = get_stats(rndoneM)
statsoneM.stats
statsoneM.aggregated_statistics


class define_data_model:
    def __init__(self,
                 swaption_data: get_swaption_data,
                 RND_data: get_RNDs,
                 stats = get_stats,
                 include_name: bool=False):
        
        self.swaption_data = swaption_data
        self.RND_data = RND_data
        self.stats = stats


        self.swaption_metrics()
        self.combine_data()

        if include_name:
            self.data_model.columns = [self.swaption_data.name + "_" + col for col in self.data_model.columns]

    def swaption_metrics(self):

        ## IMPLIED VOL
        impvol = self.swaption_data.impvol.T
        impvol.columns = ["impvol_" + str(float(strike)) for strike in self.swaption_data.strikes]
        
        # IMPLIED VOL DIFFERENCE
        strike_series = pd.Series(np.abs(self.swaption_data.strikes), index=self.swaption_data.swaptions)
        strike_counts = strike_series.value_counts()
        dup_strikes = strike_counts[strike_counts == 2].index
        for strike in dup_strikes:
            # instruments corresponding to this absolute strike
            instruments = strike_series[strike_series == strike].index.tolist()

            inst1, inst2 = instruments

            iv1 = self.swaption_data.impvol.loc[inst1]
            iv2 = self.swaption_data.impvol.loc[inst2]

            iv_diff = iv2 - iv1

            skew = pd.DataFrame(iv_diff,columns=[f"skew_+/-_{strike}"])

            impvol = pd.concat([impvol,skew],axis=1)

        protected = "impvol_0.0"
        # 1) pull out the protected column
        saved = impvol[[protected]].copy()
        # 2) dropna on *all the others*
        others = impvol.drop(columns=[protected]).dropna(axis=1)
        # 3) stitch your protected column back on
        impvol = pd.concat([saved, others], axis=1)

        impvol.ffill(axis=0, inplace=True)

        self.impvol = impvol

    def combine_data(self):
        
        data_model = self.stats.stats
        data_model.index = data_model.date
        data_model = data_model.drop(["date"],axis=1)
        
        # IMPVOL metrics
        data_model = pd.concat([data_model,self.impvol],axis=1)

        # Add swaprate
        data_model = pd.concat([data_model,self.swaption_data.swaption_details.swaprate],axis=1)

        self.data_model = data_model

oneM_data_model = define_data_model(swaption_data=oneM,RND_data=rndoneM,stats=statsoneM)
data_model = oneM_data_model.data_model


## MODEL SELECTION ##
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
## MODEL SELECTION ##


class AR_X_REGRESSION_MODEL:
    def __init__(self,
                 swaption_data: get_swaption_data,
                 data_model: pd.DataFrame,
                 target_col: str,
                 input_cols: list,
                 
                 target_lag: int=50,
                 input_lags: int=10,

                 train_pct: int=0.5,
                 lasso: bool=False,

                 alpha: float=0.05,
                 rolling_regression: bool=False,
                 rolling_regression_lookback: bool=False,
                 running_lasse: bool=False,
                 use_actual_change: bool=False,

                 jump_days: int = 10,

                 predecided_cols = []
                 ):
        self.data_model = data_model
        self.target_col = target_col
        self.input_cols = input_cols
        self.target_lag = target_lag
        self.input_lags = input_lags
        self.train_pct = train_pct
        self.lasso = lasso
        self.alpha = alpha
        self.use_actual_change = use_actual_change

        self.swaption_details = swaption_data.swaption_details
        self.pv_swaprate = swaption_data.pv_swaprate

        self.jump_days = jump_days
        

        # Define lagged
        self.df_lagged = self.create_lags(self.data_model.copy(), self.target_col, self.input_cols, self.input_lags, self.target_lag, use_delta=True)
        
        # Cols
        self.target_col = f"target_{self.target_col}_future{self.target_lag}"
        self.feature_cols = [c for c in self.df_lagged.columns if c != self.target_col]
        # Initial length
        self.init_train_length = int(len(self.df_lagged) * self.train_pct)
        self.init_test_length = len(self.df_lagged) - self.init_train_length - self.target_lag

        # Run first lasso selection
        self.selected_features = self.feature_cols
        
        if len(predecided_cols) > 0:
            self.selected_features = predecided_cols

        elif lasso:
            self.selected_features = self.input_selection(self.df_lagged[:self.init_train_length])

        self.X_test, self.y_test = self.X_y_split(self.df_lagged[-self.init_test_length:],selected_cols=self.selected_features)

        step = self.jump_days
        n_obs = len(self.df_lagged)
        T    = self.init_train_length
        
        ## Defines the type of regression to be done
        if rolling_regression_lookback:
            # both start and end move forward in jumps of self.jump_days
            starts = range(0, n_obs - T + 1, step)
            self.results = [
                self.fit_AR_X(
                    self.df_lagged.iloc[i : i + T],
                    selected_cols=self.selected_features
                )
                for i in starts
            ]

        elif rolling_regression:
            # only the end moves forward in jumps of self.jump_days
            ends = range(T, n_obs + 1, step)   # end = i+T  => i = end-T
            self.results = [
                self.fit_AR_X(
                    self.df_lagged.iloc[: end],
                    selected_cols=self.selected_features
                )
                for end in ends
            ]
        else:
            self.results = [self.fit_AR_X(df_lagged=self.df_lagged[:self.init_train_length],selected_cols=self.selected_features)]

        self.stats()


    def create_lags(self,
                    df: pd.DataFrame,
                    target_col: str,
                    input_cols: list[str],
                    input_lag_order: int,
                    target_lag_order: int,
                    use_delta: bool=False) -> pd.DataFrame:
        """
        Create lagged features for both input variables and a future target.
        Uses a single pd.concat to avoid fragmentation.
        """
        # 1) Collect all new columns in a dict
        new_cols: dict[str, pd.Series] = {}

        # input lags: col_lag1, col_lag2, …
        for col in input_cols:
            for lag in range(1, input_lag_order + 1):
                new_cols[f"{col}_lag{lag}"] = df[col].shift(lag) if not use_delta else df[col] - df[col].shift(lag) 

        tgt = f"target_{target_col}_future{target_lag_order}"
        
        # future target: target_<name>_futureN
        new_cols[tgt] = \
            df[target_col].shift(-target_lag_order) if not use_delta else df[target_col].shift(-target_lag_order) - df[target_col]

        # LOOKUP ACTUAL RANGE CHANGE NOT NOT USE THE NEW GENERIC SWAP AS PROXY
        if self.use_actual_change:
            
            swaps = self.swaption_details[self.swaption_details.index.isin(df.index)].xInstrumentSwap
            posistions = self.pv_swaprate[self.pv_swaprate.xInstrumentSwap.isin(swaps)].copy().groupby('xInstrumentSwap', as_index=False).nth(self.target_lag-1)
            posistions.index = swaps.index

            new_cols[tgt] = posistions.swaprate * 10_000 - df[target_col]

        # 2) Concat once
        df = pd.concat(
            [df[input_cols], pd.DataFrame(new_cols, index=df.index)],
            axis=1
        )

        # df = pd.DataFrame(new_cols,index=df.index)

        # 3) Drop any rows with NaNs (from shifts)
        df = df.dropna(subset=(df.columns[:-1] ))
        # 4) Return
        return df

    def X_y_split(self,df_lagged,selected_cols: list=[]) ->tuple[pd.DataFrame, pd.Series]:
        """ Returns X and y for a given lagged df"""
        if len(selected_cols) == 0:
            return df_lagged[self.feature_cols], df_lagged[self.target_col]

        return df_lagged[selected_cols], df_lagged[self.target_col]

    def fit_AR_X(self, df_lagged,selected_cols: list=[]):
        
        X_train, y_train = self.X_y_split(df_lagged,selected_cols)

        model = sm.OLS(y_train,X_train)
        results = model.fit()
        return results

    def input_selection(self,df_lagged):

        X_train, y_train = self.X_y_split(df_lagged)
            
        pipeline = make_pipeline(
            StandardScaler(),
            LassoCV(
                cv=5,           # 5-fold CV
                n_alphas=100,   # number of α’s to try
                #max_iter=100_000,
                max_iter=20_000,
                random_state=0
            )
        )

        # 3. Fit it
        pipeline.fit(X_train, y_train)

        # 4. Recover the fitted Lasso from the pipeline
        lasso = pipeline.named_steps['lassocv']

        # 5. Wrap it in a selector
        selector = SelectFromModel(lasso, prefit=True, threshold=1e-8)

        # 6. Get boolean mask of kept features and list their names
        mask = selector.get_support()             # array of True/False
        selected_features = X_train.columns[mask].tolist()

        print("\nFinal selected feature set:")
        print(selected_features)

        return selected_features

    def plot_predictions(self,summary):
        # align swaprate to only those dates in df
        swap = self.data_model.loc[self.data_model.index.intersection(summary.index), "swaprate"]

        fig, ax = plt.subplots(figsize=(8, 4))
        # primary plots
        ax.plot(summary.index, summary["mean"],          linestyle='-',  color='black', label='Mean')
        ax.plot(summary.index, summary["obs_ci_lower"],  linestyle='--', color='blue',  label='CI Lower')
        ax.plot(summary.index, summary["obs_ci_upper"],  linestyle='--', color='blue',  label='CI Upper')

        ax.plot(summary.index, summary["act"],  linestyle='dotted', color='black',  label='Delta trying to predict')

        # find where entire CI is positive or negative
        pos = summary["obs_ci_lower"] > 0     # all positive
        neg = summary["obs_ci_upper"] < 0     # all negative

        # helper to shade contiguous True‐runs
        def shade_regions(mask, color):
            grp = (mask != mask.shift(1)).cumsum()
            for _, segment in summary[mask].groupby(grp):
                start, end = segment.index[0], segment.index[-1]
                ax.axvspan(start, end, color=color, alpha=0.3)

        shade_regions(pos, 'lightgreen')
        shade_regions(neg, 'lightcoral')

        # secondary y-axis for swaprate
        ax2 = ax.twinx()
        ax2.plot(swap.index, swap.values, linestyle='-', color='red', label='Swap Rate')
        ax2.plot(summary.index, summary.cum_performance * 100, linestyle='--', color='red', label='Cum Performance')
        ax2.set_ylabel('Swap Rate', fontsize=10)
        
        # axes formatting
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Mean ± CI bounds', fontsize=10)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
        fig.autofmt_xdate()

        # combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.tight_layout()
        plt.show()

    def stats(self):
        """
        This
        
        A prediction interval for an actual future observation. It combines both
            * the uncertainty in the mean forecast (i.e. mean_se), and
            * the model’s residual noise variance (the white-noise or shock term).
        """
        results = self.results
        
        ### Running predictions ###
        summary = results[0].get_prediction(self.X_test).summary_frame(alpha=self.alpha)
        if len(results) > 1:
            summary = pd.DataFrame()

            for i in range(len(results)):
                
                from_day = (i) * self.jump_days
                
                summary = pd.concat([
                    summary,
                    results[i].get_prediction(
                        self.X_test.iloc[from_day : from_day+self.jump_days])
                        .summary_frame(alpha=self.alpha)]
                        ,axis=0)

        summary["act"] = self.y_test
        summary = summary.dropna()
        summary["corr"] = summary["mean"].expanding().corr(summary["act"])

        # print(summary)

        # 1) mask of “strong” predictions whose lower‐ and upper‐CI share the same sign
        mask = np.sign(summary.obs_ci_lower) == np.sign(summary.obs_ci_upper)
        # 2) the dates (index) of those rows
        strong_dates = summary.index[mask]
        # 3) pull out the true and fitted signs on exactly those dates
        y_true_strong = np.sign(self.y_test.loc[strong_dates])
        y_pred_strong = np.sign(summary["mean"].loc[strong_dates])

        # 4) compute accuracy
        strong_accuracy = accuracy_score(y_true_strong, y_pred_strong)
        print(f"Accuracy on ’strong‐CI’ dates ({len(strong_dates)} points): {strong_accuracy:.3f}")

        ### ACTUAL PERFORMANCE ###
        relevant_details = self.swaption_details[self.swaption_details.index.isin(strong_dates)].copy()
        traded_swaps = relevant_details.xInstrumentSwap

        # 2) build a dict {instrument_name: prediction}
        pred_map = dict(zip(traded_swaps, y_pred_strong))

        posistions = self.pv_swaprate[self.pv_swaprate.xInstrumentSwap.isin(traded_swaps)].copy()
        posistions['y_pred_strong'] = None; posistions.loc[:,'y_pred_strong'] = posistions['xInstrumentSwap'].map(pred_map).squeeze()  ## Long or short?
        posistions["pv_long_short"] = -posistions['y_pred_strong'] * posistions.pv ## Long or short?

        rolling_pv = posistions.groupby('xInstrumentSwap', as_index=False).head(self.target_lag-1) ## THIS IS HOW LONG WE ARE TRYING TO PREDICT INTO THE FUTURE
        closing_pv = posistions.groupby('xInstrumentSwap', as_index=False).nth(self.target_lag-1)


        # 1) daily sum of rolling PV
        daily_sum = rolling_pv.groupby('date')['pv_long_short'].sum()
        # 2) cumulative sum of closing PV
        cum_closing = closing_pv.groupby('date')['pv_long_short'].sum().cumsum()

        # 3) merge on the date‐index and sort
        df_merged = pd.concat(
            [daily_sum.rename('pv_open_positions'), 
            cum_closing.rename('closed_posistion')],
            axis=1
        ).sort_index()

        df_merged["closed_posistion"] = df_merged.closed_posistion.ffill()
        df_merged["cum_performance"] = df_merged.sum(axis=1)

        summary = summary.join(df_merged, how="left")
        summary = summary.ffill()
        summary = summary.fillna(0)

        summary["performance"] = summary["cum_performance"] * 100
        summary["daily_pnl"] = (summary["performance"] - summary["performance"].shift(1)).fillna(0)

        mean_pnl  = summary["daily_pnl"].mean()
        std_pnl   = summary["daily_pnl"].std(ddof=1)
        sharpe    = mean_pnl / std_pnl            # daily Sharpe
        sharpe_ann= sharpe * np.sqrt(252)         # annualized

        print(f'Sharpe: {sharpe_ann:.3f}')

        print(summary)

        self.plot_predictions(summary)

        # plt.plot(summary["cum_performance"])
        # plt.show()


AR_X_oneM_model =  AR_X_REGRESSION_MODEL(data_model=data_model,target_col="swaprate",input_cols=data_model.columns[[1,2,3,4,19,-1]],train_pct=0.5,input_lags=10,target_lag=50,alpha=0.05,swaption_data=oneM,rolling_regression=True)





########## PLOTS ##########
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern', 'DejaVu Serif']


class plot_datasets:
    def __init__(self,
                swaption_data: get_swaption_data,
                RND_data: get_RNDs,
                stats_data: get_stats
                ):
        
        self.dates = swaption_data.dates
        self.swaption_data = swaption_data
        self.RND_data = RND_data
        self.dens = pd.concat([pd.DataFrame(row.solution) for _, row in RND_data.distr.iterrows()], axis=1)
        self.stats_data = stats_data

    def swaprate(self):
        plt.figure(figsize=(8, 4))  
        plt.plot(self.dates, (self.swaption_data.swaption_details.swaprate)*100, linestyle='-', color='black')
        plt.xlabel('Date', fontsize=10)
        plt.ylabel(f'Swaprate (%)', fontsize=10)
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1))) # , interval=15
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
        plt.gcf().autofmt_xdate()
        plt.savefig(f'{figures_path}\swaprate_{self.swaption_data.swapstart}_{self.swaption_data.swapend}.jpg', dpi=600, bbox_inches='tight')
        plt.show()

    def fwd_price(self):
        plt.figure(figsize=(8, 4))  
        plt.plot(self.swaption_data.fwd_price.columns, self.swaption_data.fwd_price.values.T, linestyle='-') # , color='black')
        plt.xlabel('Date', fontsize=10)
        plt.ylabel(f'Swaption Forward Price (%%)', fontsize=10)
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1)))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
        plt.gcf().autofmt_xdate()
        # plt.savefig(f'{figures_path}\forward_price_{self.swaption_data.swapstart}_{self.swaption_data.swapend}.jpg', dpi=600, bbox_inches='tight')
        plt.show()

    def impvol(self):
        plt.figure(figsize=(8, 4))  
        plt.plot(self.swaption_data.impvol.columns, self.swaption_data.impvol.values.T, linestyle='-') # , color='black')
        plt.xlabel('Date', fontsize=10)
        plt.ylabel(f'Swaption Implied-Volatility (‱)', fontsize=10)
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1)))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
        plt.gcf().autofmt_xdate()
        #plt.savefig('sp500.jpg', dpi=300, bbox_inches='tight')
        plt.show()

    def stats(self,rows: int=2,cols: int=2):
        # Get x-axis dates and metric names (excluding 'date')
        dates = self.stats_data.stats["date"]
        metrics = self.stats_data.stats.columns.drop("date")
        
        # Create a 4 x 1 subplot figure (adjust figsize as needed)
        fig, axs = plt.subplots(rows, cols, figsize=(8, 4), sharex=True)
        axs = axs.flatten()
        
        # Loop over corresponding axes and metrics simultaneously
        for ax, metric in zip(axs, metrics):
            ax.plot(dates, self.stats_data.stats[metric], linestyle='-', color='black')
            ax.set_ylabel(metric, fontsize=10)
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1)))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
        
        fig.supxlabel("Date", fontsize=10)
        plt.gcf().autofmt_xdate()  # Ensure date labels are readable
        plt.savefig(f'{figures_path}\stats_{self.swaption_data.swapstart}_{self.swaption_data.swapend}.jpg', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.show()

    def density_surface(self,tick_interval: int=50):
        datess = pd.to_datetime(self.stats_data.stats["date"])
        # Generate date labels with only month and year
        date_labels = datess.dt.strftime('%m-%Y').tolist()
        time_numeric = np.arange(len(date_labels))
        Z = (self.dens*100).values
        Y, X = np.meshgrid(time_numeric, self.RND_data.scenario)  # Now Y is time, X is return percentages
        Z_masked = np.where((X >= -100) & (X <= 100), Z, np.nan)
        colors = [(0.2, 0.4, 0.8), (0.8, 0.2, 0.4)]  # Example colors, adjust as needed
        cmap_name = 'custom_cmap'
        custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z_masked, cmap=custom_cmap, edgecolor='none')
        ax.set_xlabel('Moneyness (‱)', fontsize=14)
        ax.set_ylabel('Date', fontsize=14)
        ax.set_zlabel('Density (%)', fontsize=14)
        ax.set_xlim(-100, 100)

        filtered_ticks = np.linspace(0, len(date_labels)-1, len(date_labels))[::tick_interval]
        filtered_labels = date_labels[::tick_interval]
        ax.set_yticks(filtered_ticks)
        ax.set_yticklabels(filtered_labels, fontsize=12)
        ax.yaxis.labelpad = 50
        # Rotate the plot (example: azimuth=220, elevation=20)
        ax.view_init(azim=230, elev=20)
        plt.savefig(f'{figures_path}\density_surface_{self.swaption_data.swapstart}_{self.swaption_data.swapend}.jpg', dpi=600, bbox_inches='tight',pad_inches=0.2)
        plt.show()

    def impvol_surface(self,tick_interval: int=50):
        minimum,maximum = min(self.swaption_data.strikes),max(self.swaption_data.strikes)
        
        datess = pd.to_datetime(self.stats_data.stats["date"])
        # Generate date labels with only month and year
        date_labels = datess.dt.strftime('%m-%Y').tolist()
        time_numeric = np.arange(len(date_labels))
        Z = (self.swaption_data.impvol.values)
        Y, X = np.meshgrid(time_numeric, self.swaption_data.strikes)  # Now Y is time, X is return percentages
        Z_masked = np.where((X >= minimum) & (X <= maximum), Z, np.nan)
        colors = [(0.2, 0.4, 0.8), (0.8, 0.2, 0.4)]  # Example colors, adjust as needed
        cmap_name = 'custom_cmap'
        custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z_masked, cmap=custom_cmap, edgecolor='none')
        ax.set_xlabel('Strikes', fontsize=14)
        ax.set_ylabel('Date', fontsize=14)
        ax.set_zlabel('Implied Volatility', fontsize=14)
        ax.set_xlim(minimum, maximum)

        filtered_ticks = np.linspace(0, len(date_labels)-1, len(date_labels))[::tick_interval]
        filtered_labels = date_labels[::tick_interval]
        ax.set_yticks(filtered_ticks)
        ax.set_yticklabels(filtered_labels, fontsize=12)
        ax.yaxis.labelpad = 50
        # Rotate the plot (example: azimuth=220, elevation=20)
        ax.view_init(azim=230, elev=20)
        plt.savefig(f'{figures_path}\impvol_surface_{self.swaption_data.swapstart}_{self.swaption_data.swapend}.jpg', dpi=600, bbox_inches='tight',pad_inches=0.2)
        plt.show()

plot = plot_datasets(swaption_data=oneM,RND_data=rndoneM,stats_data=statsoneM)
plot.swaprate()
plot.stats()
plot.density_surface(tick_interval=int(257*2*0.75))
plot.impvol_surface(tick_interval=int(257*2*0.75))
plot.fwd_price()
plot.impvol()

def plot_swap_pv(nr: int=1):
    swap_pv = oneM.pv_swaprate[oneM.pv_swaprate.swaption == oneM.pv_swaprate.swaption.unique()[nr]]
    plt.figure(figsize=(8, 4))
    plt.plot(swap_pv.date, (swap_pv.pv)*1_000_000, linestyle='-', color='black')
    #plt.plot(swap_pv.date, (swap_pv.swaprate)*100, linestyle='-', color='grey')
    plt.xlabel('Date', fontsize=10)
    plt.ylabel(f'Present Value', fontsize=10)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=None, interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    plt.gcf().autofmt_xdate()
    plt.savefig(f'{figures_path}\pv_swap.jpg', dpi=600, bbox_inches='tight')
    plt.show()

plot_swap_pv()

def plot_cashflow(nr: int=1):
    swaps = oneM.pv_swaprate[oneM.pv_swaprate.swaption == oneM.pv_swaprate.swaption.unique()[nr]]
    
    instrument = swaps.xInstrumentSwap.iloc[0]
    
    for metric in ["min","max"]:
        date = swaps.date.min() if metric == "min" else swaps.date.max()

        ## hidden ##
        cashflow.columns = cashflow.iloc[0]
        cashflow = cashflow.iloc[1:].reset_index(drop=True).copy()

        cashflow["constant"] = cashflow.cashflow.apply(lambda x: float(x.split(" ")[0]))
        ## hidden ##
        ## hidden ##

        cashflow["floating_fixings"] = cashflow["fixings"] * cashflow["constant"]
        cashflow["fixed_leg"] = cashflow["fv"] - cashflow["floating_fixings"]
        cashflow["fixed_leg"] = cashflow["fixed_leg"].apply(lambda x: np.nan if abs(x)<0.001 else x)

        cashflow["fixed_leg_fixings"] = cashflow["fixed_leg"].bfill()

        cashflow.index = cashflow.paydate
        
        fig, ax1 = plt.subplots(figsize=(8, 4))
        bar_w = 20  # days
        # 1) Bars for fixed & floating
        ax1.bar(cashflow.index, cashflow['fixed_leg'],    width=bar_w, color='blue', label='fixed',    align='center')
        ax1.bar(cashflow.index, cashflow['floating_fixings'],    width=bar_w, color='red', label='fixed',    align='center')

        # 2) Lines for fv & pv on same left axis
        ax1.plot(cashflow.index, cashflow['fixings'], marker=".", color='grey',    label='Projected Fixings')
        ax1.plot(cashflow.index, cashflow['fixed_leg_fixings'], '-', color='darkgrey',    label='Fixed Leg')
        ax1.plot(cashflow.index, cashflow['pv'].cumsum(), '-', color='black',    label='Cummulative PV')

        plt.xlabel('Date', fontsize=10)
        plt.ylabel(f'Cashflow', fontsize=10)

        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=None, interval=6))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))

        plt.gcf().autofmt_xdate()
        plt.savefig(f'{figures_path}\cashflow_{metric}.jpg', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.show()

plot_cashflow(nr=10)

def plot_yield_curve(nr: int=-1):
    input = ratemodels[ratemodels.date == ratemodels.date.unique()[nr]]

    plt.figure(figsize=(8, 4))  
    plt.plot(input['tenor_date'].loc[input['estr'].notna()], input['estr'].dropna(), linestyle='-',marker=".", color="black", label='ESTR')
    plt.plot(input['tenor_date'].loc[input['eur6m'].notna()], input['eur6m'].dropna(), linestyle='-',marker=".", color="grey", label='EURIBOR 6M')
    plt.xlabel('Tenor', fontsize=10)
    plt.ylabel(f'Yield (%)', fontsize=10)

    # only put ticks at tenors where eur6m is not NaN
    mask = input['eur6m'].notna()
    ticks     = input.loc[mask, 'tenor_date']
    ticklabels = input.loc[mask, 'tenor']
    # drop positions 0, 2 and 4
    drop_positions = {0, 2, 4}
    keep_positions = [i for i in range(len(ticks)) if i not in drop_positions]
    ticks     = ticks.iloc[keep_positions]
    ticklabels    = ticklabels.iloc[keep_positions]

    plt.gca().set_xticks(ticks)
    plt.gca().set_xticklabels(ticklabels, rotation=45)
    plt.gca().margins(x=0.01)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{figures_path}\yield_curve_snap.jpg', dpi=600, bbox_inches='tight')
    plt.show()

plot_yield_curve(-1)

def plot_yield_surface(ref_index: str="estr", index_pivot=pd.DataFrame):

    if index_pivot.empty:
        index_pivot = ratemodels.pivot(index="date",columns="tenor",values=ref_index)

    sort = index_pivot.columns.to_series().apply(
                lambda x:
                float(x[:-1]) / 12
                if x.endswith("M")   # months → years
                else float(x[:-1]) / 52
                if x.endswith("W")   # weeks → years
                else float(x[:-1])   # years)
                ).sort_values()

    index_pivot = index_pivot.reindex(columns=sort.index)

    # 1) Build your “knots” array in years from the column names:
    knots = sort.values
    # 2) Define a row‐wise interpolation function
    def interp_row(row_values):
        mask     = ~np.isnan(row_values)
        # if fewer than two points, just return the row unchanged
        if mask.sum() < 2:
            return row_values
        # interp at every “knots” position, filling in the NaNs
        return np.interp(knots, knots[mask], row_values[mask],left=np.nan, right=np.nan)

    # 3) Apply to every row and reassemble a DataFrame
    filled = index_pivot.apply(lambda row: interp_row(row.values), axis=1)
    index_pivot = pd.DataFrame(
        list(filled), 
        index=index_pivot.index, 
        columns=index_pivot.columns
    )

    tenors = knots
    minimum,maximum = tenors.min(), tenors.max()

    tick_interval = 275

    datess = pd.to_datetime(index_pivot.index)
    date_labels = datess.strftime('%m-%Y').tolist()
    time_numeric = np.arange(len(date_labels))
    Z = (index_pivot.values).T
    Y, X = np.meshgrid(time_numeric, tenors)  # Now Y is time, X is return percentages

    Z_masked = np.where((X >= minimum) & (X <= maximum), Z, np.nan)
    colors = [(0.2, 0.4, 0.8), (0.8, 0.2, 0.4)]  # Example colors, adjust as needed
    cmap_name = 'custom_cmap'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z_masked, cmap=custom_cmap, edgecolor='none')
    ax.set_xlabel('Tenors', fontsize=14)
    ax.set_ylabel('Date', fontsize=14)
    ax.set_zlabel('Yield (%)', fontsize=14)
    ax.set_xlim(minimum, maximum)
        
    # TENORS FILTERING
    keep_positions = (np.isclose(tenors, np.round(tenors)) & np.isclose(tenors/5, np.round(tenors/5))) | (sort.index == "1Y")
    ticks     = tenors[keep_positions]
    ticklabels    = sort.index[keep_positions]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)

    filtered_ticks = np.linspace(0, len(date_labels)-1, len(date_labels))[::tick_interval]
    filtered_labels = date_labels[::tick_interval]
    ax.set_yticks(filtered_ticks)
    ax.set_yticklabels(filtered_labels, fontsize=12)
    ax.yaxis.labelpad = 50
    # Rotate the plot (example: azimuth=220, elevation=20)
    ax.view_init(azim=230, elev=40) if ref_index =="basis" else ax.view_init(azim=230, elev=20)
    plt.tight_layout()
    plt.savefig(f'{figures_path}\yield_surface_{ref_index}.jpg', dpi=600, bbox_inches='tight',pad_inches=0.2)
    plt.show()

    return index_pivot

index_pivot_estr = plot_yield_surface("estr")
index_pivot_eur6m = plot_yield_surface("eur6m")
index_pivot_basis = plot_yield_surface(ref_index="basis",index_pivot=index_pivot_eur6m - index_pivot_estr)

def volatility_surface(nr:int=-1,option_tenor="1Y"):
    voldata = model[(model.date == model.date.unique()[nr]) & (model.option_tenor == option_tenor)]
    
    voldata_pivot = pd.pivot(data=voldata, index="swap_tenor",columns="moneyness",values="volatility")*100

    sort = voldata_pivot.index.to_series().apply(
                lambda x:
                float(x[:-1]) / 12
                if x.endswith("M")   # months → years
                else float(x[:-1]) / 52
                if x.endswith("W")   # weeks → years
                else float(x[:-1])   # years)
                ).sort_values()

    voldata_pivot = voldata_pivot.reindex(index=sort.index)

    knots = sort.values

    tenors = knots
    minimum,maximum = tenors.min(), tenors.max()

    Z = (voldata_pivot.values)
    Y, X = np.meshgrid(voldata_pivot.columns, tenors)  # Now Y is time, X is return percentages

    Z_masked = np.where((X >= minimum) & (X <= maximum), Z, np.nan)
    colors = [(0.2, 0.4, 0.8), (0.8, 0.2, 0.4)]  # Example colors, adjust as needed
    cmap_name = 'custom_cmap'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z_masked, cmap=custom_cmap, edgecolor='none')
    ax.set_xlabel('Swap Tenor', fontsize=14)
    ax.set_ylabel('Moneyness', fontsize=14)
    ax.set_zlabel('Implied-Volatility', fontsize=14)
    ax.set_xlim(minimum, maximum)

    ax.set_xticks(tenors)
    ax.set_xticklabels(voldata_pivot.index)
        
    # ax.set_yticklabels(fontsize=12)
    ax.yaxis.labelpad = 50
    # Rotate the plot (example: azimuth=220, elevation=20)
    ax.view_init(azim=230, elev=20)
    plt.tight_layout()
    plt.savefig(f'{figures_path}/vol_surface_{option_tenor}.jpg', dpi=600, bbox_inches='tight',pad_inches=0.2)
    plt.show()

    return voldata_pivot

voldata_pivot = volatility_surface(nr=-1,option_tenor="1Y")
