# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 12:38:19 2020

@author: ioann
"""
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo
import chart_studio.plotly as py
import chart_studio


from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import fsolve



def logistic(t, a, b, c, d):
    return c + (d - c)/(1 + a * np.exp(- b * t))
   
def exponential(t, a, b, c):
    return a * np.exp(b * t) + c

def quadratic(x,a,b,c):
    return(a*x^2+b*x+c)


def plotCases(dataframe, column, country):
    
    co = dataframe[dataframe[column] == country].iloc[:,4:].T.sum(axis = 1)
    co = pd.DataFrame(co)
    co.columns = ['Cases']
    co = co.loc[co['Cases'] >0]
    
    y = np.array(co['Cases'])
    x = np.arange(y.size)
    
    print('** ',country,' **')    
    recentdbltime = float('NaN')
    ed='NaN'
    casesoned=0
    if len(y) >= 10: #we have at least 10 days of cases reported
        
        current = y[-1]
        lastweek = y[-8]
        
        if current >= lastweek:
            print('\n** Based on Most Recent Week of Data **\n')
            print('\tConfirmed cases on',co.index[-1],'\t',current)
            print('\tConfirmed cases on',co.index[-8],'\t',lastweek)
            ratio = current/lastweek
            print('\tRatio:',round(ratio,2))
            print('\tWeekly increase:',round( 100 * (ratio - 1), 1),'%')
            dailypercentchange = round( 100 * (pow(ratio, 1/7) - 1), 1)
            print('\tDaily increase:', dailypercentchange, '% per day')
            recentdbltime = round( 7 * np.log(2) / np.log(ratio), 1)
            print('\tDoubling Time (represents recent growth):',recentdbltime,'days')

            plt.figure(figsize=(10,5))
            plt.plot(x, y, 'ko', label="Original Data")
            
            logisticworked = False
            exponentialworked = False
            if y[-1]>1000:
                try:
                    lpopt, lpcov = curve_fit(logistic, x, y, maxfev=10000)
                    lerror = np.sqrt(np.diag(lpcov))
                    a=lpopt[0]
                    b=lpopt[1]
                    c=lpopt[2]
                    d=lpopt[3]
                    # for logistic curve at half maximum, slope = growth rate/2. so doubling time = ln(2) / (growth rate/2)
                    ldoubletime = np.log(2)/(lpopt[1]/2)
                    # standard error
                    ldoubletimeerror = 1.96 * ldoubletime * np.abs(lerror[1]/lpopt[1])
                    # calculate R^2
                    residuals = y - logistic(x, *lpopt)
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((y - np.mean(y))**2)
                    logisticr2 = 1 - (ss_res / ss_tot)  
                    pred_x15= list(range(max(x)+1, max(x)+15))
                    pred_y_log=[round(logistic(i,a,b,c,d),0) for i in pred_x15]
                    logisticworked = True 
                   
                except:
                    logisticworked = False
                    pass
            else:
                try:
                    epopt, epcov = curve_fit(exponential, x, y, bounds=([0,0,-100],[100,0.9,100]), maxfev=10000)
                    eerror = np.sqrt(np.diag(epcov))
                    a=epopt[0]
                    b=epopt[1]
                    c=epopt[2]
                    # for exponential curve, slope = growth rate. so doubling time = ln(2) / growth rate
                    edoubletime = np.log(2)/epopt[1]
                    # standard error
                    edoubletimeerror = 1.96 * edoubletime * np.abs(eerror[1]/epopt[1])
                    
                    # calculate R^2
                    residuals = y - exponential(x, *epopt)
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((y - np.mean(y))**2)
                    expr2 = 1 - (ss_res / ss_tot)
                    pred_x15= list(range(max(x)+1, max(x)+15))
                    pred_y_exp=[round(exponential(i,a,b,c),0) for i in pred_x15]
                    exponentialworked = True
                    
                except:
                    exponentialworked = False
                    pass
            
            if logisticworked and logisticr2 > 0.95:
                plt.plot(x, logistic(x, *lpopt), 'b--', label="Logistic Curve Fit")
                plt.scatter(pred_x15,pred_y_log,label="Extrapolated data",color="green")
                print('\n** Based on Logistic Fit**\n')
                print('\tR^2:', logisticr2)
                print('\tDoubling Time (during middle of growth): ', round(ldoubletime,2), '(±', round(ldoubletimeerror,2),') days')
                sol = int(fsolve(lambda x : logistic(x, a, b, c, d) - int(d), 1))
                print('\tEnd Date on day No:',sol,'meaning in another',sol-y.size,'days and inflection point at day No ',int(np.log(a)/b))
                ed=datetime.strptime(mostrecentdate,'%m/%d/%y')+timedelta(days=sol-y.size)
                ed=ed.strftime('%d/%m/%Y')
                print('\tEnd Date is estimated on',ed)
                print('\tEstimated Total number of cases between [',int(logistic(sol, a, b, c, d)),',',int(logistic(sol, a, b, c, d)+lerror[3]),']')
                casesoned=int(logistic(sol, a, b, c, d)+lerror[3])
                
            if  exponentialworked and (logisticworked==False or expr2 > logisticr2): #0.95 logisticworked==False
                plt.plot(x, exponential(x, *epopt), 'r--', label="Exponential Curve Fit")
                plt.scatter(pred_x15,pred_y_exp,label="Extrapolated data",color="green")
                print('\n** Based on Exponential Fit **\n')
                print('\tR^2:', expr2)
                print('\tDoubling Time (represents overall growth): ', round(edoubletime,2), '(±', round(edoubletimeerror,2),') days')
                print('\tSince the data best fit on the exponential model End Date cannot be found')
                print('\tBut can extrapolate and predict the number of cases for the next 14 days, which are:')
                print('\t',pred_y_exp[:7])
                print('\t',pred_y_exp[7:])
                print('\tEvolution of the Ratios for next week based on estimated cases')
                print('\t',round(pred_y_exp[0]/y[-7],2), round(pred_y_exp[1]/y[-6],2),round(pred_y_exp[2]/y[-5],2), round(pred_y_exp[3]/y[-4],2))
                print('\t',round(pred_y_exp[4]/y[-3],2), round(pred_y_exp[5]/y[-2],2),round(pred_y_exp[6]/y[-1],2), round(pred_y_exp[7]/pred_y_exp[0],2))
                ed='NaN'
                casesoned=0 
                    
            plt.title(country + ' Cumulative COVID-19 Cases. (Updated on '+mostrecentdate+')', fontsize="x-large")
            plt.xlabel('Days', fontsize="x-large")
            plt.ylabel('Total Cases', fontsize="x-large")
            plt.legend(fontsize="x-large")
            plt.show()
    
            if logisticworked and exponentialworked:
                if logisticr2 > expr2:
                    return [ldoubletime, ldoubletimeerror, recentdbltime, ed, casesoned, round(y[-1]/y[-8],2)]
                else:
                    return [edoubletime, edoubletimeerror, recentdbltime, ed, casesoned, round(y[-1]/y[-8],2)]
                    
            if logisticworked and exponentialworked==False:
                return [ldoubletime, ldoubletimeerror, recentdbltime, ed,  casesoned, round(y[-1]/y[-8],2)]
            
            if exponentialworked and logisticworked==False:
                return [edoubletime, edoubletimeerror, recentdbltime, ed, casesoned, round(y[-1]/y[-8],2)]
            
            if exponentialworked==False and logisticworked==False:
                return [float('NaN'), float('NaN'), recentdbltime,'NaN',0, round(y[-1]/y[-8],2)]


url='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
df = pd.read_csv(url)
cases = df.iloc[:,[1,-1]].groupby('Country/Region').sum()
mostrecentdate = cases.columns[0]
print('\nTotal number of cases (in countries with at least 300 cases) as of', mostrecentdate)

cases = cases.sort_values(by = mostrecentdate, ascending = False)
cases = cases[cases[mostrecentdate] >= 300]

world=pd.DataFrame()
world['TotCases']=(cases[mostrecentdate])
topcountries = cases.index
world=world.reset_index()
world=world.rename(columns={'Country/Region':'country'})


inferreddoublingtime = []
recentdoublingtime = []
errors = []
countries = []
enddates=[]
endcases=[]
roc=[]
print('\n')

for c in topcountries:
    
    a = plotCases(df, 'Country/Region', c)
    if a:
        countries.append(c)
        inferreddoublingtime.append(round(a[0],2))
        errors.append(a[1])
        recentdoublingtime.append(a[2])
        enddates.append(a[3])
        endcases.append(a[4])
        roc.append(a[5])
    print('\n')    




world['roc']=recentdoublingtime
world['inferedroc'] =inferreddoublingtime
world['End Date']=enddates
world['EstTotCases']=endcases
world['Ratio']=roc


chart_studio.tools.set_credentials_file(username='johnpsom', api_key='3G8v7LAuci6P1OfllgmM')

figure1 = px.choropleth(data_frame=world,
                        locations="country",  
                        locationmode='country names',
                        color="roc", 
                        hover_name="country",
                        range_color=[1,10], 
                        hover_data=['country','roc','inferedroc','End Date','EstTotCases'],
                        color_continuous_scale="rdylgn",
                        labels={'roc':'Recent Doubling Time',
                                'End Date':'Est. End Date',
                                'EstTotCases':'Est. Total Cases',
                                'inferedroc':'Infered Doubling Time'},
                        title='Doubling Times of Total Confirmed Cases')

figure1.update_layout()
pyo.plot(figure1,show_link=True,include_plotlyjs=True)
#py.plot(figure1, filename = 'world_doublingrates', auto_open=True)  

figure2 = px.choropleth(world, locations="country", 
                        locationmode='country names',
                        color="Ratio", 
                        hover_name="country",
                        range_color=[1,3], 
                        hover_data=['country','Ratio','TotCases'],
                        color_continuous_scale="rdylgn_r",
                        labels={'Ratio':'Latest Weekly Ratio',
                                'TotCases':'Latest Total Cases'},
                        title='Rate of Change in Total Confirmed Cases')
figure2.update_layout()
pyo.plot(figure2,show_link=True,include_plotlyjs=True)
#py.plot(figure2, filename = 'World_Weekly_Ratios', auto_open=True)    
