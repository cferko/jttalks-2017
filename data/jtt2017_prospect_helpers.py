'''
Define key functions for the JT Talks 2017 Prospect Theory Jupyter notebooks.
Adam Dachowicz, December 17 2017
'''
import warnings
warnings.filterwarnings("ignore")

# build the dictionary of decision maker response data for curve fitting and analysis.
def getTeamData(nameList, responses, 
               minGrade = 40., 
               maxGrade = 100.): 
    import numpy as np
    teamData = {}
    # teamData['Y'] = np.linspace(0., 1., datapoints)
    for name in nameList:
        if 'myname' not in name.lower():
            for r in responses:
                if r[-1] == name:
                    data = r[:-1]
            a = np.array([ float(point) for point in data ])
            # if maxGrade < minGrade:
            #     a = np.insert(a,0,maxGrade) # 0 hours for the first outcome 
            #     a = np.insert(a,len(a),minGrade) # 0 hours for the last outcome 
            # else:
            a = np.insert(a,0,minGrade) # 0 hours for the first outcome 
            a = np.insert(a,len(a),maxGrade) # 0 hours for the last outcome 
            teamData[name] = a
            teamData['Y'] = np.linspace(0., 1., len(a))
    return teamData

# define a function to normalize an input array for function fitting/risk analysis.
def normalizeArray(X):
    import numpy as np
    minX = min(X)
    maxX = max(X)
    normed = np.array( [(x-minX)/(maxX-minX) for x in X] )
    return [minX, maxX, normed]

# transform the normed array back into the original array for more intuitive plotting.
def deNormalizeArray(minX, maxX, X):
    import numpy as np
    denormed = np.array( [ x*(maxX-minX)+minX for x in X ] )
    return denormed

# build the risk profile for a given user, based on the method used to fit the value function.
'''
if u(x) is an at least twice-differentiable utilty (value) function 
for the decision maker, then the local risk profile r(x) is:
    r(x) = - u''(x) / u'(x)

...Where r(x) > 0 indicates risk aversion at x and r(x) < 0 indicates risk proneness.

See: Clemen, R. T. (1996). Making Hard Decisions: An Introduction to
Decision Analysis. Belmont, CA, Wadsworth Publishing Company.
'''
def rx(x,h,k,a,b,c,rType):
    import numpy as np
    if rType == 1: # the local risk profile for u(x) modeled by utility_form1
        return (a*a*np.exp(-a*x) + b*c*c*np.exp(-c*x)) / (a*np.exp(-a*x) + b*c*np.exp(-c*x))
    elif rType == 2: # the local risk profile for u(x) modeled by utility_form2
        return (k*a*a*np.exp(-a*x) + b*c*c*np.exp(-c*x)) / (k*a*np.exp(-a*x) + b*c*np.exp(-c*x))
    else:
        print('check r(x) type')
        return x

# define a value (utility) functional form modeling decreasing risk aversion.
'''
One good function for this that fits most input data is given by:
    u(x) = h - k( exp(-a*x) + b*exp(-c*x) ),
... Where h, k, a, b, and c are parameters to be fit to the observed data.
'''
def utility_form1(x,h,k,a,b,c):
    import numpy as np
    ux = h-k*(np.exp(-a*x)+b*np.exp(-c*x))
    return ux

# define a value (utility) functional form modeling decreasing risk proneness.
'''
One good function for this that fits most (risk-prone) input data is given by:
    u(x) = k*exp(-a*x) + b*exp(-c*x) ),
... Where k, a, b, and c are parameters to be fit to the observed data.
'''
def utility_form2(x,h,k,a,b,c):
    import numpy as np
    ux = k*np.exp(-a*x)+b*np.exp(-c*x)
    return ux

# define a function to fit the utility functional form to a given, normalized X,Y data pair
'''
Here we use the curve fit utility from scipy.
func <-- the utility function form used to fit the data for the given decision maker (1 or 2).
'''
def fitUtilityFunction(X,Y,func):
    from scipy.optimize import curve_fit
    curve_opt, curve_cov = curve_fit(func,X,Y,
                                     maxfev=20000) # fit the observed data, try 20,000 iterations
    # print(curve_opt)
    return curve_opt

# define a function to plot the utility (Value) function fitting responses, and the risk profiles.
def plotNiceGraphs(teamData, minGrade = 40., maxGrade = 100.):

    import numpy as np
    import matplotlib.pyplot as plt

    # define the plot for the utility function
    plt.figure(figsize=(12,7))
    initplt = True
    teamData['rx'] = {}
    for name in teamData.keys():
    #     print name
        if name != 'Y' and name != 'rx':
            
            Y = teamData['Y']
            X = teamData[name]
            minX = min(X)
            minY = min(Y)
            maxX = max(X)
            maxY = max(Y)
            # check if the current team member is risk prone or risk averse...
            
            # get the team member's data and normalize it...
            xNormed = normalizeArray(teamData[name])
            yNormed = normalizeArray(teamData['Y'])
            if maxGrade < minGrade:
                xNormed[2] = np.array( [max(xNormed[2])+min(xNormed[2])-x for x in xNormed[2]] )
                # print(xNormed[2])
                # yNormed[2] = np.array( [max(yNormed[2])+min(yNormed[2])-x for x in yNormed[2]] )
            # fit the utility function...
    #         print xNormed[2][1:-1], yNormed[2]
            
            # try the first functional form
            try:
                # print('getting the value function for', name,'...')
                func = utility_form1
                rType = 1
                params = fitUtilityFunction(xNormed[2], yNormed[2], func)
            except RuntimeError:
                try:
                    # print('...trying risk prone value function instead...')
                    func = utility_form2
                    rType = 2
                    params = fitUtilityFunction(xNormed[2], yNormed[2], func)
                except RuntimeError:
                    print('...still getting an error. Ask for help!')
            y_2normed = func(np.linspace(0,1,100),
                             params[0],
                             params[1],
                             params[2],
                             params[3],
                             params[4]
                            )
            r = rx(np.linspace(0,1,100),
                   params[0],
                   params[1],
                   params[2],
                   params[3],
                   params[4],
                   rType
                  )
            teamData['rx'][name] = r
            y2 = deNormalizeArray(yNormed[0],yNormed[1],y_2normed)
            x2 = deNormalizeArray(xNormed[0],xNormed[1],np.linspace(0,1,100))
            if maxGrade < minGrade:
                x2 = np.array( [maxGrade+minGrade-x for x in x2] )
            if initplt == True:
                if maxGrade < minGrade:
                    plt.plot([maxX,minX],[minY,maxY],linestyle='-',c='k',linewidth=4,label='Risk-Neutral')
                else:
                    plt.plot([minX,maxX],[minY,maxY],linestyle='-',c='k',linewidth=4,label='Risk-Neutral')
                initplt = False
    #         plt.plot(x2,y2,linestyle="--",linewidth=3,label=name)
            p = plt.plot(x2,y2,linestyle="--",linewidth=3,label=name)
            co = p[0].get_color()
            plt.scatter(teamData[name],teamData['Y'],c=co, s=100, marker='X')
    plt.legend(loc='best')
    plt.grid()
    plt.ylabel('u(x): Certainty Equivalent Probability')
    plt.xlabel('x: Grade')
    plt.xlim([minGrade,maxGrade])
    plt.ylim([0,1])
    plt.title('Value Functions for Each Team Member')
    plt.show()

    # define a function to plot the local risk profiles for each team member
    plt.figure(figsize=(12,7))
    for name in teamData.keys():
        if name != 'Y' and name != 'rx':
            plt.plot(np.linspace(minGrade,maxGrade,len(teamData['rx'][name])),
                     np.zeros(len(teamData['rx'][name])),
             linestyle=":",linewidth=5,c='k')
            plt.plot(np.linspace(minGrade,maxGrade,len(teamData['rx'][name])),
                     teamData['rx'][name],
                     linestyle="--",linewidth=3,label=name)
            
    plt.legend(loc='best')
    plt.grid()
    plt.ylabel('r(x): Risk Aversion')
    plt.xlabel('x: Grade')
    plt.xlim([minGrade,maxGrade])
    plt.title('Local Risk Profiles for Each Team Member')
    plt.show()

    return