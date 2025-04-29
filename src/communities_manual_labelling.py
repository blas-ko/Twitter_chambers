def communities_manual_labelling(persistent_leading_users, M=50):
    '''Returns the empirical partition of climate believers, skeptics and news media for either the top M=30 or M=50 persistent leading users. 
    '''
    
    if M == 30: # TODO: Remove persistent users depedency
        skeptics = [ 'PrisonPlanet', 'DineshDSouza', 'RealSaavedra', 'manny_ottawa', 'chuckwoolery', 'JWSpry' ]
        other = ['LilNasX', 'CNN', 'nytimes', 'ajplus', 'nowthisnews']
        believers = list( set( persistent_leading_users ).difference( set(other) ).difference(set(skeptics)) )
    elif M == 50:
        skeptics = [ 'PrisonPlanet', 'DineshDSouza', 'RealSaavedra', 'manny_ottawa', 'chuckwoolery', 'JWSpry',
            'mitchellvii', 'charliekirk11', 'catturd2', 'prageru']
        other = ['LilNasX', 'CNN', 'nytimes', 'ajplus', 'nowthisnews', 'tictoc', 'MSNBC', 'TIME', 'BhadDhad']
        believers = ['IlhanMN', 'sunrisemvmt', 'ewarren', 'sunfloweraidil', 'BernieSanders',
            'AOC', 'UNFCCC', 'PaulEDawson', '314action', 'KamalaHarris',
            'SenSanders', 'jeremycorbyn', 'GeraldKutney', 'JimMFelton',
            'RepAdamSchiff', 'GretaThunberg', 'cathmckenna', 'JayInslee',
            'billmckibben', 'JoeBiden', 'BillGates', 'ProfStrachan', 'MichaelEMann',
            'BetoORourke', 'mmpadellan', 'brianschatz', 'MikeHudema',
            'ProudResister', 'JustinTrudeau', 'narendramodi', 'SenSchumer']
    else:
        Exception('Empirical partition only available for M=30 and M=50.')

    # Creating partition
    _p = {believer:'believers' for believer in believers}
    _a = {skeptic:'skeptics' for skeptic in skeptics}
    _o = {oth:'other' for oth in other}
    P_empirical = {**_p, **_o, **_a}
    return P_empirical

def communities_manual_labelling_anonymized(M=50):
    '''Returns the empirical partition of climate believers, skeptics and news media for either the top M=30 or M=50 persistent leading users. 
    '''
    if M == 30:
        pass # TODO:        
    elif M == 50:
        skeptics = [3831362, 3137558, 1050757, 1981792, 7357447, 1767967, 6430374,  124607, 243821, 1831181]
        other = [4818939, 4910822, 4566779, 5168196, 61061, 3466622, 623626, 2634401, 6443584]
        believers = [6378566, 6234729, 6720666, 7170514, 3155210, 1611369, 6026025, 4323687,
            5625821, 6428323, 6314467, 4059245, 7106479, 4645949, 1858711, 5784291,
            3233983, 7001891, 7184782, 3810918, 2338903, 4195795, 6357256, 6448658,
            2085114,  395868, 5282130, 4239058, 4907296, 1169973, 6309179]
    else:
        Exception('Empirical partition only available for M=30 and M=50.')

    # Creating partition
    _p = {believer:'believers' for believer in believers}
    _a = {skeptic:'skeptics' for skeptic in skeptics}
    _o = {oth:'other' for oth in other}
    P_empirical = {**_p, **_o, **_a}
    return P_empirical