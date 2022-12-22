def _edit_opr_init(len1, len2):
    
    lev = []
    for i in range(len1):
        lev.append([0,""] * len2)           # initialize 2D array to zero
    
    for i in range(len1):
        lev[i][0] = [i, ""]           # column 0: 0,1,2,3,4,...

    for j in range(len2):
        lev[0][j] = [j, ""]           # row 0: 0,1,2,3,4,...

    return lev

def _edit_opr_step(lev, i, j, query, candidate):
    c1 = query[i - 1]
    c2 = candidate[j - 1]

    opr = list([])
#   Deletion
    # skipping a character in Query
    a = lev[i - 1][j][0] + 1
    
    opr.append([a,lev[i - 1][j][1]+"_D:"+str(c1)+str(c2)])

#    Insertion
    # skipping a character in Candidate
    b = lev[i][j - 1][0] + 1
    opr.append([b,lev[i][j - 1][1]+"_I:"+str(c1)+str(c2)])
    
    # substitution
    c = lev[i - 1][j - 1][0] + (c1 != c2)
    opr.append([c,lev[i - 1][j - 1][1]+"_S:"+str(c1)+str(c2)])

    d = c+1
    # transposition
    if i > 1 and j > 1:
        if query[i - 2] == c2 and candidate[j - 2] == c1:
            d = lev[i - 2][j - 2][0] + 1
            opr.append([d,lev[i][j - 1][1]+"_T:"+str(c1)+str(c2)])


    # pick the cheapest
    min_dist = 9999
    min_val = ""
#    print(opr)
    for x in opr:
        if x[0] < min_dist:
#            print(i)

            min_dist = x[0]
            min_val = x[1]
            
    lev[i][j] = [min_dist,min_val]
#    lev[i][j][0] = min_dist
#    lev[i][j][1] = min_val
        

def edit_opr(s1, s2):
    # set up a 2-D array
    s1 = "["+s1+"["
    s2 = "["+s2+"["
    len1 = len(s1)
    len2 = len(s2)
    lev = _edit_opr_init(len1 + 1, len2 + 1)

    # iterate over the array
    for i in range(len1):
        for j in range(len2):
            _edit_opr_step(lev, i + 1, j + 1, s1,s2)
    
#    print(lev[len1][len2][0])
    return lev[len1][len2][1]