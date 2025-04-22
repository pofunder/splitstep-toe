def schwarzschild_metric(r, rs):
    gtt = -(1-rs/r)
    grr = 1/(1-rs/r)
    return [[gtt,0,0,0],[0,grr,0,0],[0,0,r*r,0],[0,0,0,r*r]]
