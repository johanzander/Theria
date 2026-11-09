from ml import *
days=daylist("dev");act=actual(days);w=hw(days,7)
B={"lag":1,"flow":1,"sun":1,"wind":1}
res={}
def go(k,c,ev=7):
    res[k]=run(c,CH,days,every=ev);show(k,score(add(res[k],w),act,days))
for hl in (60,120,180,365,730,1e6):go(f"linear hl={hl}",{**B,"hl":hl})
for k,s in ((3,1),(3,.5),(7,1),(7,.5),(14,1),(14,.5),(28,1),(28,.5)):go(f"linear hl365 lev{k} s{s}",{**B,"hl":365,"lev":k,"shr":s})
for lv,it in ((15,150),(15,300),(7,300),(31,200)):go(f"gb leaves{lv} it{it} hl365",{"gb":1,"hl":365,"leaves":lv,"iters":it},14)
go("gb hl730",{"gb":1,"hl":730,"leaves":15,"iters":200},14)
go("gb hl365 lev7",{"gb":1,"hl":365,"leaves":15,"iters":200,"lev":7,"shr":.5},14)
pickle.dump(res,open(d+"/c3.pkl","wb"))
a=res["linear hl=365"];g=res["gb leaves15 it150 hl365"]
show("mean(linear,gb)",score(add({o:(a[o]+g[o])/2 for o in a},w),act,days))
