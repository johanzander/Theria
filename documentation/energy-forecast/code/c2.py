from ml import *
import itertools
days=daylist("dev");act=actual(days);w=hw(days,7)
base={"hl":365}
C={"hinge":{},"hinge+lag":{"lag":1},"hinge+rts":{"rts":1},"hinge+flow":{"flow":1},"hinge+rts+flow":{"rts":1,"flow":1},
"+lag+rts":{"lag":1,"rts":1},"+lag+flow":{"lag":1,"flow":1},"+lag+flow+sun+wind":{"lag":1,"flow":1,"sun":1,"wind":1},"+lag+rts+sun+wind":{"lag":1,"rts":1,"sun":1,"wind":1},
"+lag+rts+sun+wind+hod":{"lag":1,"rts":1,"sun":1,"wind":1,"hod":1},"+lag+rts+sun+wind+hod+hodh":{"lag":1,"rts":1,"sun":1,"wind":1,"hod":1,"hodh":1}}
for k,v in C.items():show(k,score(add(run({**base,**v},CH,days),w),act,days))
