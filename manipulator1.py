import numpy as np
from math import cos, sin
import glb_

def manipulator1(q, qdot, tau):

    theta=q
    thetadot=qdot

    lg1=glb_.l1*0.5
    lg2=glb_.l2*0.5

    M11=glb_.m1*lg1**2+glb_.I1+glb_.m2*(glb_.l1**2+lg2**2 +2*glb_.l1*lg2*cos(theta[1]))+glb_.I2
    M12=glb_.m2*(lg2**2+glb_.l1*lg2*cos(theta[1]))+glb_.I2
    M22=glb_.m2*lg2**2+glb_.I2
    g1 =glb_.m1*glb_.ghat*lg1*cos(theta[1])+glb_.m2*glb_.ghat*(glb_.l1*cos(theta[1])+lg2*cos(theta[0]+theta[1]))
    g2 =glb_.m2*glb_.ghat*lg2*cos(theta[0]+theta[1])

    h122=-glb_.m2*glb_.l1*lg2*sin(theta[1])
    h112=h122
    h211=-h122

    h1 = h122*thetadot[1]**2+2*h112*thetadot[0]*thetadot[1]
    h2 = h211*thetadot[0]**2

    M=np.array(
        [
            [M11, M12],
            [M12, M22]
        ]
    )

    g=np.array(
        [
            [g1],
            [g2]
        ]
    )

    h=np.array(
        [
            [h1],
            [h2]
        ]
    )

    qddot= np.linalg.inv(M)*(tau-h-g)
    print(qddot)

    newq=0
    newqdot=0
    return newq, newqdot

if __name__=="__main__":
    tau=np.array(
        [
            [1], [1]
        ]
    )
    print(manipulator1([1, 2],[0, 0],tau))