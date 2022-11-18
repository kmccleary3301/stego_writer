import new_stego_v1 as stego
import sys, cv2
import ECC
import math
from math import log2, log, exp



test_message = "lipsum lipsum lipsum lipsum lipsum lipsum lipsum"

ecc_keys = ECC.getPubPrivKeys();

print("pubkey:", ecc_keys[0])
print("privkey:", ecc_keys[1])

n = 5975750469202303846015008333786227023583197065596626711065162427033334890932
classical_time = exp(1.9*(log(n)**(1/3))*((log(log(n)))**(2/3)))
q_time = (log(n)**2)*(log(log(n)))*(log(log(log(n))))

print("log2 of int :", log2(n))
print("classical time :", classical_time)
print("q time:", q_time)