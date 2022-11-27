import codecs
from Crypto.Cipher import AES
from Crypto import Random
import base64
import hashlib
import stego
import ECC
import random
import cv2


ECCKeys = ECC.getPubPrivKeys()
pub_key = ECC.makePubKeyText(ECCKeys[1])
priv_key = hex(ECCKeys[0])





password="hello"

message = "super secret message"

encrypt = ECC.encrypt_AES_plain(message, password, return_bits=True)
print("message:",message)


#encrypt_ecc = ECC.encrypt_ECC_plain(message, ECCKeys[1])
#encrypt_ecc = ECC.encrypt_Plain_old(message, ECCKeys[1])
#print("encrypt_ecc:",encrypt_ecc)
#
#print("encrypted_text:", encrypt)

#decrypt_ecc = ECC.decrypt_ECC_plain(encrypt_ecc, ECCKeys[0])
#decrypt_ecc = ECC.decrypt_Plain_old(encrypt_ecc, ECCKeys[0])
#print("decrypt_ecc:",decrypt_ecc)

#e_1 = ECC.encrypt_ECC_plain(message, ECCKeys[1])

#print("e_1:",e_1)
#d_1 = ECC.decrypt_ECC_plain(e_1, ECCKeys[0])


#message_recover = ECC.decrypt_AES_plain(encrypt, password)
#print("message recover:", message_recover)

img = cv2.imread("C:/Users/subje/Downloads/dhop.jpg")

message_write = stego.image_write_new(img, message, encryption='aes', key=password)

message_recover = stego.image_read_new(message_write, encryption='unencrypted', key=password)
print("recovered message:", message_recover)



