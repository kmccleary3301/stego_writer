from tinyec import registry, ec
from Crypto.Cipher import AES
import hashlib, secrets, binascii

def encrypt_AES_GCM(msg, secretKey):
    aesCipher = AES.new(secretKey, AES.MODE_GCM)
    ciphertext, authTag = aesCipher.encrypt_and_digest(msg)
    return (ciphertext, aesCipher.nonce, authTag)

def decrypt_AES_GCM(ciphertext, nonce, authTag, secretKey):
    aesCipher = AES.new(secretKey, AES.MODE_GCM, nonce)
    plaintext = aesCipher.decrypt_and_verify(ciphertext, authTag)
    return plaintext

def ecc_point_to_256_bit_key(point):
    sha = hashlib.sha256(int.to_bytes(point.x, 32, 'big'))
    sha.update(int.to_bytes(point.y, 32, 'big'))
    return sha.digest()

curve = registry.get_curve('brainpoolP256r1')

def encrypt_ECC(msg, pubKey):
    ciphertextPrivKey = secrets.randbelow(curve.field.n)
    sharedECCKey = ciphertextPrivKey * pubKey
    secretKey = ecc_point_to_256_bit_key(sharedECCKey)
    ciphertext, nonce, authTag = encrypt_AES_GCM(msg, secretKey)
    ciphertextPubKey = ciphertextPrivKey * curve.g
    return (ciphertext, nonce, authTag, ciphertextPubKey)

def decrypt_ECC(encryptedMsg, privKey):
    (ciphertext, nonce, authTag, ciphertextPubKey) = encryptedMsg
    sharedECCKey = privKey * ciphertextPubKey
    secretKey = ecc_point_to_256_bit_key(sharedECCKey)
    plaintext = decrypt_AES_GCM(ciphertext, nonce, authTag, secretKey)
    return plaintext

def encrypt_Plain(msg, pubKey):
    return encrypt_ECC(msg.encode('utf-8'), pubKey)

def decrypt_Plain(encryptMsg, privKey):
    return decrypt_ECC(encryptMsg, privKey).decode('utf-8')

def getPubPrivKeys():
    privKey = secrets.randbelow(curve.field.n)
    pubKey = privKey * curve.g
    return [privKey, pubKey]

def getMsgObj(encryptedMsg):
    encryptedMsgObj = {
        'ciphertext': binascii.hexlify(encryptedMsg[0]),
        'nonce': binascii.hexlify(encryptedMsg[1]),
        'authTag': binascii.hexlify(encryptedMsg[2]),
        'ciphertextPubKey': hex(encryptedMsg[3].x) + hex(encryptedMsg[3].y % 2)[2:]
    }
    return encryptedMsgObj

def enMsg2Hex(msgObj):
    strGet = ''
    uGet = [binascii.hexlify(i).decode('utf-8') for i in msgObj[:3]] + [hex(msgObj[3].x), hex(msgObj[3].y)]
    for i in uGet:
        strGet += i+'\n'
    return strGet

def hex2EnMsg(hexInput):
    uGet = hexInput.split('\n')
    enMsgReturn = (binascii.unhexlify(uGet[0].encode('utf-8')), binascii.unhexlify(uGet[1].encode('utf-8')),
             binascii.unhexlify(uGet[2].encode('utf-8')), ec.Point(curve, int(uGet[3], 16), int(uGet[4], 16)))
    return enMsgReturn

def makePubKeyText(pubKey):
    keyObj = [hex(pubKey.x), hex(pubKey.y)]
    return keyObj[0]+'\n'+keyObj[1]

def pubKeyFromText(textGet):
    ptGet = [int(i, 16) for i in textGet.split('\n')]
    return ec.Point(curve, ptGet[0], ptGet[1])