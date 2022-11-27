from tinyec import registry, ec
from Crypto.Cipher import AES
import hashlib, secrets, binascii
import copy

def encrypt_AES_GCM(msg, secretKey):
    aesCipher = AES.new(secretKey, AES.MODE_GCM)
    ciphertext, authTag = aesCipher.encrypt_and_digest(msg)
    return (ciphertext, aesCipher.nonce, authTag)

def decrypt_AES_GCM(ciphertext, nonce, authTag, secretKey):
    aesCipher = AES.new(secretKey, AES.MODE_GCM, nonce)
    plaintext = aesCipher.decrypt_and_verify(ciphertext, authTag)
    return plaintext

def encrypt_AES_GCM_2(msg, key, key_is_hashed=None, message_encoding=None, return_bits=None):
    if key_is_hashed is None:
        key_is_hashed = False
    if message_encoding is None:
        message_encoding = 'utf-8'
    if return_bits is None:
        return_bits = False
    if not key_is_hashed:
        key = hashlib.sha256(key.encode()).digest()
    aesCipher = AES.new(key, AES.MODE_GCM)
    ciphertext, authTag = aesCipher.encrypt_and_digest(msg.encode(message_encoding))
    ciphertext = ciphertext.hex()
    authTag = authTag.hex()
    nonce = aesCipher.nonce.hex()
    if return_bits:
        ciphertext, nonce, authTag = bin(int(ciphertext, 16))[2:], bin(int(nonce, 16))[2:], bin(int(authTag, 16))[2:]
    return (ciphertext, nonce, authTag)

def decrypt_AES_GCM_2(ciphertext, nonce, key, key_is_hashed=None, tag=None, message_encoding=None):
    if key_is_hashed is None:
        key_is_hashed = False
    if message_encoding is None:
        message_encoding = 'utf-8'
    if not key_is_hashed:
        key = hashlib.sha256(key.encode()).digest()
    aesCipher = AES.new(key, AES.MODE_GCM, nonce=bytes.fromhex(nonce))
    print("cipher_text:", ciphertext)
    plaintext = aesCipher.decrypt(bytes.fromhex(ciphertext))
    plaintext = plaintext.decode(message_encoding)
    tag = bytes.fromhex(tag)
    #plaintext = aesCipher.decrypt_and_verify(ciphertext, tag)
    if not tag is None:
        try:
            aesCipher.verify(tag)
            print("The message is authentic:", plaintext)
        except ValueError:
            print("Key incorrect or message corrupted")
    return plaintext

def encrypt_AES_plain(msg, key, return_bits=None):
    if return_bits is None:
        return_bits = False
    (ciphertext, nonce, auth_tag) = encrypt_AES_GCM_2(msg, key)
    print("nonce:", nonce)
    print("auth_tag:", auth_tag)
    print("ciphertext:", ciphertext)
    return_string = ''.join([nonce, auth_tag, ciphertext])
    if return_bits:
        return_string = bin(int(return_string, 16))[2:]
    return return_string
    
def decrypt_AES_plain(en_msg, key):
    try:
        en_msg = hex(int(en_msg, 2))[2:]
    except ValueError:
        pass
    nonce = en_msg[:32]
    auth_tag = en_msg[32:64]
    ciphertext = en_msg[64:]
    print("nonce:", nonce)
    print("auth_tag:", auth_tag)
    print("ciphertext:", ciphertext)
    return decrypt_AES_GCM_2(ciphertext, nonce, key, tag=auth_tag)

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

def encrypt_ECC_plain(msg, pubKey, return_bits=None):
    msg, pubKey = copy.copy(msg), copy.copy(pubKey)
    if return_bits is None:
        return_bits = False
    (ciphertext, nonce, authTag, ciphertextPubKey) = encrypt_ECC(msg.encode('utf-8'), pubKey)
    return_array = [nonce.hex(), authTag.hex(), makePubKeyText(ciphertextPubKey), ciphertext.hex()]
    return_string = ''.join(return_array)
    if return_bits:
        return_string = bin(int(return_string, 16))[2:]
    return return_string

def decrypt_ECC_plain(en_msg, privKey):
    try:
        en_msg = hex(int(en_msg, 2))[2:]
    except ValueError:
        pass
    en_msg, privKey = copy.copy(en_msg), copy.copy(privKey)
    (nonce, authTag, ciphertextPubKey, ciphertext) = (en_msg[0:32], en_msg[32:64], en_msg[64:192], en_msg[192:])
    ciphertextPubKey = pubKeyFromText(ciphertextPubKey)
    pass_data = (bytes.fromhex(ciphertext), bytes.fromhex(nonce), bytes.fromhex(authTag), ciphertextPubKey)
    return decrypt_ECC(pass_data, privKey).decode('utf-8')

def encrypt_Plain_old(msg, pubKey):
    return encrypt_ECC(msg.encode('utf-8'), pubKey)

def decrypt_Plain_old(encryptMsg, privKey):
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
    keyObj = [hex(pubKey.x)[2:].zfill(64), hex(pubKey.y)[2:].zfill(64)]
    return keyObj[0]+keyObj[1]

def pubKeyFromText(textGet):
    ptGet = [int(textGet[:64], 16), int(textGet[64:], 16)]
    return ec.Point(curve, ptGet[0], ptGet[1])