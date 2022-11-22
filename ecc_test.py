import ECC

def test_1():
    key_set = ECC.getPubPrivKeys()
    pub_key = key_set[1]
    priv_key = key_set[0]
    msg = "Test message 123456789"
    en_msg = ECC.encrypt_Plain(msg, pub_key)

    print(pub_key)
    print(priv_key)
    print(en_msg)

test_1()
