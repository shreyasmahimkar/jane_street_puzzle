import urllib.request
import json
import ssl

hash_str = "c7ef65233c40aa32c2b9ace37595fa7c"
url = f"https://md5decrypt.net/Api/api.php?hash={hash_str}&hash_type=md5&email=test@example.com&code=1152464b80a61728"
# Not sure if md5decrypt api requires proper email/code now.

# Let's try nitrxgen
try:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(f"http://www.nitrxgen.net/md5db/{hash_str}")
    res = urllib.request.urlopen(req, context=ctx).read().decode('utf-8')
    if res:
        print(f"Nitrxgen found: {res}")
    else:
        print("Nitrxgen: Not found")
except Exception as e:
    print(f"Nitrxgen error: {e}")

