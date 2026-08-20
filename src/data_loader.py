import requests

BASE = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda"

# 1. Le schéma : liste des champs et leurs types
champs = requests.get(f"{BASE}").json()["fields"]
for c in champs:
    print(f"{c['name']:35} {c['type']:12} {c.get('label','')}")

# 2. Un enregistrement réel, pour voir ce qu'il y a vraiment dedans
rec = requests.get(f"{BASE}/records", params={"limit": 1}).json()
print(rec["results"][0])

# 3. Le volume disponible sur le Grand Est
vol = requests.get(f"{BASE}/records", params={
    "where": 'location_region="Grand Est"', "limit": 1
}).json()
print("Total Grand Est :", vol["total_count"])