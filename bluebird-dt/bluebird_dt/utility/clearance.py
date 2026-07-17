from __future__ import annotations

import re
from typing import Literal

from bluebird_dt.core import (
    Action,
    Aircraft,
    ClearanceAndResponse,
    Environment,
    FlightPlan,
    HoldParameters,
    Route,
)
from bluebird_dt.logger import logger

phonetic = {
    "A": "alpha",
    "B": "bravo",
    "C": "charlie",
    "D": "delta",
    "E": "echo",
    "F": "foxtrot",
    "G": "golf",
    "H": "hotel",
    "I": "india",
    "J": "juliett",
    "K": "kilo",
    "L": "lima",
    "M": "mike",
    "N": "november",
    "O": "oscar",
    "P": "papa",
    "Q": "quebec",
    "R": "romeo",
    "S": "sierra",
    "T": "tango",
    "U": "uniform",
    "V": "victor",
    "W": "whiskey",
    "X": "x-ray",
    "Y": "yankee",
    "Z": "zulu",
    "0": "zero",
    "1": "wun",
    "2": "too",
    "3": "tree",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "niner",
    ".": "decimal",
}

# A few IATA airport codes: some apparent modifications, perhaps for phonetic clearances E.g. EGPH - Endinbrugh.
# Consider importing from external resource.
WAYPOINTS: dict[str, str] = {
    "ADN": "Aberdeen",
    "BKY": "Barkway",
    "BEL": "Belfast",
    "BEN": "Benbecular",
    "BHD": "Berryhead",
    "BIG": "Biggin",
    "BOV": "Bovingdon",
    "BCN": "Brecon",
    "BPK": "Brookmans Park",
    "CLN": "Clackton",
    "CPT": "Compton",
    "DTY": "Daventry",
    "DCS": "Deanscross",
    "DET": "Detling",
    "DVR": "Dover",
    "DUD": "Dundonald",
    "GAM": "Gamston",
    "GOW": "Glasgow",
    "GWC": "Goodwood",
    "DUF": "Great Dun Fell",
    "GLO": "Green Lowther",
    "HEN": "Henton",
    "HON": "Honily",
    "IOM": "Isle of Man",
    "JSY": "Jersey",
    "LAM": "Lamborne",
    "LND": "Lands end",
    "LON": "London",
    "LYD": "Lydd",
    "MAC": "MACHRIHANISH",
    "MCT": "MANCHESTER",
    "MAY": "Mayfield",
    "MID": "Midhurst",
    "NEW": "Newcastle",
    "OCK": "Ockham",
    "OTR": "Ottringham",
    "PTH": "Perth",
    "POL": "Pole hill",
    "SAB": "Saint abbs",
    "SFD": "Seaford",
    "SAM": "Southampton",
    "STN": "Stornoway",
    "STU": "Strumble",
    "SUM": "Sumburgh",
    "TLA": "Talla",
    "TIR": "Tiree",
    "TNT": "Trent",
    "TRN": "Turnberry",
    "WAL": "Wallasey",
    "WIK": "Wick",
    "WOD": "Woodley",
    "LL": "Heathrow",
    "EGLL": "Heathrow",
    "KK": "Gatwick",
    "EGKK": "Gatwick",
    "SS": "Stan-sted",
    "EGSS": "Stan-sted",
    "EGLC": "London City",
    "EGGW": "LUTON",
    "GW": "LUTON",
    "EHAM": "Amsterdam",
    "KJFK": "New York",
    "KALT": "Atlanta",
    "EIDW": "Dublin",
    "EGCC": "MANCHESTER",
    "EGBB": "BIRMINGHAM",
    "EGAA": "BELFAST",
    "EGPF": "Glasgow",
    "EGPD": "Aberdeen",
    "EGPH": "Endinbrugh",
    "EGNX": "East midlands",
    "EGNM": "Leeds Bradford",
    "EGGD": "Bristol",
}

operators = {
    "AAB": ["ABG", "a b g"],
    "AAC": ["Army air", "armyair", "ENGLISH"],
    "AAD": ["Ambassador", "ambassador"],
    "AAG": ["Atlantic", "Atlantic"],
    "AAL": ["American", "American", "AMERICAN"],
    "AAR": ["Asiana", "asiana"],
    "AAW": ["Afriqiyah", "afrik iyaa"],
    "ABD": ["Atlanta", "atlanta"],
    "ABQ": ["Pakblue", "pakblue"],
    "ABR": ["Contract", "contract"],
    "ABX": ["Abex", "a becks"],
    "ACA": ["Air canada", "aircanada", "AMERICAN"],
    "ACX": ["Loadmaster", "loadmaster"],
    "ADB": ["Antonov bureau", "Antonov bureau"],
    "ADH": ["Heron", "Heron"],
    "AEA": ["Europa", "Europa"],
    "AEE": ["Aegean", "a jee an"],
    "AEL": ["Air europe", "aireurope"],
    "AEU": ["Flystar", "flystar"],
    "AFL": ["Aeroflot", "aeroflot"],
    "AFR": ["Air france", "airfrance"],
    "AHO": ["Air Hamburg", "air hamburg"],
    "AIA": ["Avies", "a vees"],
    "AIC": ["Air india", "airindia"],
    "AIN": ["Fly Cargo", "fly cargo"],
    "AJK": ["Bambi", "Bambi"],
    "AJM": ["Jamaica", "Jamaica"],
    "ALK": ["Srilankan", "srilankan"],
    "AMB": ["Civil air ambulance", "civil air ambulance"],
    "AMC": ["Air malta", "airmalta"],
    "AMT": ["Am Tran", "amtran", "AMERICAN"],
    "AMX": ["Aeromexico", "aero mexico"],
    "ANA": ["All nippon", "all nippon"],
    "ANE": ["Nostru Air", "nostru air"],
    "ANZ": ["New Zealand", "NewZealand"],
    "ARA": ["Arik Air", "arik air"],
    "ARG": ["Argentina", "argentina"],
    "ATJ": ["Snoopy", "Snoopy"],
    "ATX": ["Airtax", "air tax"],
    "AUA": ["Austrian", "Austrian"],
    "AUR": ["Ayline", "ayline"],
    "AVB": ["Beaupair", "beaupair"],
    "AWC": ["Zap", "zap"],
    "AWE": ["Cactus", "cactus", "AMERICAN"],
    "AXN": ["Alexandros", "Alexandros"],
    "AYY": ["Lupus", "LooPuss"],
    "AZA": ["Alitalia", "Alitalia"],
    "AZE": ["Arcus Air", "Arcusair"],
    "AZW": ["Air Zimbabwe", "Airzimbabwe"],
    "BAE": ["Felix", "felix"],
    "BAW": ["Speed bird", "speedbird", "ENGLISH"],
    "BBC": ["Bangladesh", "Bangladesh"],
    "BBD": ["Blue Cargo", "blue cargo"],
    "BBO": ["Baboo", "baboo"],
    "BCI": ["Blue Island", "Blue island"],
    "BCS": ["Eurotrans", "Eurotrans"],
    "BCY": ["City Ireland", "cityireland"],
    "BDN": ["Gauntlet", "Gauntlet"],
    "BEE": ["Jersey", "Jersy", "ENGLISH"],
    "BEL": ["Beeline", "bee line"],
    "BER": ["Air berlin", "airBerlin"],
    "BGH": ["Balkan holidays", "Balkanholidays"],
    "BID": ["Binair", "bin air"],
    "BLE": ["Blue Berry", "blue berry"],
    "BLF": ["Bluefin", "blue fin"],
    "BLX": ["Bluescan", "bluescan"],
    "BMI": ["Baby", "baby"],
    "BMM": ["Atlas Blue", "atlas blue"],
    "BMR": ["Midland", "midland"],
    "BNW": ["British north", "britisnorth", "AMERICAN"],
    "BOE": ["Boeing", "boeing"],
    "BOO": ["Bookajet", "book a jet"],
    "BOS": ["Mistral", "mistral"],
    "BOX": ["German Cargo", "german cargo"],
    "BPA": ["Blue panorama", "blue panorama"],
    "BRE": ["Aviabreeze", "aviabreeze"],
    "BRO": ["Broadsword", "broadsword"],
    "BRU": ["Belarus Air", "belarus air"],
    "BTI": ["Air Baltic", "air baltic"],
    "BTU": ["Rolls", "rolls"],
    "BWA": ["Caribbean Airlines", "caribbean airlines"],
    "BWY": ["Broadway", "broadway", "ENGLISH"],
    "BZH": ["Brit air", "Britair", "ENGLISH"],
    "CAT": ["Aircat", "air cat"],
    "CCA": ["Air china", "airchina"],
    "CEG": ["Cega", "cega"],
    "CES": ["China Eastern", "china eastern"],
    "CFC": ["Can force", "canforce", "AMERICAN"],
    "CFD": ["Aeronaut", "aeronaut"],
    "CFE": ["Flyer", "flyer", "BRITISH"],
    "CFG": ["Condor", "condor"],
    "CFU": ["Min air", "minair", "ENGLISH"],
    "CIM": ["Cimber", "Kimber"],
    "CJA": ["Canjet", "canjet"],
    "CKS": ["Connie", "Connie"],
    "CLB": ["Calibrator", "Calibrator"],
    "CLF": ["Clifton", "Clifton"],
    "CLG": ["Chalair", "chalair"],
    "CLW": ["Central wings", "centralwings"],
    "CLX": ["Cargolux", "cargolux"],
    "CMB": ["Camber", "camber"],
    "CNO": ["Scanor", "scanor"],
    "COA": ["Continental", "continental", "AMERICAN"],
    "CPA": ["Cathay", "cathay"],
    "CPH": ["Champagne", "champagne"],
    "CRK": ["Bauhinia", "bow hinia"],
    "CRL": ["Corsair", "course air"],
    "CRX": ["Crossair", "crossair"],
    "CSA": ["CSA", "CSA"],
    "CSN": ["China Southern", "china southern"],
    "CTM": ["Cotam", "cotam"],
    "CTN": ["Croatia", "Croatia"],
    "CUB": ["Cub ana", "cubana"],
    "CWC": ["Challenge Cargo", "challenge cargo"],
    "CWL": ["Cranwell", "cranwell"],
    "CWY": ["Causeway", "cause way"],
    "CYP": ["Cyprus", "Cyprus"],
    "DAF": ["Danish air force", "danish airforce"],
    "DAH": ["Air algerie", "airalgerie"],
    "DAL": ["Delta", "delta", "AMERICAN"],
    "DCS": ["Twin Star", "twin star"],
    "DHK": ["World Express", "world express"],
    "DHL": ["DHL", "DHL"],
    "DHX": ["Dilmun", "dilmun"],
    "DLH": ["Lufthansa", "lufthansa"],
    "DMO": ["Domodeovo", "Domodeovo"],
    "DNM": ["Denim", "denim"],
    "DRD": ["Alada air", "alada air"],
    "DRT": ["Darta", "darta"],
    "DSR": ["Dairair", "dairair"],
    "DTA": ["Angola", "ang gola"],
    "DUB": ["Dubai", "Dubai"],
    "DWT": ["Darwin", "darwin"],
    "EAF": ["Euro charter", "Eurocharter"],
    "EAL": ["Star Wing", "star wing"],
    "EAT": ["Trans Europe", "trans europe"],
    "ECA": ["Eurocypria", "EuroCypria"],
    "ECN": ["Euro Continental", "euro continental"],
    "EDC": ["Saltire", "saltire"],
    "EDW": ["Edelweiss", "edelweiss"],
    "EFF": ["Emerald", "emerald"],
    "EGL": ["Prestige", "prestige"],
    "EIA": ["Evergreen", "evergreen"],
    "EIN": ["Shamrock", "shamrock"],
    "EJM": ["Jet Speed", "jet speed"],
    "ELL": ["Estonian", "Estonian"],
    "ELY": ["Elal", "elal"],
    "EMX": ["Euromanx", "Euromanx"],
    "ENZ": ["Enzo", "enzo"],
    "ESK": ["Relax", "relax"],
    "ETD": ["Ety Had", "Etihad"],
    "ETH": ["Ethiopian", "eeth ee opian"],
    "ETI": ["Jet Hawk", "jet hawk"],
    "ETP": ["Tester", "tester"],
    "EUK": ["Snowbird", "snowbird"],
    "EVA": ["Eva", "A va"],
    "EWG": ["Eurowings", "Eurowings"],
    "EXM": ["Exam", "exam"],
    "EXS": ["Channex", "channex", "AMERICAN"],
    "EXT": ["Executive", "executive"],
    "EZE": ["Eastflight", "eastflight"],
    "EZS": ["Topswiss", "topswiss"],
    "EZY": ["Easy", "easy", "ENGLISH"],
    "FAF": ["French air force", "French airforce"],
    "FAH": ["Blue strip", "blue strip"],
    "FAT": ["Farner", "farner"],
    "FCA": ["Jet set", "jet set"],
    "FDX": ["Fedex", "fedex", "AMERICAN"],
    "FFD": ["First flight", "firstflight"],
    "FHY": ["Freebird", "freebird"],
    "FIN": ["Finnair", "Finnair"],
    "FKI": ["Kiel Air", "keel air"],
    "FLJ": ["Flairjet", "flair jet"],
    "FLT": ["Flight line", "flightline"],
    "FMY": ["French Army", "french army"],
    "FNF": ["Finnish Airforce", "finnish air force"],
    "FOB": ["Ford air", "fordair"],
    "FOR": ["Formula", "formula"],
    "FPG": ["Tag aviation", "tag aviation"],
    "FRA": ["Rushton", "Rushton"],
    "FSD": ["Flugservice", "flugservice"],
    "FUA": ["Futura", "futura"],
    "FYG": ["Flying group", "flying group"],
    "GAC": ["Dream Team", "dream team"],
    "GAF": ["German air force", "german airforce"],
    "GBL": ["GB airways", "GB airways", "ENGLISH"],
    "GDK": ["Goldeck Flug", "goldeck flug"],
    "GEC": ["Lufthansa cargo", "lufthansa cargo"],
    "GES": ["Gestair", "gestair"],
    "GFA": ["Gulf air", "gulfair"],
    "GIA": ["Indonesia", "in don ees ea"],
    "GMA": ["Gamma", "gamma"],
    "GMI": ["Germania", "germania"],
    "GNY": ["German navy", "german navy"],
    "GOJ": ["Gojet", "gojet"],
    "GOM": ["Gomel", "gomel"],
    "GRE": ["Greece airways", "greece airways"],
    "GRL": ["Greenland", "greenland"],
    "GSM": ["Globespan", "globespan"],
    "GSW": ["Arrow Jet", "arrow jet"],
    "GTH": ["Gotham", "gotham"],
    "GTI": ["Giant", "giant"],
    "GWI": ["German wings", "Germanwings"],
    "GWL": ["Great Wall", "great wall"],
    "HAF": ["Hellenic airforce", "hellenic airforce"],
    "HAT": ["Taxi bird", "taxibird"],
    "HCC": ["czech holidays", "czech holidays"],
    "HDA": ["Dragon", "dragon"],
    "HEJ": ["Hellas jet", "hellasjet"],
    "HFY": ["Sky Flyer", "sky flyer"],
    "HGR": ["Hang", "hang"],
    "HKY": ["Herky", "herky"],
    "HLF": ["Hapaglloyd", "hapaglloyd"],
    "HLX": ["Yellow Cab", "Yellow Cab"],
    "HMS": ["Hemus air", "hemusair"],
    "HOP": ["Air Hop", "air hop"],
    "HSG": ["Sky Dolphin", "sky dolphin"],
    "HVN": ["Vietnam Airlines", "vietnam airlines"],
    "HWY": ["Hiway", "highway"],
    "IAM": ["Italian Airforce", "italian airforce"],
    "IBE": ["Iberia", "Iberia"],
    "IBK": ["Nortrans", "nortrans"],
    "ICB": ["Ice bird", "icebird"],
    "ICE": ["Ice air", "iceair"],
    "ICL": ["Cal", "cal"],
    "IFA": ["Red angle", "redangle"],
    "IFT": ["Interflight", "interflight"],
    "IJM": ["Jet Management", "jet management"],
    "IMX": ["Zimex", "zimex"],
    "IRA": ["Iranair", "iran air"],
    "IRL": ["Irish", "Irish"],
    "ISK": ["Intersky", "intersky"],
    "ISS": ["Air Italy", "air italy"],
    "IWD": ["Iberworld", "iberworld"],
    "JAF": ["Beauty", "beauty"],
    "JAG": ["Jet Alliance", "jet alliance"],
    "JAI": ["Jet Airways", "jet airways"],
    "JAL": ["Japan air", "japan air"],
    "JAR": ["Airlink", "airlink"],
    "JAT": ["JAT", "jat"],
    "JCB": ["JCB", "JCB"],
    "JDI": ["Jedi", "jedi"],
    "JEF": ["Jetflite", "Jetflite"],
    "JFK": ["Keenair", "keenair"],
    "JGN": ["Joint guardian", "jointguardian"],
    "JKK": ["Spanair", "spanair"],
    "JTR": ["Jester", "jester"],
    "JXT": ["Vannin", "vannin"],
    "KAC": ["Kuwaiti", "Kuwaiti"],
    "KAL": ["Korean air", "koreanair"],
    "KFR": ["Kingfisher", "king fisher"],
    "KHB": ["Dalavia", "dalavia"],
    "KKK": ["Atlasjet", "Atlas Jet"],
    "KLM": ["KLM", "KLM"],
    "KQA": ["Kenya", "Kenya"],
    "KRF": ["Kittyhawk", "Kittyhawk", "ENGLISH"],
    "KYV": ["Air Kibris", "Airkibris"],
    "KZR": ["Astana Line", "astana line"],
    "LAA": ["Libair", "libe air"],
    "LAE": ["Lanco", "lanco"],
    "LAN": ["Lan Chile", "lan chile"],
    "LBC": ["Albanian", "albanian"],
    "LCO": ["LAN Chile Cargo", "lan chile cargo"],
    "LCT": ["Stellair", "stellair"],
    "LDA": ["Lauda air", "Laudaair"],
    "LDI": ["Lauda Italy", "Laudaitaly"],
    "LEE": ["Javelin", "javelin"],
    "LGL": ["Lux air", "luxair"],
    "LIL": ["Lithuania air", "Lithuania air"],
    "LKS": ["Airlin", "airlin"],
    "LMJ": ["Masterjet", "master jet"],
    "LNQ": ["Fastlink", "fast link"],
    "LNX": ["Lonex", "lonex"],
    "LOG": ["Logan", "logan"],
    "LOT": ["Pollot", "pollot"],
    "LOV": ["Loveair", "loveair"],
    "LTE": ["Fun jet", "funjet"],
    "LTU": ["LTU", "LTU"],
    "LXR": ["Air Luxor", "Airluxor"],
    "LZB": ["Flying Bulgaria", "Flyingbulgaria"],
    "MAH": ["Malev", "malev"],
    "MAS": ["Malaysian", "Malaysian"],
    "MAU": ["Air mauritius", "air mauritius"],
    "MCD": ["Air Med", "air med"],
    "MDG": ["Air Madagascar", "air mad a gas car"],
    "MDI": ["Delta Ice", "delta ice"],
    "MDT": ["Midnight", "midnight"],
    "MEA": ["Cedar jet", "cedarjet"],
    "MJE": ["Emjet", "emjet"],
    "MKA": ["Kruger air", "Krugerair"],
    "MMD": ["Mermaid", "mermaid"],
    "MNB": ["Black sea", "blacksea"],
    "MNL": ["Miniliner", "mini liner"],
    "MON": ["Monarch", "monarch", "ENGLISH"],
    "MOV": ["Mov air", "movair"],
    "MPD": ["Red comet", "redcomet"],
    "MPH": ["Martin air", "Martinair"],
    "MPJ": ["Mapjet", "map jet"],
    "MSR": ["Egyptair", "egyptair"],
    "MUA": ["Murray air", "Murrayair"],
    "MXA": ["Mexicana", "mexicana"],
    "MXY": ["Myjet", "myjet"],
    "MYO": ["Mayoral", "mayoral"],
    "NAF": ["Netherlands air forc", "Netherlands airforce"],
    "NAX": ["Norshuttle", "norshuttle"],
    "NCA": ["Nippon cargo", "nipponcargo"],
    "NEX": ["Neatax", "neatax"],
    "NFA": ["North flying", "northflying"],
    "NIG": ["Aeroline", "aeroline"],
    "NJE": ["Fraction", "fraction"],
    "NLY": ["Flyniki", "fly nicky"],
    "NMB": ["Air Namibia", "air nam ib ea"],
    "NOW": ["Norwegian", "Norwegian"],
    "NPT": ["Neptune", "neptune"],
    "NRD": ["North rider", "northrider"],
    "NRN": ["Netherlands navy", "netherlandsnavy"],
    "NTR": ["Nitro", "nitro"],
    "NVD": ["Nordvind", "nord vind"],
    "NVR": ["Navigator", "navigator"],
    "NVY": ["Navy", "navy"],
    "NWA": ["Northwest", "northwest"],
    "OAE": ["Omni Express", "omni express"],
    "OAI": ["Torline", "tour line"],
    "OAL": ["Olympic", "Olympic"],
    "OAS": ["Orchid", "orchid"],
    "OCN": ["O Bird", "o bird"],
    "OHY": ["Onur air", "onurair"],
    "OJT": ["Orystar", "orystar"],
    "OLT": ["Oltra", "oltra"],
    "OOM": ["Zoom", "zoom"],
    "ORF": ["Oman", "oman"],
    "OVA": ["Aeronova", "aeronova"],
    "OXE": ["Oxoe", "oxo"],
    "OXF": ["Oxford", "oxford"],
    "PAC": ["Polar", "polar"],
    "PCH": ["Pilatus wings", "pilatuswings"],
    "PGA": ["Portugalia", "Portugalia"],
    "PGL": ["Premiere", "premiere"],
    "PGT": ["Sunturk", "sunturk"],
    "PHV": ["New bird", "newbird"],
    "PIA": ["Pakistan", "Pakistan"],
    "POE": ["Porter Air", "porter air"],
    "QAF": ["Amiri", "amiri"],
    "QAJ": ["Dagobert", "dagobert"],
    "QFA": ["Quantas", "quantas"],
    "QID": ["Quid", "quid"],
    "QTR": ["Qatari", "Cat arr ee"],
    "RAE": ["Regional europe", "regional Europe"],
    "RAM": ["Royalair maroc", "royalair Maroc"],
    "RCH": ["Reach", "reach"],
    "REA": ["Aer Arann", "airAran"],
    "REI": ["Ray aviation", "rayaviation"],
    "RFR": ["raf air", "Rafair"],
    "RJA": ["Jordanian", "Jordanin"],
    "ROF": ["Romanian Air Force", "Romanian Air Force"],
    "ROT": ["Tarom", "Tarom"],
    "RPX": ["Rapex", "rapex"],
    "RRF": ["Kitty", "kitty"],
    "RRL": ["Merlin", "Merlin"],
    "RRR": ["Ascot", "Ascot"],
    "RSF": ["Arsaf", "Arsaf"],
    "RVL": ["Air Vallee", "air valley"],
    "RVR": ["Raven", "raven"],
    "RYR": ["Ryanair", "Ryanair"],
    "SAA": ["Springbok", "springbok"],
    "SAF": ["Singa", "singa"],
    "SAS": ["Scandanavian", "Scandanavian"],
    "SAZ": ["Swiss ambulance", "swissambulance"],
    "SCM": ["Screamer", "screamer"],
    "SDL": ["Skydrift", "Skydrift"],
    "SFB": ["Air sofia", "airSofia"],
    "SHF": ["Vortex", "vortex"],
    "SHT": ["Shuttle", "shuttle"],
    "SIA": ["Singapore", "Singapore"],
    "SKS": ["Skyservice", "skyservice"],
    "SMX": ["Aliexpress", "alley express"],
    "SNB": ["Sterling", "sterling"],
    "SNM": ["Servizi Aerei", "servitsi airai"],
    "SOO": ["Southern Air", "southern air"],
    "SPR": ["Speedair", "speedair"],
    "SQC": ["Singcargo", "singcargo"],
    "SRR": ["Whitestar", "whitestar"],
    "SSP": ["Starspeed", "starspeed"],
    "STK": ["Stobart", "stobart", "ENGLISH"],
    "STU": ["Luxliner", "luxliner"],
    "SUI": ["Swiss air force", "swiss airforce"],
    "SUS": ["Sunscan", "sunscan"],
    "SVA": ["Saudia", "Saudia"],
    "SVH": ["Silver", "silver"],
    "SVK": ["Slovakia", "slovakia"],
    "SWG": ["Sunwing", "sun wing"],
    "SWN": ["Air sweden", "airsweden"],
    "SWQ": ["Swiftflight", "swift flight"],
    "SWR": ["Swiss", "Swiss"],
    "SWT": ["Swift", "swift"],
    "SXN": ["Saxonair", "saxonair"],
    "SYR": ["Syrianair", "syrian air"],
    "TAM": ["TAM", "tam"],
    "TAP": ["Air portugal", "airPortugal"],
    "TAR": ["Tune air", "tuneair"],
    "TAY": ["Quality", "quality"],
    "TCV": ["Caboverde", "cab o verd"],
    "TCX": ["Kestrel", "kestrel"],
    "TFL": ["Arkefly", "arkefly"],
    "THA": ["Thai", "Thai"],
    "THT": ["Tahiti airlines", "tahitiairlines"],
    "THY": ["Turkish", "Turkish"],
    "TJS": ["Tyroljet", "tirrol jet"],
    "TLB": ["Triple A", "Triple A"],
    "TOM": ["tomjet", "Tomjet"],
    "TRA": ["Transavia", "transavia"],
    "TRJ": ["Trans Euro", "transeuro"],
    "TSC": ["Transat", "transat"],
    "TSI": ["Transport air", "transportair"],
    "TUA": ["Turkmenistan", "turkmenistan"],
    "TUI": ["too e jet", "Too e jet"],
    "TWI": ["Tailwind", "tailwind"],
    "TYR": ["Tyrolean", "Tyrolean"],
    "UAE": ["Emirates", "emirates"],
    "UAF": ["Uniforce", "uniforce"],
    "UAL": ["United", "united", "AMERICAN"],
    "UJT": ["Uni-Jet", "uni jet"],
    "UPA": ["Foyle", "foil"],
    "UPS": ["UPS", "UPS", "AMERICAN"],
    "UZB": ["Uzbek", "Uzz Bek"],
    "VAA": ["Eurovan", "euro van"],
    "VAL": ["Voyageur", "voyageur"],
    "VDA": ["Volga dnepr", "Volga dnepr"],
    "VEA": ["Vega airlines", "vega airlines"],
    "VIK": ["Swedjet", "swedjet"],
    "VIR": ["Virgin", "virgin", "ENGLISH"],
    "VIZ": ["Aeroviz", "aero viz"],
    "VJS": ["Vista Jet", "vista jet"],
    "VJT": ["Vista", "vista"],
    "VKG": ["Viking", "viking"],
    "VKH": ["Delphi", "delphi"],
    "VLE": ["Vola", "vola"],
    "VLG": ["Vueling", "Vweyling"],
    "VLM": ["Rubens", "Rubens"],
    "VMP": ["Vampire", "vampire"],
    "VRG": ["Varig", "varig"],
    "VSB": ["Vickers", "vickers"],
    "VUE": ["Flightvue", "flightvue"],
    "VYT": ["Anglesea", "Anglesea"],
    "WDG": ["Watchdog", "watchdog"],
    "WDL": ["WDL", "WDL"],
    "WEA": ["White Eagle", "white eagle"],
    "WGP": ["Grand Prix", "grond pree"],
    "WIF": ["Wideroe", "wideroe"],
    "WLX": ["West lux", "westlux"],
    "WOA": ["World", "world"],
    "WOW": ["Wowair", "wowair"],
    "WZZ": ["Wizz air", "wizzair"],
    "XLA": ["Expo", "expo"],
}

frequencies: dict[str, list[str]] = {}

DESCEND_PHRASEOLOGY: str = "dee send"


def spell_phonetically(spelled_output: str) -> str:
    """
    Break up a string and read out phonetically character by character
    """
    return " ".join([phonetic[character.capitalize()] for character in spelled_output])


def beautify_callsign(callsign: str) -> str:
    """
    Abbreviates or reads out callsign
    """

    RT_callsign = ""
    if callsign[0] == "N" and callsign[1] in [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ]:
        abbreviation = "N" + callsign[-3:]
        return spell_phonetically(abbreviation)
    if callsign[0:3] in operators:
        RT_callsign += operators[callsign[0:3]][0]
        identifier = callsign[3:]
        for i in range(0, len(identifier)):
            RT_callsign += " " + phonetic[identifier[i]]

        return RT_callsign
    if re.search(r"[A-Z]{3}[0-9]{1,4}[A-Z]*", callsign):
        return spell_phonetically(callsign)

    abbreviation = callsign[0] + callsign[-2:]
    return spell_phonetically(abbreviation)


def which_way(current: float, cleared: float) -> Literal["left", "right"]:
    left = ((current - cleared) + 360) % 360
    right = ((cleared - current) + 360) % 360
    if left < right:
        return "left"
    return "right"


def text_phraseology(action: Action, environment: Environment) -> ClearanceAndResponse:
    """
    Generate clearance that would be issued for an action with matching response

    An ATC clearance is an authorisation for an aircraft to proceed under
    specific conditions within controlled airspace.  This clearance would be
    issued to a pilot, who then responds with a read-back.

    The clearance depend primarily on the action, but can also depend on the
    current state of the aircraft and its flight plan. For example, an action to
    change flight level to FL200 could either be issued as "climb flight level
    200" or "descend flight level 200", depending on the altitude when the
    clearance was delivered.  This reference data is contained in the
    environment object.

    Parameters
    ----------
    action: Action
        Action to be converted to an ATC clearance
    environment: Environment
        Current state of the environment (in particular the Aircraft objects)

    Returns
    ----------
    clearance : str
        Clearance to be issued to a pilot
    response : str
        Expected pilot response (read-back) from an issued clearance

    Notes
    ----------
    Clearance and response are returned as a named tuple.

    Reference:
    https://www.faa.gov/air_traffic/publications/atpubs/aim_html/chap4_section_4.html
    "FAA: ATC Clearances and Aircraft Separation"
    """
    callsign: str = action.callsign
    action_kind: str = action.kind
    value: int | float | str | list[str] | tuple[int, str] | None = action.value

    aircraft: Aircraft | None = environment.aircraft.get(callsign, None)

    if aircraft is None:
        logger.warning(f"Callsign {callsign} unavailable in environment.", stacklevel=2)
        aircraft = Aircraft(
            lat=0,
            lon=0,
            fl=0,
            heading=0,
            flight_plan=FlightPlan(route=Route(["ALPHA", "BRAVO"])),
            callsign=callsign,
        )

    fl = aircraft.fl
    cleared_fl = aircraft.cleared_fl

    clearance_parts: list[str] = []

    match action_kind:
        # vertical actions: "change_flight_level_to", "change_flight_level_by", "change_vertical_speed_to",
        #                   "descend_when_ready,level_by_fix", "descend_now,level_by_fix"
        case "change_flight_level_to":
            # Only for text clearance, (not phonetic clearance)
            direction = "climb" if value > fl else "descend"
            # Aircraft flight level stored as a float - change to int
            value_string = str(int(value))
            clearance_parts.extend([direction, "flight level", value_string])

        case "change_flight_level_by":
            # Possibly doesn't exist as a required action/clearance
            clearance_parts.extend(["change flight level by", str(value)])

        case "change_vertical_speed_to":
            direction = "climb" if cleared_fl > fl else "descent"
            clearance_parts.extend(["rate of", direction, str(value), "feet per minute"])

        case "descend_when_ready,level_by_fix":
            assert isinstance(value, tuple)
            if aircraft.on_route:
                clearance_parts.extend(["when ready descend flight level", str(value[0]), "level by", value[1]])
            else:
                clearance_parts.extend(["when ready descend flight level", str(value[0]), "level abeam", value[1]])

        case "descend_now,level_by_fix":
            assert isinstance(value, tuple)
            if aircraft.on_route:
                clearance_parts.extend(["descend flight level", str(value[0]), "level by", value[1]])
            else:
                clearance_parts.extend(["descend flight level", str(value[0]), "level abeam", value[1]])

        # lateral actions: "route_direct_to", "change_heading_to", "change_heading_by", "maintain_current_heading",
        #                  "change_cas_to", "change_mach_to"
        case "change_heading_to":
            # Values come from HMI as floats - change to int before formatting.
            new_heading = f"{int(value):03d} degrees"
            clearance_parts.extend(["fly heading", new_heading])

        case "maintain_current_heading":
            clearance_parts.append("continue present heading")

        case "change_heading_by":
            turn_direction = "right" if value > 0 else "left"
            turn_amount = f"{abs(value)} degrees"
            clearance_parts.extend(["turn", turn_direction, turn_amount])

        case "route_direct_to":
            if isinstance(value, str):
                fix: str = WAYPOINTS.get(value, value)

            elif isinstance(value, list):
                fix = "[" + ", ".join(WAYPOINTS.get(v, str(v)) for v in value) + "]"

            else:
                # Values other than strings or lists can still be converted
                fix = str(value)

            if aircraft.cleared_instructions.on_route:
                clearance_parts.extend(["route direct", fix])
            else:
                clearance_parts.extend(["resume own navigation", str(fix)])

        case "hold_at_location":
            assert isinstance(value, HoldParameters)
            minutes = f"{value.outbound_time / 60.0:g}"
            location = value.fix or f"latitude {value.location[0]:g} longitude {value.location[1]:g}"
            clearance_parts.extend(
                ["hold at", location, value.turn_direction, "hand", "outbound time", minutes, "minutes"]
            )

        case "change_cas_to":
            clearance_parts.extend(["fly speed", str(value), "knots"])

        case "change_mach_to":
            # Future phonetic clearance example: 0.74 -> "decimal seven four"
            clearance_parts.extend(["fly speed mach", str(value)])

        # outcomm action: "outcomm"
        case "outcomm":
            clearance_parts.append("contact next frequency")

        case "using_speed_limit":
            if value:
                clearance_parts.append("obeying speed limit")
            else:
                clearance_parts.append("no speed restrictions")

        case "set_squawk":
            return ClearanceAndResponse(
                clearance=f"{callsign}, set squawk {value!s}",
                pilot_response=f"Squawking {value!s}, {callsign}",
            )

        case "squawk_ident":
            return ClearanceAndResponse(
                clearance=f"{callsign}, squawk ident",
                pilot_response=f"Squawking ident, {callsign}",
            )
        case "change_heading_to_by_direction":
            return ClearanceAndResponse(
                clearance=f"{callsign}, turn {action.value[1]} heading {action.value[0]}",
                pilot_response=f"Turning {action.value[1]} heading {action.value[0]}, {callsign}",
            )

        case "message":
            return ClearanceAndResponse(clearance=str(value), pilot_response="")

        # other actions not yet modelled. in theory, this should never be hit if the case statements above capture all
        # possible valid actions because the Action() class should prevent invalid actions from being created
        case _:
            logger.warning(
                f"Aircraft {callsign} issued unavailable action: {action!s}",
                stacklevel=2,
            )
            clearance_parts.extend([action_kind, str(value)])

    clearance = " ".join([callsign, *clearance_parts])
    pilot_response = " ".join([*clearance_parts, callsign])

    return ClearanceAndResponse(clearance=clearance, pilot_response=pilot_response)


def voice_phraseology(action: Action, environment: Environment) -> ClearanceAndResponse:
    """
    Generate clearance that would be issued for an action with matching response

    An ATC clearance is an authorisation for an aircraft to proceed under
    specific conditions within controlled airspace.  This clearance would be
    issued to a pilot, who then responds with a read-back.

    The clearance depend primarily on the action, but can also depend on the
    current state of the aircraft and its flight plan. For example, an action to
    change flight level to FL200 could either be issued as "climb flight level
    200" or "descend flight level 200", depending on the altitude when the
    clearance was delivered.  This reference data is contained in the
    environment object.

    Parameters
    ----------
    action: Action
        Action to be converted to an ATC clearance
    environment: Environment
        Current state of the environment (in particular the Aircraft objects)

    Returns
    ----------
    clearance : str
        Clearance to be issued to a pilot
    response : str
        Expected pilot response (read-back) from an issued clearance

    Notes
    ----------
    Clearance and response are returned as a named tuple.

    Reference:
    https://www.faa.gov/air_traffic/publications/atpubs/aim_html/chap4_section_4.html
    "FAA: ATC Clearances and Aircraft Separation"
    """
    callsign: str = action.callsign
    action_kind: str = action.kind
    value: int | float | str | list[str] | tuple[int, str] | None = action.value

    aircraft: Aircraft | None = environment.aircraft.get(callsign, None)

    if aircraft is None:
        logger.warning(f"Callsign {callsign} unavailable in environment.", stacklevel=2)
        aircraft = Aircraft(
            lat=0,
            lon=0,
            fl=0,
            heading=0,
            flight_plan=FlightPlan(route=Route(["ALPHA", "BRAVO"])),
            callsign=callsign,
        )

    fl = aircraft.fl
    cleared_fl = aircraft.cleared_fl

    clearance_parts: list[str] = []

    match action_kind:
        # vertical actions: "change_flight_level_to", "change_flight_level_by", "change_vertical_speed_to",
        #                   "descend_when_ready,level_by_fix", "descend_now,level_by_fix"
        case "change_flight_level_to":
            direction = "climb" if value > fl else "dee send"
            # Aircraft flight level stored as a float - change to int
            value_string = str(int(value))
            if value_string[-2:] == "00":
                clearance_parts.extend(
                    [
                        direction,
                        "flight level",
                        spell_phonetically(value_string[0]),
                        "hundred",
                    ]
                )
            else:
                clearance_parts.extend([direction, "flight level", spell_phonetically(value_string)])

        # case "change_flight_level_by":
        # Possibly doesn't exist as a required action/clearance
        # clearance_parts.extend(["change flight level by", str(value)])

        case "change_vertical_speed_to":
            direction = "climb" if cleared_fl > fl else DESCEND_PHRASEOLOGY
            clearance_parts.extend(["rate of", direction, str(value), "feet per minute"])

        case "descend_when_ready,level_by_fix":
            assert isinstance(value, tuple)
            value_string = str(int(value[0]))
            if value_string[-2:] == "00":
                clearance_parts.extend(
                    [
                        f"when ready {DESCEND_PHRASEOLOGY} flight level",
                        spell_phonetically(str(value[0])[0]),
                        "hundred level by",
                        value[1],
                    ]
                )
            else:
                clearance_parts.extend(
                    [
                        f"when ready {DESCEND_PHRASEOLOGY} flight level",
                        spell_phonetically(str(value[0])),
                        "level by",
                        value[1],
                    ]
                )

        case "descend_now,level_by_fix":
            assert isinstance(value, tuple)
            clearance_parts.extend(
                [
                    f"{DESCEND_PHRASEOLOGY} flight level",
                    spell_phonetically(str(value[0])),
                    "level by",
                    value[1],
                ]
            )

        # lateral actions: "route_direct_to", "change_heading_to", "change_heading_by", "maintain_current_heading",
        #                  "change_cas_to", "change_mach_to"
        case "change_heading_to":
            # Values come from HMI as floats - change to int before formatting
            new_heading = f"{int(value):03d}"
            # ensure the new_heading is in the range [005,360]
            if new_heading == "000":
                new_heading = "360"

            if new_heading[-2:] == "00":
                clearance_parts.extend(
                    [
                        "turn",
                        which_way(aircraft.heading, value),
                        "heading",
                        spell_phonetically(new_heading[0]),
                        "hundred degrees",
                    ]
                )
            elif new_heading[-1] == "0":
                clearance_parts.extend(
                    [
                        "turn",
                        which_way(aircraft.heading, value),
                        "heading",
                        spell_phonetically(new_heading),
                        "degrees",
                    ]
                )
            else:
                clearance_parts.extend(
                    [
                        "turn",
                        which_way(aircraft.heading, value),
                        "heading",
                        spell_phonetically(new_heading),
                    ]
                )

        case "maintain_current_heading":
            clearance_parts.append("continue present heading")

        case "change_heading_by":
            turn_direction = "right" if value > 0 else "left"
            turn_amount = f"{abs(value)}"
            clearance_parts.extend(["turn", turn_direction, spell_phonetically(turn_amount), "degrees"])

        case "route_direct_to":
            if isinstance(value, str):
                fix: str = WAYPOINTS.get(value, value)

            elif isinstance(value, list):
                fix = "[" + ", ".join(WAYPOINTS.get(v, str(v)) for v in value) + "]"

            else:
                # Values other than strings or lists can still be converted
                fix = str(value)

            if aircraft.cleared_instructions.on_route:
                clearance_parts.extend(["route direct", fix])
            else:
                clearance_parts.extend(["resume own navigation", str(fix)])

        case "hold_at_location":
            assert isinstance(value, HoldParameters)
            minutes = f"{value.outbound_time / 60.0:g}"
            location = value.fix or f"latitude {value.location[0]:g} longitude {value.location[1]:g}"
            clearance_parts.extend(
                ["hold at", location, value.turn_direction, "hand", "outbound time", minutes, "minutes"]
            )

        case "change_cas_to":
            clearance_parts.extend(["speed", spell_phonetically(str(value)), "knots"])

        case "change_mach_to":
            clearance_parts.extend(["speed mach", spell_phonetically(str(value))])

        case "using_speed_limit":
            if value:
                return ClearanceAndResponse(
                    clearance=f"{beautify_callsign(callsign)} apply speed restrictions",
                    pilot_response=f"applying speed restrictions {beautify_callsign(callsign)}",
                )
            # Compliant phraseology with CAP413, paragraph 6.3.
            return ClearanceAndResponse(
                clearance=f"{beautify_callsign(callsign)} no a t c speed restrictions",
                pilot_response=f"no speed restrictions {beautify_callsign(callsign)}",
            )

        case "change_heading_to_by_direction":
            return ClearanceAndResponse(
                clearance=f"{callsign}, turn {action.value[1]} heading {action.value[0]}",
                pilot_response=f"Turning {action.value[1]} heading {action.value[0]}, {callsign}",
            )
        # outcomm action: "outcomm"
        case "outcomm":
            next_sector = environment.next_sector_of_aircraft(callsign)

            if next_sector in frequencies:
                frequency_value = frequencies[next_sector][1]
                frequency_callsign = frequencies[next_sector][0]

                if frequency_value[-2:] == "00":
                    clearance_parts.extend(
                        "contact",
                        frequency_callsign,
                        spell_phonetically(frequency_value[0:5]),
                    )
                else:
                    clearance_parts.extend("contact", frequency_callsign, spell_phonetically(frequency_value))

            else:
                clearance_parts.extend("contact next frequency")

        # other actions not yet modelled. in theory, this should never be hit if the case statements above capture all
        # possible valid actions because the Action() class should prevent invalid actions from being created
        case _:
            logger.warning(
                f"Aircraft {callsign} issued unavailable action: {action!s}",
                stacklevel=2,
            )
            clearance_parts.extend([action_kind, str(value)])

    clearance = " ".join([beautify_callsign(callsign), *clearance_parts])
    pilot_response = " ".join([*clearance_parts, beautify_callsign(callsign)])

    return ClearanceAndResponse(clearance=clearance, pilot_response=pilot_response)


def add_phraseology(action: Action, environment: Environment) -> Action:
    action.voice_representation = voice_phraseology(action, environment)
    action.text_representation = text_phraseology(action, environment)
    return action
