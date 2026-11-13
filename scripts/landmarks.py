# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Named places of Hoogeveen: the viewer's place list and the stories on its info cards.

Two sources. CURATED holds the places with a story, found by their BAG address or a position; the facts come from
the Rijksmonumentenregister (RCE), archimon.nl, standbeelden.vanderkrogt.net, visitdrenthe.com (city walk
Trommelslager), binnenstadhoogeveen.nl, hoogeveen2000.com (Albert Metselaar), vangoghhuis.nl, cascaderun.nl,
artatsite.com and landschaplopen.com. The rest comes from OpenStreetMap (data/raw/osm_landmarks.json): churches,
only for the position of some of them. A place without a story is not on the map.

An entry may also carry style overrides for the render ("style", "wall", "roof", "tower", ...); none do yet.

Output: data/build/landmarks.json
"""

import json
import re

import numpy as np

from config import BUILD, CX, CY, RAW, ROOT, XMAX, XMIN, YMAX, YMIN
from geo import rd_to_wgs, wgs_to_rd

RM = "Rijksmonument"

# Places with a story. "addr" is a BAG address (the building under it becomes the hover area), "latlon" a point,
# "osm" the name of the OpenStreetMap element whose position to use (and which is then not listed twice).
CURATED = [
    {
        "name": "Grote Kerk",
        "addr": "Grote Kerkstraat 40",
        "kind": f"{RM} · 17e eeuw",
        "note": "Kruiskerk uit de 17e eeuw, later uitgebouwd tot hallenkerk",
        "address": "Grote Kerkstraat 40",
        "osm": ["Grote Kerk"],
        "text": "De bouw begon rond 1652, als kruiskerk met vier even lange armen. Tussen 1766 en 1804 is hij in drie stappen uitgebouwd tot een driebeukige hallenkerk. Hoogeveen werd niet met klokken maar met een trom naar de kerk geroepen. Het bekende verhaal dat de veengrond geen toren kon dragen klopt volgens lokaal historicus Albert Metselaar niet: de eerste kerk had een toren, maar klokken vond men te duur, en toen er later klokken werden aangeboden wees men ze af, want er was al een trommelslager. In de klokkenstoel hangt een klok van F. en P. Hemony uit 1649. Boven de zuidingang zit een houten zonnewijzer, gesigneerd Hend. Casp. Dekker, 1804.",
    },
    {
        "name": "Trommelslager",
        "osm": ["Trommelslager"],
        "kind": "Standbeeld · 1971",
        "note": "Beeld van Johan Sterenberg",
        "text": "Beeld van Johan Sterenberg uit 1971. De trommelslager riep Hoogeveen elke zondag naar de kerk en rukte ook uit bij brand. Waar het gebruik vandaan komt is niet zeker. De waarschijnlijkste verklaring is een militaire: Roelof van Echten, de man achter de vervening, was kapitein, en in Heerenveen begon en eindigde het werk in het veen in 1622 ook al op de trom. In 1754 is er over gepraat om de trommelslager af te schaffen. Het is er niet van gekomen. Het beeld is een keer gestolen en staat weer op zijn sokkel.",
    },
    {
        "name": "Remonstrantse kerk",
        "addr": "Grote Kerkstraat 39",
        "kind": "Kerk · 1894",
        "address": "Grote Kerkstraat 39",
        "osm": ["Remonstrantse Kerk"],
        "text": "Gebouwd in 1893 en 1894 in neorenaissancestijl, naar ontwerp van gemeentearchitect J. Carmiggelt.",
    },
    {
        "name": "Hoofdstraatkerk",
        "addr": "Hoofdstraat 95",
        "kind": "Kerk · 1904",
        "address": "Hoofdstraat 95",
        "osm": ["Protestantse Gemeente Hoogeveen"],
        "text": "Gereformeerde kerk uit 1904 van architect G. Hoekzema, in een eigenzinnige neoromaanse stijl met hoektorens. Hoog in de gevel staat een steen met de jaartallen 1834-1886: de Afscheiding en de Doleantie, de twee keren dat groepen de Hervormde Kerk verlieten.",
    },
    {
        "name": "Gemeentehuis",
        "addr": "Raadhuisplein 1",
        "kind": f"{RM} · 1940",
        "note": "Raadhuis met carillontoren",
        "address": "Raadhuisplein 1",
        "text": "Raadhuis uit 1940 van architect C.J. Blaauw, in de stijl van de Delftse School, met een toren waarin een carillon hangt. Het is gebouwd met het geld dat overbleef toen de Compagnie van de 5000 Morgen, de veenmaatschappij waar Hoogeveen mee begon, werd opgeheven. De laatste NSB-burgemeester, Jan Marinus Veldhuis, was na een spoedcursus benoemd en zat er van juli 1944 tot april 1945. Toen de Canadezen kwamen vluchtte hij op de fiets en verstopte hij zich in een wc. Op het Raadhuisplein start elk voorjaar de Cascaderun.",
    },
    {
        "name": "Schippershuus",
        "addr": "Hoofdstraat 254",
        "kind": f"{RM} · 17e eeuw",
        "note": "Rentmeesterskantoor van de Compagnie, later schipperscafé",
        "address": "Hoofdstraat 254",
        "text": "Het oudste gebouw van Hoogeveen, met een kern uit 1632 of 1633. De Compagnie van de 5000 Morgen liet het bouwen als kantoor van de rentmeester, waar de bewoners hun pacht kwamen betalen. De eerste rentmeester die we kennen is Carst Peters uit Hasselt. Na een verbouwing in 1882 werd het een café waar de turfschippers samenkwamen. Op het pleintje erachter staat een stenen huisje van één kamer dat doorgaat voor het oudste huis van Hoogeveen, uit 1630. Twee vrijwilligers van het museum hebben het in 2006 bekeken: het metselwerk is twintigste-eeuws en in het register van 1758 komt het huisje niet voor.",
    },
    {
        "name": "Huis met de Duivengaten",
        "addr": "Alteveerstraat 1",
        "kind": f"{RM} · eind 17e eeuw",
        "address": "Alteveerstraat 1",
        "text": "Huis uit het eind van de 17e eeuw, met een gevel vol duivengaten, waterlijsten en pinakels. Het staat aan Het Kruis, de plek waar Hoogeveen begon. Nu is het een restaurant.",
    },
    {
        "name": "Het Kruis",
        "latlon": (52.72056, 6.47778),
        "kind": "Hier begon Hoogeveen",
        "text": "Op deze kruising van de Hoofdstraat, Het Haagje, de Alteveerstraat en de Schutstraat kwamen twee vaarten samen. Hier vestigden zich de eerste winkeliers, turfhandelaren en ambachtslieden. Op de kaart van 1637 liep de vaart die nu Het Haagje is door tot aan het Groote Meer, dat verdwenen is. De vaarten zijn gedempt: de Hoofdstraat als eerste, in 1948. Op 2 oktober 1883 stapte Vincent van Gogh hier op de trekschuit naar Nieuw-Amsterdam.",
    },
    {
        "name": "Turfschipper",
        "osm": ["Turfschipper"],
        "kind": "Standbeeld · 1982",
        "note": "Schippersmonument van Arie Teeuwisse",
        "text": "Bronzen turfschipper van Arie Teeuwisse, met op zijn rug het anker waarmee hij zijn schip vastlegde. Commissaris der Koningin A.P. Oele onthulde het beeld op 24 november 1982. Het stond eerst bij het Kerkplein en is verhuisd toen de Cascade werd aangelegd.",
    },
    {
        "name": "De Cascade",
        "latlon": (52.723544, 6.477679),
        "kind": "Waterkunstwerk",
        "note": "Waterkunstwerk van Karin Daan in de Hoofdstraat",
        "text": "Waterkunstwerk van Karin Daan, zo'n 250 meter lang en 10 meter breed, midden in de Hoofdstraat. Hier liep de vaart waarover de turf werd afgevoerd; het laatste stuk is in 1972 gedempt. De Cascade wordt het langste kunstwerk van Europa genoemd. Volgens Albert Metselaar zei de ontwerper zelf dat ze zich had laten inspireren door oosterse tuinen, niet door het veenverleden. Hij noemt het met een knipoog het grootste vruchtbaarheidssymbool van Nederland. De Cascaderun, waarvan de finish hier ligt, is ernaar genoemd.",
    },
    {
        "name": "De Tas",
        "osm": ["De Tas"],
        "kind": "Kunstwerk · 1999",
        "text": "Boodschappentas van Gert-Jan Mulder bij de ingang van de Tamboerpassage, geplaatst toen het winkelcentrum in 1999 openging. Hij kostte 95.000 gulden.",
    },
    {
        "name": "Schouwburg De Tamboer",
        "addr": "Hoofdstraat 17",
        "kind": "Theater",
        "address": "Hoofdstraat 17",
        "web": "https://www.detamboer.nl",
        "text": "De schouwburg is genoemd naar de trommelslager. Bij het gebouw staan De Tamboer van Gooitzen de Jong uit 1967 en vier figuren uit de commedia dell'arte van Arie Teeuwisse uit 1989: Colombina, Pantalone, Pulcinella en Scapino.",
    },
    {
        "name": "Logement Hartsuiker",
        "addr": "Pesserstraat 24",
        "kind": "Van Gogh sliep hier · 1883",
        "note": "Hier logeerde Vincent van Gogh in 1883",
        "address": "Pesserstraat 24",
        "web": "https://logementhartsuiker.nl",
        "text": "Vincent van Gogh kwam op dinsdag 11 september 1883 om negen uur 's avonds met de trein in Hoogeveen aan. Spoorwegarbeider Albertus Hartsuiker bood hem kost en inwoning aan voor een gulden per dag. Hij bleef een paar weken en trok daarna verder Drenthe in. Lang was niet duidelijk welk huis het was; sinds onderzoek van Albert Metselaar uit 2001 geldt dit logement aan de Pesserstraat, toen de Toldijk, als het adres. Op de gevel zit een plaquette.",
    },
    {
        "name": "Station Hoogeveen",
        "addr": "Stationsplein 1",
        "kind": "Station",
        "address": "Stationsplein 1",
        "text": "De spoorlijn van Meppel naar Groningen ging op 1 mei 1870 open, met een station in Hoogeveen. Het gebouw van nu is uit 1984 en ontworpen door Cees Douma.",
    },
    {
        "name": "Het Klooster",
        "addr": "Brinkstraat 5",
        "kind": "Voormalig klooster · ca. 1905",
        "address": "Brinkstraat 5",
        "osm": ["Het Klooster"],
        "text": "De paters karmelieten uit Zenderen lieten dit neogotische gebouw ontwerpen door architect K.L. Croonen uit Almelo. Het kwam rond 1905 in gebruik. Vanuit Hoogeveen bedienden ze de katholieken die verspreid over de streek woonden. In 1957 vertrokken ze.",
    },
    {
        "name": "Dubbel woonhuis Brinkstraat",
        "addr": "Brinkstraat 32",
        "kind": f"{RM} · 1904",
        "address": "Brinkstraat 32 en 34",
        "text": "Dubbel woonhuis uit 1904, ontworpen door G. Hofman voor mevrouw J. van Hees, met art-nouveaudetails: glas-in-lood, bloemmotieven en mozaïeken van baksteen. Het stond aan de Hoogeveensche Vaart, die hier nu een straat is.",
    },
    {
        "name": "Voormalige synagoge",
        "addr": "Schutstraat 147",
        "kind": "Voormalige synagoge",
        "address": "Schutstraat 147",
        "osm": ["Baptistengemeente de Schutse"],
        "text": "De synagoge van Hoogeveen. Nu kerkt de baptistengemeente De Schutse er.",
    },
    {
        "name": "Molen De Zwaluw",
        "addr": "Van Echtenstraat 47",
        "kind": "Korenmolen · 1834",
        "address": "Van Echtenstraat 47",
        "text": "Korenmolen uit 1834, de opvolger van een molen die in 1787 afbrandde. De naam kreeg hij pas na de restauratie van 1980. Hij maalt nog: als het niet waait op een elektromotor. Het meel is te koop in de molenwinkel.",
    },
    {
        "name": "Oude Joodse Begraafplaats",
        "osm": ["Oude Joodse Begraafplaats"],
        "kind": "Begraafplaats · 18e eeuw",
        "text": "Joodse begraafplaats uit de 18e eeuw, achter de oude pastorie aan de Van Echtenstraat. De bronnen noemen 1725 en 1730 als begin; hij was in gebruik tot 1839. Daarna kwam de begraafplaats aan de Zuiderweg, uit 1829.",
    },
    {
        "name": "De Hoogeveense Kei",
        "osm": ["De Hoogeveense Kei"],
        "kind": "Zwerfkei",
        "text": "Zwerfkei die in de ijstijd uit Scandinavië is meegekomen en in het Hollandscheveld is gevonden. Toen de kei in 2008 werd verplaatst, kwam er een loden koker onder vandaan met een oorkonde van 14 juni 1955, gericht aan het nageslacht in het atoomtijdperk. Dat schrijft de wandelaar van landschaplopen.com, die het verhaal bij de kei zelf vond.",
    },
    {
        "name": "De kerk op de rotonde",
        "osm": ["De kerk op de rotonde"],
        "kind": "Kunstwerk · 2008",
        "text": "Kunstwerk van Filip Jonker uit 2008, op de rotonde van de Middenveldweg en de A.G. Bellstraat.",
    },
    {
        "name": "De Turfgravers",
        "osm": ["Turfgravers"],
        "kind": "Beeld · 1975",
        "text": "Beeld van Han Wezelaar uit 1975, in het Burgemeester Tjalmapark.",
    },
    {
        "name": "Piet van de Velde",
        "osm": ["Pieter Cornelis Marinus van de Velde (1897-1945)"],
        "kind": "Standbeeld · 2010",
        "text": "Beeld van Daphné Du Barry uit 2010, bij het ziekenhuis.",
    },
    {
        "name": "Ziekenhuis Bethesda",
        "addr": "Dr G H Amshoffweg 1",
        "kind": "Ziekenhuis",
        "address": "Dr. G.H. Amshoffweg 1",
        "text": "Het ziekenhuis is {h} meter hoog. Rond het gebouw staan vijf beelden: De Haan van Frank Letterie (1970), De Belhamel van Adri de Waard (1988), De Bron van Meri Matacin (1989), De Turenden van Sjer Jacobs (2002) en Piet van de Velde van Daphné Du Barry (2010).",
    },
    {
        "name": "Voormalig postkantoor",
        "addr": "Hoofdstraat 167",
        "kind": "Postkantoor · 1912",
        "address": "Hoofdstraat 167",
        "text": "Het postkantoor uit 1912, ontworpen door rijksbouwmeester C.H. Peters. Op 17 februari 1944 overviel het verzet het postkantoor van Hoogeveen en nam zo'n 13.000 bonkaarten mee, voor onderduikers. Loketbeambte Henk Raak, die had meegeholpen, is later opgepakt en in de duinen bij Overveen gefusilleerd.",
    },
    {
        "name": "Bibliotheek Hoogeveen",
        "addr": "Willemskade 27",
        "kind": "Bibliotheek",
        "address": "Willemskade 27",
        "text": "Museum De 5000 Morgen sloot in 2015 en ging op in de bibliotheek. Een deel van de collectie is hier te zien, met een replica van de kano van Pesse, de oudste boot ter wereld, die een paar kilometer verderop is gevonden.",
    },
    {
        "name": "Velden van Nevel",
        "osm": ["Velden van Nevel", "Velden van nevel", "De Lichtnaald"],
        "kind": "Kunstwerk · 2007",
        "text": "Een reeks beelden van zo'n tien meter hoog langs de A28 en de A37, van Rudi van de Wint (1942-2006). Zijn zonen hebben ze na zijn dood geplaatst. De titel verwijst naar het volksgeloof in nevelheksen, de witte wieven die rond Hoogeveen over de velden zouden spoken.",
    },
    # Infrastructure.
    {
        "name": "Vliegveld Hoogeveen",
        "osm": ["Vliegveld Hoogeveen"],
        "point": True,
        "kind": "Vliegveld · EHHO",
        "note": "Grasbaan van ruim een kilometer",
        "web": "https://ehho.nl",
        "text": "Vliegveld met één grasbaan van ruim een kilometer, de 09-27. Het was eerst bedoeld voor de oudere propellervliegtuigen van Fokker, dat een fabriek in Hoogeveen heeft. Nu zitten er vliegclubs, een zweefvliegclub, een paraclub en vliegscholen. Op het veld staat een oude Fokker-hangar uit 1936, die van vliegveld Ypenburg komt en hier begin deze eeuw weer is opgebouwd. Elk jaar is er Wings and Wheels, met klassieke vliegtuigen en auto's.",
    },
    {
        "name": "A28",
        "osm": ["A28"],
        "point": True,
        "kind": "Snelweg",
        "text": "De snelweg van Utrecht naar Groningen. Het tracé tussen knooppunt Lankhorst en Hoogeveen werd in 1971 vastgesteld en het eerste stuk ging in 1976 open, toen nog als Rijksweg 37. Sinds het Rijkswegenplan van 1986 heet de hele route A28. Hoogeveen heeft twee afritten: 26 Hoogeveen en 27 Fluitenberg.",
    },
    {
        "name": "A37",
        "osm": ["A37"],
        "point": True,
        "kind": "Snelweg",
        "text": "De snelweg naar Emmen en Duitsland, 42 kilometer lang en helemaal in Drenthe. Het was de tweebaans N37, tot het stuk van Hoogeveen tot Holsloot in 2003 als A37 openging; in oktober 2007 volgde de rest tot de grens. Daar gaat hij verder als de B402. De hele weg is deel van de E233.",
    },
    {
        "name": "Knooppunt Hoogeveen",
        "osm": ["Knooppunt Hoogeveen"],
        "point": True,
        "kind": "Knooppunt · 1981",
        "text": "Hier komen de A28, de A37 en de N48 bij elkaar. Het knooppunt was in 1981 klaar en maakte van Hoogeveen een kruispunt van het noorden. In de tweede helft van de jaren zestig was Hoogeveen de snelst groeiende gemeente van Nederland; er werd gerekend op 100.000 inwoners rond 1990. Het bleef bij zo'n 45.000.",
    },
    {
        "name": "Hoogeveensche Vaart",
        "osm": ["Hoogeveensche Vaart"],
        "point": True,
        "kind": "Kanaal · 17e eeuw",
        "text": "Het kanaal waarover de turf naar Meppel ging. Het stuk van Echten naar Meppel, de Nieuwe Grift, was in 1625 klaar. Van de vaart gingen om de 160 meter wijken het veen in: zo ver kon een arbeider met een kruiwagen turf lopen. In de tweede helft van de 19e eeuw was Hoogeveen de derde schippersplaats van het land en lagen er 's winters honderden schepen. De Compagnie verkocht de vaart in 1851 aan de Drentse Kanaal Maatschappij. Tussen 1964 en 1988 is hij geschikt gemaakt voor schepen van 300 ton en om de bebouwde kom heen gelegd; de vaarten in het centrum waren toen al gedempt. Sinds 1987 is de provincie de eigenaar.",
    },
    {
        "name": "Nieuwebrugsluis",
        "osm": ["Nieuwebrugsluis"],
        "point": True,
        "kind": "Sluis",
        "text": "Een van de drie sluizen in de Hoogeveensche Vaart, met de Rogatsluis en de Ossesluis. De kolk is 65 meter lang en 7,5 meter breed. Sinds 18 juni 2009 gaat de bediening op afstand, vanuit de bedienpost bij deze sluis.",
    },
    {
        "name": "Oude Diep",
        "osm": ["Oude Diep"],
        "point": True,
        "kind": "Beek",
        "text": "Een beekje dat bij Mantinge ontspringt en voorbij Hoogeveen in de Hoogeveensche Vaart uitkomt. De beek was rechtgetrokken; de laatste decennia heeft hij zijn bochten teruggekregen en in 2009 is er een rietveld van 7 hectare aangelegd dat het water zuivert. Bij onderzoek in het beekdal zijn resten uit de bronstijd gevonden.",
    },
    {
        "name": "Monument voor Albert Albino",
        "latlon": (52.73332, 6.47305),
        "kind": "Oorlogsmonument",
        "text": "Een propellerblad van een Lockheed P-38 Lightning. Op de middag van 29 november 1943 begeleidde tweede luitenant Albert A. Albino, 24 jaar, bommenwerpers naar Bremen. Op de terugweg schoot een Duitse jager hem neer. Het toestel sloeg naast het stationsgebouw in het spoor en verdween in de grond. Pas in maart 1978, bij de aanleg van de voetgangerstunnel, kwamen een propeller en de piloot zelf boven. Hij is begraven in Portland, Oregon, waar hij vandaan kwam.",
    },
    # Companies.
    {
        "name": "DOC Kaas",
        "osm": ["DOC Kaas"],
        "largest": True,
        "kind": "Kaasfabriek · {year}",
        "text": "De kaasfabriek op het Zuivelpark, de opvolger van de Coöperatieve Zuivelfabriek die Hoogeveen al rond 1900 had. De fabriek is gebouwd om zonder leidingwater, grondwater of oppervlaktewater te draaien: het proceswater komt uit de wei die wordt ingedampt. Uit 1000 liter melk komt zo'n 110 kilo kaas. Per jaar verlaat ongeveer 25 miljoen kilo gerijpte kaas het pakhuis. Het hoofdgebouw is {h} meter hoog en beslaat {area} vierkante meter.",
    },
    {
        "name": "DOC Kaas Alteveerstraat",
        "osm": ["DOC Kaas locatie Alteveerstraat"],
        "point": True,
        "kind": "Zuivelfabriek",
        "text": "De oude zuivelfabriek van Hoogeveen. De fabriekspijp is een provinciaal monument. Voor deze fabriek begon Drenthina in 1925 blikken te maken, en zo kreeg Hoogeveen ook een blikfabriek.",
    },
    {
        "name": "Trivium Packaging",
        "osm": ["Trivium Packaging"],
        "kind": "Blikfabriek",
        "text": "In 1925 begon metaalfabriek Drenthina blikken te maken voor de zuivelfabriek. Daarna kwam conservenfabriek Lukas Aardenburg, die er bonen in deed. Drenthina ging op in Thomassen & Drijver-Verblifa. Dit gebouw is uit {year}.",
    },
    {
        "name": "Schietsportvereniging Lucas Aardenburg",
        "osm": ["Schietsport Vereniging Lucas Aardenburg"],
        "kind": "Schietvereniging",
        "text": "De vereniging draagt nog de naam van de Kennemer Conservenfabriek Lukas Aardenburg, die zich tussen 1920 en 1930 in Hoogeveen vestigde om de oogst uit de streek in te blikken. Later maakte de fabriek diepvries van Iglo, als onderdeel van Unilever.",
    },
    {
        "name": "Fokker Aerostructures",
        "osm": ["Fokker Aerostructures"],
        "kind": "Vliegtuigbouw",
        "text": "Begonnen op 18 september 1961 als NV Lichtwerk. In 1969 werd het onderdeel van Fokker, en vanaf de jaren zeventig heette het Fokker Bedrijf Hoogeveen. Er werden vliegtuigonderdelen en shelters gebouwd.",
    },
    {
        "name": "TVM Verzekeringen",
        "osm": ["TVM Verzekeringen"],
        "kind": "Hoofdkantoor · {year}",
        "text": "Verzekeraar voor het wegtransport, en jarenlang naamgever van een van de grote Nederlandse wielerploegen, met renners als Jeroen Blijlevens, Gert-Jan Theunisse, Peter Van Petegem, Bart Voskamp en Servais Knaven.",
    },
    {
        "name": "Asielzoekerscentrum (oude gevangenis De Grittenborgh)",
        "osm": ["Asielzoekerscentrum Hoogeveen"],
        "kind": "Voormalige gevangenis",
        "text": "Dit was gevangenis De Grittenborgh, bij de sluiting een van de modernste van het land en toen 25 jaar oud. Op 3 oktober 2013 gingen de laatste zes gedetineerden naar Veenhuizen: staatssecretaris Teeven sloot negentien gevangenissen om te bezuinigen. In de jaren negentig ontsnapten hier in een paar jaar zestien langgestraften. Bij de uitbraak van 6 januari 1993 staken drie gedetineerden bewaarders neer en kwamen ze buiten met een mes op de keel van een bewaarster. Nu is het een asielzoekerscentrum.",
    },
    # Sport.
    {
        "name": "vv Hoogeveen",
        "osm": ["Voetbalvereniging Hoogeveen"],
        "point": True,
        "kind": "Voetbalclub · 1930",
        "text": "Opgericht op 29 april 1930, blauw-wit, zo'n 650 leden. In 1978 werd de club kampioen van de Hoofdklasse en daarna algeheel landskampioen bij de zondagamateurs. De noordelijke districtsbeker kwam in 1980 en 2008 naar Hoogeveen. Op het hoofdveld in het Bentinckspark kunnen ongeveer 6000 mensen kijken.",
    },
    {
        "name": "HZVV",
        "osm": ["Hoogeveense Zaterdag Voetbal Vereniging"],
        "point": True,
        "kind": "Voetbalclub · 1929",
        "text": "De Hoogeveense Zaterdag Voetbal Vereniging, opgericht op 5 december 1929. De zaterdagclub speelt net als vv Hoogeveen op het Bentinckspark en is een paar maanden ouder dan de buurman.",
    },
    # Tall and old buildings; the numbers come from the BAG and 3D BAG.
    {
        "name": "De Arend",
        "osm": ["De Arend"],
        "kind": "Flat · {year}",
        "text": "Woonflat uit {year}, {h} meter hoog en daarmee de hoogste woontoren van Hoogeveen. Rond 2010 werd over sloop gesproken.",
    },
    {"name": "Flat Jennerstraat", "addr": "Jennerstraat 130", "kind": "Flat · {year}", "address": "Jennerstraat", "text": "Flat uit {year}, {h} meter hoog."},
    {"name": "Flat Hoofdstraat 159", "addr": "Hoofdstraat 159-63", "kind": "Flat · {year}", "address": "Hoofdstraat 159", "text": "Flat uit {year}, {h} meter hoog."},
    {"name": "Huize Venendal", "addr": "Hoofdstraat 9", "kind": "Herenhuis · {year}", "address": "Hoofdstraat 9", "text": "Huize Venendal staat in de BAG met bouwjaar {year}; alleen het Schippershuus en de Grote Kerk zijn daar ouder. Tot 2015 zat hier Museum De 5000 Morgen, het museum over de veenmaatschappij waar Hoogeveen mee begon."},
    {"name": "Schutstraat 8", "addr": "Schutstraat 8a", "kind": f"{RM} · {{year}}", "address": "Schutstraat 8", "text": "Winkelwoning met een gevelsteen uit {year}, bij Het Kruis. Het was lang een drogisterij."},
    {"name": "Het Haagje 38 en 40", "addr": "Het Haagje 38", "kind": "Pand · {year}", "address": "Het Haagje 38 en 40", "text": "Twee panden die volgens de BAG uit {year} zijn, aan de vaart die van Het Kruis naar het Groote Meer liep."},
    {
        "name": "Herman Bavinckhuis",
        "osm": ["Herman Bavinckhuis"],
        "kind": "Kerkelijk centrum",
        "text": "Genoemd naar Herman Bavinck (1854-1921), de theoloog die in Hoogeveen is geboren en het gereformeerde denken van zijn tijd mee heeft gevormd.",
    },
]

# Lattice masts: {"latlon": (lat, lon), "height": m, "base": m, "tip": m}.
MASTS = []

# Series in the viewer's panel: a walk past these places, starting with the one nearest the village.
CATEGORIES = []

# What the viewer shows when a place is clicked; the rest of an entry is for the render.
INFO = ("note", "kind", "text", "address", "web", "more", "photo", "stop")

EXTRA = RAW.parent / "extra"

SKIP_NAMES = {"Vliegveld Hoogeveen", "1", "2", "Hoogeveen"}

# What is known about a neighbourhood beyond the years its houses were built.
WIJKEN = {
    "Oranjebuurt": "De eerste huizen kwamen rond 1885 aan de Oranjestraat. Vanaf 1914 liet Bouwvereniging Hoogeveen er arbeiderswoningen bouwen naar ontwerp van gemeentearchitect J. Carmiggelt: 40 in 1915, daarna een deel van een plan voor nog 142.",
    "Venesluis": "Een wijk uit de jaren zestig, de tijd dat Philips, Fokker en Standard Electric naar Hoogeveen kwamen en de gemeente de snelst groeiende van het land was.",
    "Erflanden": "De jongste grote wijk, tegen de A28 aan, met een geluidswal ertussen.",
    "De Weide": "Gebouwd nadat een grenswijziging uitbreiding naar het westen mogelijk maakte, aan de overkant van de A28.",
    "Wolfsbos": "De wijk ligt tegen de A37, achter geluidsschermen.",
    "Zuid": "Wijk met portiekflats. Het Zuiderpark is in 2009 vernieuwd.",
}


def osm_index():
    """Named OpenStreetMap things inside the map, from the three downloads: {name: [(x, y, is_building, tags)]}."""
    out, seen = {}, set()
    files = [RAW / "osm_landmarks.json", EXTRA / "osm_pois.json", EXTRA / "osm_named.json"]
    if not files[1].exists():
        files[1] = RAW / "osm_pois.json"
    for path in files:
        if not path.exists():
            continue
        for e in json.loads(path.read_text())["elements"]:
            t = e.get("tags", {})
            key = (e["type"], e["id"])
            if key in seen:
                continue
            seen.add(key)
            if "lat" in e:
                pts = [(e["lat"], e["lon"])]
            elif "center" in e:
                pts = [(e["center"]["lat"], e["center"]["lon"])]
            else:
                pts = [(g["lat"], g["lon"]) for g in e.get("geometry", [])]
                pts += [(g["lat"], g["lon"]) for m in e.get("members", []) for g in m.get("geometry", [])]
            if not pts:
                continue
            x, y = np.mean([wgs_to_rd(lat, lon) for lat, lon in pts], axis=0)
            if not (XMIN < x < XMAX and YMIN < y < YMAX):
                continue
            for name in {t.get("name"), t.get("ref") if t.get("highway") else None} - {None}:
                out.setdefault(name, []).append((float(x), float(y), e["type"] != "node" and "building" in t, t))
    return out


class Bag:
    """The 3D BAG buildings: which one stands at a point, with its year, height and footprint."""

    def __init__(self):
        self.ids, self.rings, self.attrs = [], [], []
        for line in open(RAW / "3dbag.jsonl"):
            f = json.loads(line)
            t = f["transform"]
            V = np.array(f["vertices"], dtype=np.float64) * t["scale"] + t["translate"]
            co = f["CityObjects"][f["id"]]
            self.ids.append(f["id"])
            self.rings.append(V[co["geometry"][0]["boundaries"][0][0]][:, :2])
            self.attrs.append(co["attributes"])
        self.mid = np.array([r.mean(0) for r in self.rings])
        self.height = np.array([(a.get("b3_h_dak_max") or 0) - (a.get("b3_h_maaiveld") or 0) for a in self.attrs])
        self.year = np.array([a.get("oorspronkelijkbouwjaar") or 0 for a in self.attrs])
        self.area = np.array([a.get("b3_opp_grond") or 0 for a in self.attrs])
        # The tallest buildings, without the chimneys and masts.
        self.tallest = [i for i in np.argsort(-self.height) if self.area[i] >= 150][:10]

    def at(self, x, y, reach=20):
        """Index of the building under a point, or of the nearest one within reach; None when there is none."""
        d = np.hypot(self.mid[:, 0] - x, self.mid[:, 1] - y)
        for i in np.argsort(d)[:40]:
            if d[i] > 300:
                break
            if point_in_ring(x, y, self.rings[i]):
                return int(i)
        i = int(d.argmin())
        return i if d[i] <= reach else None

    def facts(self, i):
        rank = self.tallest.index(i) + 1 if i in self.tallest else None
        return {"year": int(self.year[i]), "h": round(float(self.height[i])), "area": f"{round(float(self.area[i]), -2):,.0f}".replace(",", "."), "rank": rank}


def rank_text(rank):
    return "Het hoogste gebouw van Hoogeveen." if rank == 1 else f"Het op {['', 'een', 'twee', 'drie', 'vier', 'vijf', 'zes', 'zeven', 'acht', 'negen'][rank - 1]} na hoogste gebouw van Hoogeveen."


def fill(entry, facts):
    """Put a building's year, height and footprint into an entry's {year}, {h} and {area}, and add its rank."""
    for k in ("kind", "text", "note"):
        if k in entry:
            for name, v in facts.items():
                entry[k] = entry[k].replace("{" + name + "}", str(v))
    if facts["rank"] and "hoogste" not in entry.get("text", ""):
        entry["text"] = f"{entry.get('text', '')} {rank_text(facts['rank'])}".strip()


SPORTS = {"soccer": "voetbal", "tennis": "tennis", "padel": "padel", "cycling": "wielrennen", "billiards": "biljart", "shooting": "schietsport",
          "swimming": "zwemmen", "hockey": "hockey", "korfball": "korfbal", "athletics": "atletiek", "parachuting": "parachutespringen"}
CHURCHES = {"baptist": "Baptistenkerk", "roman_catholic": "Rooms-katholieke kerk", "jehovahs_witness": "Koninkrijkszaal van Jehovah's Getuigen",
            "reformed": "Gereformeerde kerk", "evangelical": "Evangelische kerk"}


def describe(t):
    """What OpenStreetMap says a named place is, as (kind, sentence), or None when it says too little to tell."""
    if t.get("amenity") in ("school", "college"):
        what = t.get("official_name") or t.get("description")
        if not what:
            return None
        level = {"0;1": "Basisschool", "2": "Middelbare school", "2;3": "Middelbare school", "3": "Mbo-school"}.get(t.get("isced:level"), "School")
        said = f"{what[0].upper()}{what[1:]}" if what else level
        if t.get("official_name") and t.get("description"):
            said += f": {t['description']}"
        if t.get("operator"):
            said += f", van {t['operator']}"
        if t.get("old_name"):
            said += f". Heette eerder {' en '.join(t['old_name'].split(';'))}"
        return "School", said + "."
    if t.get("amenity") == "place_of_worship" and (t.get("denomination") in CHURCHES or t.get("operator")):
        said = "Moskee" if t.get("religion") == "muslim" else CHURCHES.get(t.get("denomination"), "Kerk")
        if t.get("operator"):
            said += f" van de {' en de '.join(t['operator'].split(';'))}"
        return ("Moskee" if t.get("religion") == "muslim" else "Kerk"), said + "."
    sports = [SPORTS[s] for s in t.get("sport", "").split(";") if s in SPORTS]
    if sports and (t.get("club") or t.get("community_centre") == "club_home"):
        return "Sportclub", f"Clubgebouw voor {' en '.join(sports)}."
    if t.get("social_facility") == "nursing_home" and t.get("operator"):
        return "Verpleeghuis", f"Verpleeghuis van {t['operator']}."
    if t.get("social_facility") == "assisted_living" and t.get("operator"):
        return "Woonzorg", f"Woonzorgcomplex van {t['operator']}."
    return None


def places(bag):
    """BUILDINGS and POINTS: the curated places, then the named buildings and the neighbourhoods of OpenStreetMap,
    each with the few facts the data gives. A place without a story is not on the map."""
    osm = osm_index()
    buildings, points, used, names = [], [], set(), set()

    def add_building(c, i):
        if i in used:
            return False
        used.add(i)
        fill(c, bag.facts(i))
        buildings.append({**c, "pid": bag.ids[i]})
        return True

    for c in CURATED:
        c = dict(c)
        hints = c.pop("osm", [])
        names.update(hints)
        names.add(c["name"])
        as_point = c.pop("point", False)
        if "addr" in c:
            c["at"] = address_xy(c.pop("addr"))
        elif "latlon" in c:
            c["at"] = wgs_to_rd(*c.pop("latlon"))
            as_point = True
        else:
            found = [p for n in hints for p in osm.get(n, [])]
            assert found, f"{c['name']}: not in OpenStreetMap"
            # Of a road or a stream in many pieces, the piece nearest the middle of the map.
            best = min(found, key=lambda p: (not p[2] and not as_point, np.hypot(p[0] - CX, p[1] - CY)))
            c["at"] = best[:2]
            c["wikidata"] = next((p[3]["wikidata"] for p in found if "wikidata" in p[3]), None)
        i = None if as_point else bag.at(*c["at"])
        if c.pop("largest", False):  # a factory of many parts: the biggest building around
            near = np.hypot(bag.mid[:, 0] - c["at"][0], bag.mid[:, 1] - c["at"][1]) < 250
            i = int(np.argmax(np.where(near, bag.area, 0)))
        if i is None or not add_building(c, i):
            assert as_point or i is None, f"{c['name']}: its building is already taken"
            assert not any("{" in c.get(k, "") for k in ("kind", "text")), f"{c['name']}: no building to take its facts from"
            points.append(c)

    # Named buildings that OpenStreetMap says enough about to tell what they are.
    for name, found in sorted(osm.items()):
        if name in names or name in SKIP_NAMES:
            continue
        for x, y, is_building, t in found:
            said = describe(t)
            i = bag.at(x, y, reach=0) if said else None
            if i is None or bag.year[i] < 1000:
                continue
            kind, text = said
            f = bag.facts(i)
            entry = {"name": t.get("official_name", name) if kind == "School" else name, "at": (x, y), "kind": kind,
                     "text": f"{text} Het gebouw is uit {f['year']}.", "wikidata": t.get("wikidata")}
            street, number = t.get("addr:street"), t.get("addr:housenumber")
            if street and number:
                entry["address"] = f"{street} {number}"
            web = t.get("website") or t.get("contact:website")
            if web and web.startswith("http"):
                entry["web"] = web
            if add_building(entry, i):
                names.add(name)
                break

    # The tallest buildings that are not on the map yet.
    for rank, i in enumerate(bag.tallest, 1):
        if i in used:
            continue
        f = bag.facts(i)
        d = np.hypot(ADDRESS_XY[:, 0] - bag.mid[i, 0], ADDRESS_XY[:, 1] - bag.mid[i, 1])
        street = re.sub(r"\s+\d.*$", "", ADDRESS_LABELS[int(d.argmin())])
        add_building({"name": f"Hoogbouw {street}" if street not in names else f"Hoogbouw {street} ({rank})", "at": tuple(bag.mid[i]), "kind": f"Hoogbouw · {f['year']}",
                      "text": f"Gebouw uit {f['year']}, {f['h']} meter hoog."}, i)
        names.add(street)

    # Neighbourhoods there is something to tell about.
    for name, text in WIJKEN.items():
        if name not in osm:
            continue
        x, y, _, t = osm[name][0]
        near = (np.hypot(bag.mid[:, 0] - x, bag.mid[:, 1] - y) < 350) & (bag.year > 1000) & (bag.area > 30)
        lo, hi = np.percentile(bag.year[near], [25, 75]).astype(int)
        points.append({"name": name, "at": (x, y), "kind": "Wijk" if t.get("place") != "quarter" else "Buurt",
                       "text": f"{text} De helft van de gebouwen binnen 350 meter is gebouwd tussen {lo} en {hi}."})
    return buildings, points


def point_in_ring(x, y, ring):
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, np.roll(ring, -1, axis=0)):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


_A = json.loads((RAW / "addresses.json").read_text())
ADDRESS_XY = np.array([(x, y) for x, y, _ in _A])
ADDRESS_LABELS = [label.split(", ")[0] for _, _, label in _A]
ADDRESSES = None


def address_xy(name):
    """RD position of a BAG address such as "Hoofdstraat 3" (data/raw/addresses.json, from scripts/fetch.py)."""
    global ADDRESSES
    if ADDRESSES is None:
        ADDRESSES = {label.replace(", Hoogeveen", "").split(", ")[0].lower(): (x, y) for x, y, label in json.loads((RAW / "addresses.json").read_text())}
    return ADDRESSES[name.lower()]


def main():
    bag = Bag()
    out = {"buildings": {}, "places": [], "masts": [], "hotspots": [], "categories": []}
    buildings, points = places(bag)
    # Photos from Wikimedia Commons, chosen by eye from what scripts/fetch_photos.py found: photo_chosen.json says
    # which file goes with which place, photo_credits.json who made it and under which licence.
    here = ROOT / "scripts"
    chosen = json.loads((here / "photo_chosen.json").read_text()) if (here / "photo_chosen.json").exists() else {}
    credits = json.loads((here / "photo_credits.json").read_text()) if (here / "photo_credits.json").exists() else {}
    for e in buildings + points:
        if e["name"] in chosen and chosen[e["name"]] in credits:
            e["photo"] = (chosen[e["name"]], credits[chosen[e["name"]]])
    for b in buildings:
        x, y = b["at"]
        pid = b["pid"]
        i = bag.ids.index(pid)
        height = b["tower"]["top"] if "tower" in b else b["turret"]["top"] if "turret" in b else float(bag.height[i])
        out["places"].append({"name": b["name"], "note": b.get("note"), "x": x, "y": y, "h": round(height, 1)})
        if "style" in b:
            out["buildings"][pid] = {k: v for k, v in b.items() if k not in ("at", "pid", "name", *INFO)}
        out["hotspots"].append({"name": b["name"], "pid": pid, "x": x, "y": y, **{k: b[k] for k in INFO if k in b}})
    for p in points:
        x, y = p["at"]
        out["places"].append({"name": p["name"], "note": p.get("note"), "x": x, "y": y, "h": 2.0})
        out["hotspots"].append({"name": p["name"], "x": x, "y": y, **{k: p[k] for k in INFO if k in p}})
    for m in MASTS:
        x, y = wgs_to_rd(*m["latlon"])
        out["masts"].append({**{k: v for k, v in m.items() if k != "latlon"}, "x": x, "y": y})
    names = {h["name"] for h in out["hotspots"]}
    for key, label, members in CATEGORIES:
        missing = [n for n in members if n not in names]
        assert not missing, f"{label}: no hotspot for {missing}"
        out["categories"].append({"key": key, "label": label, "names": members})
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "landmarks.json").write_text(json.dumps(out, indent=1))
    # The places to look for photos of (scripts/fetch_photos.py), in the repository because that runs on GitHub.
    wanted = [{"name": e["name"], "latlon": [round(v, 5) for v in rd_to_wgs(*e["at"])], "wikidata": e.get("wikidata"), "kind": e.get("kind")}
              for e in buildings + points]
    (ROOT / "scripts" / "photo_places.json").write_text(json.dumps(wanted, ensure_ascii=False, indent=0))
    print(f"{len(buildings)} buildings and {len(points)} points with a story")


if __name__ == "__main__":
    main()
