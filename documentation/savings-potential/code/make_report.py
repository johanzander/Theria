"""Assemble the self-contained HTML report from figs/*.png and figs/numbers.json."""
import base64, json, pickle
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
FIG = HERE / "figs"
N = json.load(open(FIG / "numbers.json"))
M = N["methods"]
PW = N["per_winter"]
V = N["validation"]
dp = pickle.load(open(HERE / "results" / "dpcheck.pkl", "rb"))
dp_gap = max((x["dp"] - x["lp"]) / x["base"] * 100 for x in dp)
robust = pickle.load(open(HERE / "results" / "robust.pkl", "rb"))
methods_raw = pickle.load(open(HERE / "results" / "methods.pkl", "rb"))
window_raw = pickle.load(open(HERE / "results" / "window.pkl", "rb"))
dhw = pickle.load(open(HERE / "results" / "dhw_sens.pkl", "rb"))
DHW_CAP = 5.0 * 0.86
dp_idag = max((x["dp"] - x["lp"]) / x["base"] * 100 for x in dp if x["scen"] == "Idag")
dp_s3 = max((x["dp"] - x["lp"]) / x["base"] * 100 for x in dp if x["scen"] == "Utvidgat")
dp_never_worse = all(x["dp"] >= x["lp"] - 1e-6 for x in dp)

def horizon(sn, key):
    return 100 * sum(methods_raw[(sn, wi)][key]["saving_neutral"] for wi in range(4)) / sum(methods_raw[(sn, wi)][key]["base_cost"] for wi in range(4))

def whole_winter(sn):
    return 100 * sum(window_raw[(wi, sn, 0)]["perfekt"]["saving_neutral"] for wi in range(4)) / sum(window_raw[(wi, sn, 0)]["perfekt"]["base_cost"] for wi in range(4))

def dhw_pct(sn, mode, cap):
    return 100 * sum(dhw[(sn, wi, cap)][mode]["saving_neutral"] for wi in range(4)) / sum(dhw[(sn, wi, cap)]["base_cost"] for wi in range(4))

def dhw_sek(sn, mode, cap):
    return sum(dhw[(sn, wi, cap)][mode]["saving_neutral"] for wi in range(4)) / 4

def img(name, alt):
    b = base64.b64encode((FIG / name).read_bytes()).decode()
    return f'<figure><img alt="{alt}" src="data:image/png;base64,{b}"><figcaption>{alt}</figcaption></figure>'

def m(sn, label_start):
    for k, v in M[sn].items():
        if k.startswith(label_start):
            return v
    raise KeyError(label_start)

def pl(sn):
    return np.mean([100 * sum(robust[(wi, sn, "placebo", s)]["saving_neutral"] for wi in range(4)) /
                    sum(robust[(wi, sn, "placebo", s)]["base_cost"] for wi in range(4)) for s in (1, 2, 3)])

S = {k: {"max": m(k, "Tak"), "day": m(k, "Praktiskt"), "md": m(k, "Regel, flerdygns"), "n": m(k, "Regel: N")}
     for k in ("S1 Idag", "S3 Nedsänkning", "S4 Stort spann")}
base_avg = np.mean(PW["S1 Idag"]["base_cost"])

def row(label, sn):
    d = S[sn]
    return (f"<tr><td>{label}</td><td>{d['max']['pct']:.1f} %<br><small>{d['max']['sek_per_winter']:.0f} kr/vinter</small></td>"
            f"<td>{d['day']['pct']:.1f} %<br><small>{d['day']['sek_per_winter']:.0f} kr/vinter</small></td>"
            f"<td>{d['md']['pct']:.1f} %</td><td>{d['n']['pct']:.1f} %</td></tr>")

winter_rows = ""
for wi, lab in enumerate(["2022/23", "2023/24", "2024/25", "2025/26"]):
    winter_rows += (f"<tr><td>{lab}</td><td>{PW['S1 Idag']['base_cost'][wi]:.0f}</td>"
                    + "".join(f"<td>{PW[sn]['LP perfekt'][wi]:.1f} / {PW[sn]['LP dygnet före'][wi]:.1f}</td>"
                              for sn in ("S1 Idag", "S3 Nedsänkning", "S4 Stort spann")) + "</tr>")

html = f"""<!doctype html><html lang="sv"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Besparingspotential</title>
<style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e6e5e1;--accent:#2a78d6;--card:#f4f3f0}}
@media (prefers-color-scheme: dark){{:root:not([data-theme=light]){{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#383835;--card:#242423}}}}
body{{background:var(--bg);color:var(--ink);font:16px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;margin:0}}
main{{max-width:980px;margin:0 auto;padding:24px 16px 80px}}
h1{{font-size:28px;margin:8px 0 4px}} h2{{margin-top:44px;border-top:1px solid var(--line);padding-top:20px}}
.lede{{color:var(--ink2);font-size:18px}}
table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:15px}} th,td{{border-bottom:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}}
th{{color:var(--ink2);font-weight:600}} small{{color:var(--ink2)}}
figure{{margin:20px 0;background:#fcfcfb;border:1px solid var(--line);border-radius:10px;padding:10px}} figure img{{max-width:100%;height:auto;display:block}}
figcaption{{color:#52514e;font-size:13px;margin-top:6px}}
.card{{background:var(--card);border-radius:10px;padding:14px 18px;margin:16px 0}}
.kpi{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:18px 0}}
.kpi div{{background:var(--card);border-radius:10px;padding:12px 16px}} .kpi b{{font-size:28px;display:block}} .kpi span{{color:var(--ink2);font-size:14px}}
code{{background:var(--card);padding:1px 5px;border-radius:4px}}
</style></head><body><main>
<h1>Hur mycket kan uppvärmningen flyttas till billiga timmar?</h1>
<p class="lede">Maximal teoretisk besparing för Johans hus, beräknad på fyra historiska vintrar med verkliga spotpriser och utetemperaturer. Alla siffror gäller <b>samma komfort</b>: varje vecka ska rummet ha minst samma medeltemperatur som idag.</p>

<div class="card"><b>Version 2 (16 okt 2026).</b> Första versionen räknade med kompressorns tak 3,6 kW, men det gäller AirX 9. Huset har en AirX 170 (4,4–5,8 kW elinsats vid maxvarv, datablad i <code>documentation/private/IVT</code>), och laddningen av mätdata kastade bort timmar över 4 kWh som spikar, vilket tog bort riktiga kalla timmar. Nu är taket 5,0 kW, spikgränsen 5,0 kWh/h och elmodellen anpassad om (<code>−0,95 − 0,238·T<sub>ute</sub> + 0,256·börvärde</code>). Scenarier med flöde över 40 °C är borttagna: golvets 40 °C är ett hårt krav. S3 och S4 är nu nedsänkning (börvärde 8–20) respektive stort spann (börvärde 8–26), båda med flöde ≤ 40 °C.</div>

<div class="kpi">
<div><b>{S['S1 Idag']['max']['pct']:.1f} %</b><span>teoretiskt tak med dagens spelrum (börvärde 16–20 °C, flöde ≤ 40 °C, golvets hårda gräns): {S['S1 Idag']['max']['sek_per_winter']:.0f} kr per vinter</span></div>
<div><b>{S['S1 Idag']['day']['pct']:.1f} %</b><span>vad en praktisk optimering ett dygn framåt klarar av idag ({S['S1 Idag']['day']['sek_per_winter']:.0f} kr)</span></div>
<div><b>{S['S3 Nedsänkning']['max']['pct']:.1f} %</b><span>tak om börvärdet får sänkas ned till 8 °C (S3, flöde ≤ 40 °C): {S['S3 Nedsänkning']['max']['sek_per_winter']:.0f} kr per vinter, praktiskt {S['S3 Nedsänkning']['day']['pct']:.1f} %</span></div>
<div><b>{S['S4 Stort spann']['max']['pct']:.1f} %</b><span>tak om börvärdet får gå 8–26 °C (S4, flöde ≤ 40 °C): {S['S4 Stort spann']['max']['sek_per_winter']:.0f} kr per vinter</span></div>
</div>

<h2>Sammanfattning</h2>
<ul>
<li><b>Det som begränsar är reglaget och husets tröghet, inte komfortbandet.</b> Att tillåta 16–23 °C i stället för 18–21 °C ger exakt samma besparing så länge börvärdet får röra sig lika lite som idag (S1 = S2). Bandet börjar spela roll först när reglaget är större.</li>
<li><b>Huset dämpar dygnssvängningar kraftigt</b> (tidskonstant ≈ 5 dygn: bara ca 3 % av en dygnssvängning i börvärdet når rummet), så dagsrankade regler tar lite. Men <i>taket</i> beror knappt på tröghet: ett hus som är 16 gånger snabbare eller 4 gånger långsammare får 7,3–7,9 % med dagens spelrum. Det som begränsar är hur mycket energi pumpen får flytta, inte hur långsamt golvet svarar.</li>
<li><b>Dagens regel (”N billigaste / dyraste timmar”) tar bara en liten del.</b> {S['S1 Idag']['n']['pct']:.1f} % med dagens spelrum mot taket {S['S1 Idag']['max']['pct']:.1f} %. Regler som tittar på flera dygns pris (kräver prisprognos) tar {S['S1 Idag']['md']['pct']:.1f} %, och en riktig optimering (LP/DP) {S['S1 Idag']['day']['pct']:.1f} % redan med bara dygnet före känt.</li>
<li><b>Prognoshorisonten är värd pengar:</b> från ett till fjorton dygns känt pris stiger den praktiska besparingen från {horizon('S1 Idag', 'LP dygnet före'):.1f} % till {horizon('S1 Idag', 'LP 14 dygn'):.1f} % (S1, taket är {S['S1 Idag']['max']['pct']:.1f} %), och med nedsänkning från {horizon('S3 Nedsänkning', 'LP dygnet före'):.1f} % till {horizon('S3 Nedsänkning', 'LP 14 dygn'):.1f} % (S3).</li>
<li><b>Din regel ”stäng pumpen de 6 dyraste timmarna, förvärm de 6 billigaste” sparar ingenting över en vinter</b> (−3 till +2,5 % per vinter, se avsnittet om enskilda dagar). Skälet är att kalla timmar redan ligger nära flödesgränsen 40 °C, så värmen går inte att ta igen, och att regeln kör samma ingrepp varje dygn oavsett prisskillnad.</li>
<li>Medelbesparing över de fyra vintrarna, mot en baslinjekostnad (spot, värme) på i snitt {base_avg:.0f} kr/vinter. Spridningen mellan vintrar är stor (se tabell): 2024/25 med volatila priser ger ungefär 1,5 gånger så mycket som 2023/24 och 2025/26 (taket 10,5 % mot 7,0 %).</li>
</ul>

<h2>Resultat per spelrum och metod</h2>
<table><tr><th>Spelrum</th><th>Teoretiskt tak<br><small>känd framtid, 95 % KI i diagram</small></th><th>Praktisk optimering<br><small>priser känns ett dygn framåt</small></th><th>Regel, flerdygnspris<br><small>kräver prognos</small></th><th>Regel, N billigaste/dyraste<br><small>bäst av rutnätet</small></th></tr>
{row("S1 Idag · band 18–21, börvärde 16–20, flöde ≤ 40", "S1 Idag")}
{row("S3 Nedsänkning · band 16–23, börvärde 8–20, flöde ≤ 40", "S3 Nedsänkning")}
{row("S4 Stort spann · band 16–23, börvärde 8–26, flöde ≤ 40", "S4 Stort spann")}
</table>
<p><small>Procent = andel av spotkostnaden för värme (pumpens kompressor + elpatron, utan varmvatten; varmvattnet står för 9–19 % av kompressorns el på vintern och ingår inte). Praktiskt för S3 och S4 varierar mycket mellan vintrar (konfidensintervallen i diagrammet är breda). Kronor är medel per vinter. S2 (band 16–23, reglage som idag) är identisk med S1.</small></p>
{img("03_metoder.png", "Besparing per metod och spelrum. Felstaplar: 95 % konfidensintervall, blockbootstrap över veckor.")}

<table><tr><th>Vinter</th><th>Baslinje (kr, spot)</th><th>S1 Idag<br><small>tak / praktiskt, %</small></th><th>S3 Nedsänkning<br><small>tak / praktiskt, %</small></th><th>S4 Stort spann<br><small>tak / praktiskt, %</small></th></tr>{winter_rows}</table>

<h2>Hur det gjordes</h2>
<div class="card"><b>Data.</b> 1 nov – 15 apr, vintrarna 2022/23–2025/26 (≈ 16 000 timmar). Spotpris SE4 från elprisetjustnu.se (korrelation med din prissensor 0,999 på de 9 700 timmar där båda finns). Utetemperatur från din sensor <code>sensor.outdoor</code>, luckor fyllda från Open-Meteo.</div>
<div class="card"><b>Hus och pump.</b> Theria:s egna kalibrerade modeller: rummets svar på flödestemperaturen (förstärkning 0,45 °C per °C, tidskonstant 5,2 dygn, golvvärme) och pumpens elförbrukning mot utetemp och börvärde (anpassad på 9 258 rena värmetimmar), inklusive kompressorns tak på 5,0 kW (AirX 170) och elpatronen över 44 °C. Husets okända tillskott (spis, människor, grannzoner) förankras så att fast börvärde 18 °C håller exakt 19,5 °C — varje metod bedöms mot den banan.</div>
<div class="card"><b>Samma komfort.</b> Rummet ska hålla sig i bandet hela tiden, och <i>varje vecka</i> ska medeltemperaturen vara minst baslinjens 19,5 °C. Utan veckokravet (hela vintern som en period) ger optimeringen skenbart {whole_winter('S1 Idag'):.0f} / {whole_winter('S3 Nedsänkning'):.0f} / {whole_winter('S4 Stort spann'):.0f} % (S1/S3/S4) genom att ligga på 16 °C under dyra veckor och 23 °C under billiga — det är komfortarbitrage, inte värmelagring. Figuren ”komfortkravets längd” visar hur resultatet beror på kravet. Eventuell komfortbrist hos en metod debiteras till säsongens 90:e prisprocentil, så ingen vinner på att vara kall.</div>
<div class="card"><b>Metoder.</b> (1) <i>Teoretiskt tak</i>: linjärprogram (LP) över hela vintern med känd framtid — konvext, alltså globalt optimum. (2) <i>Praktisk</i>: LP som planerar om varje dag kl 14 med priser till slutet av morgondagen och värderar kvarvarande värme till senaste veckans pris (bess-managers terminalvärde). (3) <i>Regler</i>: N billigaste/dyraste timmar per dygn (Theria idag), prisviktad inom dygnet, köp–sälj-par i bess-stil, samt en flerdygnsregel; varje regel trimmas till samma medeltemperatur och rutnätssöks (N, justering, vikt). Allt spelas upp i den icke-linjära simuleraren.</div>

<h2>Kontroller att modellen och optimeringen håller</h2>
<ul>
<li><b>Pumpmodellen mot mätdata:</b> R² = {V['r2']:.2f} på {V['days']} dygn, bias {V['bias_pct']:+.1f} %, medelfel {V['mae_kwh']:.1f} kWh/dygn.</li>
<li><b>LP mot DP (bess-manager-metoden):</b> en oberoende dynamisk programmering på 10 femdygnsfönster hamnar som mest {dp_idag:.1f} % av fönstrets kostnad över LP-optimum med dagens spelrum och {dp_s3:.1f} % med börvärde 8–20 (S3). LP är {"aldrig" if dp_never_worse else "inte alltid"} sämre. Gapet växer med spelrummet, vilket tyder på diskretiseringsfel i DP:ns rutnät (jag har inte provat ett finare rutnät); LP är konvext och därför ett exakt optimum.</li>
<li><b>Placebo:</b> en plan byggd på fel års priser men betald med rätt års priser ger {pl('S1 Idag'):.1f} / {pl('S3 Nedsänkning'):.1f} / {pl('S4 Stort spann'):.1f} % (S1/S3/S4). Det är den del som kommer av prisernas generella dygnsform; resten är verklig prisinformation.</li>
<li><b>Känslighet för golvets förstärkning:</b> 0,6–1,25 × skattad förstärkning ändrar besparingen mindre än 0,1 procentenhet. Besparingen sätts av hur mycket energi pumpen kan flytta (börvärdesspannet, kompressortaket) och av tidskonstanten, inte av hur starkt golvet svarar — det gör resultatet robust mot den osäkraste parametern.</li>
</ul>
{img("02_validering.png", f"Modellens dygnsförbrukning mot uppmätt, {V['days']} dygn.")}
{img("10_robusthet.png", "Känslighet: golvförstärkning, placebo med fel års priser, och vikten på lagrad värme vid planens slut (1,0 = beräknat värde).")}

<h2>Varför det är så lite att hämta idag: husets tröghet</h2>
{img("01_tröghet.png", "Golvvärmehuset (blå) följer börvärdet över dygn. En dygnssvängning dämpas ca 97 %.")}
{img("06_tröghet_värde.png", "Besparing mot husets tidskonstant. Taket är nästan oberoende av tröghet: S1 7,3–7,9 % och S4 13,6–15,7 % över tidskonstanter från 0,3 till 21 dygn. Den dagsbundna regeln (6 timmar, justering 2 °C) tar 0,7–1,3 % oavsett hus. Praktisk planering ett dygn framåt faller i ett mycket trögt hus (S3: 1,1 % vid 21 dygn).")}

<h2>Spelrum: band × reglage</h2>
{img("04_band_x_reglage.png", "Teoretisk besparing vid perfekt framsyn, per tillåtet rumsband och börvärdets spelrum, flöde ≤ 40 °C.")}
<p>Rad för rad ger bandet nästan ingenting: med dagens börvärde ±2 °C blir det 7,3 % redan vid ±0,5 °C band, och band bredare än ±1 °C ändrar knappt något för någon reglagestorlek (högst 0,2 procentenheter). Kolumn för kolumn växer besparingen med reglaget, men med avtagande effekt: ±1 °C ger 4,3 %, ±2 °C 7,4 %, ±4 °C 10,9 %, ±8 °C 13,7 % och ±12 °C 14,4 %. Flödesgränsen 40 °C (shuntarna) är ett hårt krav, och den gör att uppåtrörelsen snart tar slut; nästan hela vinsten från större spann kommer från att kunna sänka börvärdet djupt under dyra timmar.</p>

<h2>Din regel på en enskild dag med stor prisspridning</h2>
<p>Regeln ”stäng pumpen de 6 dyraste timmarna, förvärm de 6 billigaste” (börvärde −3 eller −8 °C under dyra timmar, +8 °C under billiga; flöde ≤ 40 °C; kompressor 5,0 kW). Två verkliga dagar bland dem med störst skillnad mellan dyraste och billigaste timmarna. Besparingen är netto efter att rummets underskott i slutet värmts upp till 90:e percentilens pris.</p>
<table><tr><th>Dag</th><th>Spridning dyra–billiga</th><th>Sänkning dyra timmar</th><th>Besparing</th><th>Dygnskostnad</th><th>Rum lägst</th></tr>
<tr><td>Mild, +1,6 °C ute</td><td>5,27 kr</td><td>−3 °C</td><td>17,5 kr (6 %)</td><td>279 kr</td><td>19,4 °C</td></tr>
<tr><td>Mild, +1,6 °C ute</td><td>5,27 kr</td><td>−8 °C</td><td>50,5 kr (18 %)</td><td>279 kr</td><td>19,2 °C</td></tr>
<tr><td>Kall, −7 °C ute</td><td>3,13 kr</td><td>−3 °C</td><td>16,9 kr (6 %)</td><td>261 kr</td><td>19,4 °C</td></tr>
<tr><td>Kall, −7 °C ute</td><td>3,13 kr</td><td>−8 °C</td><td>41,3 kr (16 %)</td><td>261 kr</td><td>19,3 °C</td></tr></table>
<p>Dagarna är extrema (spridning 3–5 kr) och ger alltså ett övre exempel. Över en hel vinter, med samma regel varje dygn och samma komfort, blir resultatet −3,0 / −1,7 / +2,5 / −3,8 % (2022/23–2025/26), alltså ingen besparing: de flesta dygn har liten spridning, kalla timmar ligger nära flödesgränsen och regeln gör samma ingrepp oavsett. Den optimerade planen på samma börvärdesspann (8–20, S3) ger 9,6 / 12,0 / 20,3 / 11,3 %.</p>

<h2>Pumpens kapacitet och varmvattnet</h2>
<p>AirX 170 enligt databladet (EN 14511), flöde 40 °C (medel av 35 och 45 °C): maximalt cirka <b>9,8 kW värme vid −15 °C och 12,5 kW vid −7 °C</b>, med <b>4,6 respektive 5,0 kW el</b>, COP 2,2 och 2,5. Vid +2 °C och varmare anger databladet bara dellast (COP 3,6 / 4,2 / 4,8 vid +2 / +7 / +12 °C).</p>
<table><tr><th>Ute</th><th>Värme, kompressor kWh/h</th><th>Varmvatten, kompressor kWh/h</th><th>Varmvattnets andel</th></tr>
<tr><td>−4 till 0 °C</td><td>3,91</td><td>0,48</td><td>11 %</td></tr>
<tr><td>0 till 4 °C</td><td>3,10–3,59</td><td>0,47–0,49</td><td>12–13 %</td></tr>
<tr><td>4 till 8 °C</td><td>1,97–2,50</td><td>0,38–0,42</td><td>14–16 %</td></tr>
<tr><td>8 till 12 °C</td><td>1,19</td><td>0,44</td><td>27 %</td></tr></table>
<p>Uppmätt 2022–2026 (timvis från pumpens räknare, konfidensintervall ±0,05 kWh/h från 0 °C och uppåt). Varmvattnet är 9–19 % av kompressorns el på vintern men upptar kompressorn 23–33 % av timmarna med högre framledning. Känslighetskörning där varmvattnet tar 14 % av kompressortiden (taket 4,3 i stället för 5,0 kW): S1 tak {dhw_pct('S1 Idag', 'perfekt', DHW_CAP):.1f} % (mot {dhw_pct('S1 Idag', 'perfekt', 5.0):.1f} %), praktiskt {dhw_pct('S1 Idag', 'dygn', DHW_CAP):.1f} % (mot {dhw_pct('S1 Idag', 'dygn', 5.0):.1f} %); S3 tak {dhw_pct('S3 Nedsänkning', 'perfekt', DHW_CAP):.1f} % (mot {dhw_pct('S3 Nedsänkning', 'perfekt', 5.0):.1f} %), praktiskt {dhw_pct('S3 Nedsänkning', 'dygn', DHW_CAP):.1f} % (mot {dhw_pct('S3 Nedsänkning', 'dygn', 5.0):.1f} %). Varmvattnet sänker alltså besparingen med ungefär 0,4–0,5 procentenheter. Själva varmvattnet är också flyttbart, men det ingår inte här.</p>

<h2>Bästa inställningar med verkliga begränsningar</h2>
<p>Priset känns bara så långt Nordpool har publicerat (planeringen sker kl 14 med priser till slutet av morgondagen). Golvet får högst 40 °C, kompressorn 5,0 kW, varje vecka ska medeltemperaturen vara minst 19,5 °C, rampgräns 2 °C/h åt båda hållen om inget annat anges. Praktisk besparing, fyra vintrar, andel av spotkostnaden för värme. Allt är trimmat på samma fyra vintrar (<code>code/sweep.py</code>).</p>
<table><tr><th>Börvärdesspann (nedre–övre)</th><th>8–20</th><th>12–20</th><th>14–20</th><th>16–20</th><th>12–19</th><th>12–22</th><th>14–22</th><th>8–24</th></tr>
<tr><td>Besparing</td><td>7,2 %</td><td>7,7 %</td><td>7,1 %</td><td>5,3 %</td><td>4,8 %</td><td><b>8,5 %</b></td><td>7,7 %</td><td>8,4 %</td></tr>
<tr><td>Kronor per vinter</td><td>727</td><td>770</td><td>715</td><td>535</td><td>483</td><td><b>849</b></td><td>769</td><td>848</td></tr></table>
<p>Lägre än 12 °C och högre än 22 °C tillför ingenting. Är övre gränsen bara 18 °C (alltså ingen uppvärmning över basnivån) blir det förlust, eftersom planen också behöver förvärma. Rummet gick ned till 17,4 °C vid spann 12–22 med komfortgolv 17 °C.</p>
<table><tr><th>Inställning</th><th>Värde</th><th>Besparing</th></tr>
<tr><td rowspan="5">Rampgräns, spann 8–20</td><td>obegränsad</td><td>8,8 %</td></tr>
<tr><td>4 °C/h båda håll</td><td>8,3 %</td></tr>
<tr><td>2 °C/h uppåt, fritt nedåt</td><td>8,1 %</td></tr>
<tr><td>2 °C/h båda håll</td><td>7,2 %</td></tr>
<tr><td>1 °C/h båda håll</td><td>5,7 %</td></tr>
<tr><td rowspan="4">Komfortgolv (tak 22 °C), spann 12–22</td><td>17,0 °C</td><td>8,5 %</td></tr>
<tr><td>18,0 °C</td><td>8,6 %</td></tr>
<tr><td>18,5 °C</td><td><b>8,9 %</b></td></tr>
<tr><td>19,0 °C</td><td>8,9 %</td></tr>
<tr><td rowspan="6">Värde på kvarvarande värme (× senaste veckans snittpris)</td><td>0,5</td><td>2,2 %</td></tr>
<tr><td>0,75</td><td>6,9 %</td></tr>
<tr><td>1,0</td><td><b>8,5 %</b></td></tr>
<tr><td>1,25</td><td>7,9 %</td></tr>
<tr><td>1,5</td><td>7,3 %</td></tr>
<tr><td>2,0</td><td>4,0 %</td></tr>
<tr><td rowspan="3">Planeringstid</td><td>kl 14</td><td>8,5 %</td></tr>
<tr><td>kl 16</td><td>8,3 %</td></tr>
<tr><td>kl 18</td><td>8,2 %</td></tr></table>
<div class="card"><b>Rekommendation.</b> Börvärde 12–22 med basnivå 18, steg 2 °C/h (eller större om pumpen tål det: ungefär 1,5 procentenheter), komfortgolv 18,5 °C, tak 22 °C, värde på kvarvarande värme 1,0 gånger veckomedelpriset, omplanering kl 14. Införd i Theria: nedre gräns 15 (6,9 %; 13 ger 8,4 %, 14 ger 7,7 %, 16 ger 5,6 %), eftersom huset bara har ungefär 130 timmar under 15,5 °C och djupare sänkningar än −3 °C är oprövade. 118 uppmätta sänkningar visar att flödet följer inom 1–3 timmar och att pumpen inte stannar i kyla men går av/på i milt väder. Nästa steg är kontrollerade stegförsök (−4, −5, sedan djupare) innan gränsen sänks. Börvärden under 16 °C har aldrig körts i huset. Varmvattnet sänker siffrorna med 0,4–0,5 procentenheter. Spridningen mellan vintrar är stor (5 % till 14 % med 12–22).</div>

<h2>Vad hade en lägre komfort sparat?</h2>
<p>Ett fast börvärde som håller en lägre medeltemperatur i stället för 19,5 °C, utan någon optimering (samma modell, fyra vintrar, spotpris):</p>
<table><tr><th>Medeltemperatur</th><th>Börvärde mot idag</th><th>Besparing per vinter</th><th>Energi per vinter</th></tr>
<tr><td>19,0 °C</td><td>−0,67 °C</td><td>628 kr (6,3 %)</td><td>678 kWh (7,0 %)</td></tr>
<tr><td>18,5 °C</td><td>−1,29 °C</td><td>1 218 kr (12,1 %)</td><td>1 314 kWh (13,7 %)</td></tr>
<tr><td>18,0 °C</td><td>−1,91 °C</td><td>1 794 kr (17,9 %)</td><td>1 935 kWh (20,1 %)</td></tr></table>
<p>Att sänka komforten en halv grad, till 19 °C, sparar alltså lika mycket som den praktiska optimeringen med dagens börvärdesspann (540 kr) och ligger nära dess tak (738 kr), utan prognos och utan att pumpen styrs timme för timme. Det är ungefär 14 % av värmeenergin per grads rumstemperatur (10 % per grads börvärde). Komfort och optimering kan kombineras, men det är inte simulerat. Script: <code>code/cooler.py</code>.</p>

<h2>Så ser en optimal plan ut</h2>
{img("05a_exempel_idag.png", "Optimal styrning med dagens spelrum, 14 dygn. Den praktiska planen (orange) slutar kallare än baslinjen; underskottet debiteras i tabellerna men inte i den ackumulerade kurvan.")}
{img("05b_exempel_utvidgat.png", "Optimal styrning med nedsänkning (S3, börvärde 8–20, flöde ≤ 40 °C): börvärdet sänks djupt under dyra timmar.")}
{img("05c_exempel_teoretiskt.png", "Stort spann (S4, börvärde 8–26, flöde ≤ 40 °C).")}

<h2>Prognoshorisont och komfortkrav</h2>
{img("07_prognoshorisont.png", "Besparing mot hur många dygn framåt priset är känt (praktisk planering, 7-dygns terminalvärde), prickat = hela vintern känd.")}
{img("11_komfortfönster.png", "Besparing mot hur strikt ”samma komfort” definieras. Ju längre period som får snitta ut, desto mer kan flyttas — och desto mer är komfortarbitrage.")}

<h2>Vikter</h2>
{img("09_regelvikter.png", "Regelvikter: besparing för ”N billigaste/dyraste timmar” per justering. Optimum ligger vid många timmar och stor justering — regeln vill vara mer flerdygns än dagsbunden.")}
{img("08_komfortvikt.png", "Pareto: pris på temperatur. Utan komfortkrav ”sparar” man 17–52 % men det mesta är att vara kallare.")}

<h2>Begränsningar — läs innan du litar på siffrorna</h2>
<ul>
<li><b>S3 och S4 extrapolerar.</b> Börvärden under 16 °C har aldrig körts i större omfattning. Modellerna är anpassade på börvärde 16–20 och flöde ≈ 30–40 °C. Ta S1 som det mest tillförlitliga talet och S3/S4 som riktning; verklig sänkning till 8 °C måste provas.</li>
<li><b>Theria:s egen planerare är inte denna LP.</b> Inställningarna ovan är optimala för LP-planeraren. Med börvärde 12–22 klarar Theria:s DP-planerare inte sina egna energineutralitetstester på syntetiska dygn (medelbörvärdet hamnar 1,8 °C under basen och rummet slutar 0,26 °C lägre än den platta banan). Därför kör testerna fortfarande på 16–20, och det nya spannet bör följas i skuggläge innan planen får skriva till pumpen.</li>
<li><b>Pumpens elmodell är linjär</b> i utetemperatur och börvärde. Den följer mätt förbrukning bra från 0 °C och uppåt (R² 0,91 per dygn) men är grov under −4 °C där det bara finns 144 timmars data, och den modellerar inte COP-kurvan explicit.</li>
<li><b>Rumsmodellen är identifierad på dygnsmedel</b> och förutsäger inte timvis bättre än ”ingen förändring” (korsvalidering 0,48 °C mot 0,48 °C). Därför visas känslighet: tak och praktisk besparing rör sig inte med golvets förstärkning, men dygnsdynamiken (tidskonstanten) och ”husets okända tillskott” är fortfarande modellantaganden.</li>
<li><b>Perfekt väder, inga störningar.</b> Spis, sol och människor ingår bara som ett konstant tillskott förankrat mot baslinjen. Sol- och nattmodellerna i Theria är avstängda här.</li>
<li><b>Bara ground-floor-modellen</b> och bara värmetimmar (varmvattnet ingår bara som en känslighet på kompressortaket). Övriga zoner (radiatorer) på samma flöde begränsar börvärdesrörelsen i verkligheten; Theria:s ”avvikelsekostnad” är inte med.</li>
<li><b>Spot, inte fakturapris.</b> Nätavgift, skatt och moms flyttar inte på timmarna, så kronbesparingen är densamma, men procenten är av spotkostnaden — av en total elfaktura med fasta avgifter blir den lägre. Negativa priser hanteras bara när kompressorn ändå drar ström.</li>
<li><b>Parametrar är trimmade på samma fyra vintrar</b> (terminalvikt 1,0 och regelrutnät). Konfidensintervallen speglar väder- och prisvariation mellan veckor men inte detta in-sample-val.</li>
</ul>

<h2>Vad det betyder för Theria</h2>
<ol>
<li>Dagens regel och den dagliga planeringen kan som mest ge ca {S['S1 Idag']['max']['pct']:.0f} % ({S['S1 Idag']['max']['sek_per_winter']:.0f} kr/vinter) med dagens börvärdesspann — räkna med hälften av det i praktiken ({S['S1 Idag']['day']['sek_per_winter']:.0f} kr) tills priset är känt mer än ett dygn framåt.</li>
<li>Den största hävstången är <b>börvärdets spelrum nedåt</b>: från ±2 °C till 8 °C ned ökar taket från {S['S1 Idag']['max']['pct']:.1f} till {S['S3 Nedsänkning']['max']['pct']:.1f} % och den praktiska besparingen från {S['S1 Idag']['day']['pct']:.1f} till {S['S3 Nedsänkning']['day']['pct']:.1f} %. Flödesgränsen ligger fast på 40 °C.</li>
<li>Planeringshorisonten bör vara <b>flera dygn</b> (prisprognos) — golvet kan inte följa dygnssvängningar. Om bara regler ska användas: flerdygns prisavvikelse slår dygnsrank.</li>
</ol>
<p><small>Kod och data: <code>documentation/savings-potential/code</code>. Körordning: fetch.py → prices.py → dataset.py → run_all.py (och <code>run_all.py window</code>) → dp_check.py → dhw_sens.py → figs.py → make_report.py. Enskilda dagar och säsongsregeln: ball.py, oneday.py. Inställningar: sweep.py (range, ramp, term, band, hour). Lägre komfort: cooler.py.</small></p>
</main></body></html>"""
(HERE / "report.html").write_text(html)
print("report bytes", len(html))
