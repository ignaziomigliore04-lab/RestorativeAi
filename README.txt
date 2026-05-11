Restorative AI - Material-first decision support v2.9.5 OPTIMAL CANDIDATE

Focus del progetto
- La decisione principale è sempre la classe/tipo di materiale dentario.
- Il tipo di restauro resta solo come envelope operativo secondario: non determina lo score materiale.
- Il ranking è deterministico, database-driven, senza adaptive learning e senza feedback automatici.

Novità v2.9.5
1. Output ancora più diretto
   - Meno testo visibile.
   - Più card, parole chiave, pesi, score e tabelle compatte.
   - Le spiegazioni lunghe restano disponibili solo in accordion/tendine.

2. Breve spiegazione visibile del perché il materiale è ottimale
   - Nuova card “Perché è il materiale ottimale”.
   - Mostra 3 punti brevi:
     a) driver dominante e fit,
     b) coerenza clinica con settore/strategia/score,
     c) vantaggio o alternativa vicina.

3. Miglioramento leggibilità keyword
   - I chip ora rendono davvero in grassetto le parole chiave principali.
   - La schermata risultato comunica subito il materiale migliore e i motivi essenziali.

4. Logica material-first revisionata
   - Caso clinico -> filtro actionability -> indici clinici -> pesi materiali -> database -> ranking classi materiali.
   - Restauro compatibile mostrato solo come cornice secondaria.
   - Sesso paziente resta micro-contesto biologico debole e non prescrive mai il materiale.
   - Settore anteriore/posteriore resta driver forte di estetica, meccanica e workflow.

5. Validazione aggiornata su 200 casi
   - Casi sintetici plausibili e actionable.
   - Copertura completa degli input categoriali.
   - 0 red flag cliniche gravi.

Risultati validazione v2.9.5
- Casi testati: 200
- Casi actionable: 200
- Errori: 0
- Red flag gravi: 0
- Yellow flag / near ties: 102
- Copertura input: 110/110 livelli
- Direct top: 49.0%
- Indirect top: 51.0%
- Classi arrivate prime: 10
- Score medio top: 82.7
- Gap medio: 2.8

Distribuzione classi top principali
- Zirconia 3Y-TZP / high-strength: 56
- Composito nanoibrido: 55
- Composito microibrido: 32
- Lithium disilicate: 20
- Zirconia ad alta traslucenza 4Y/5Y: 19
- Nanofilled / nanoriempito: 9

Avvio locale
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py

Nota
- Non vengono salvati dati di utilizzo.
- Non ci sono componenti adaptive/learning.
- La Validation Mode resta disponibile per testare 200 casi sintetici material-first.
