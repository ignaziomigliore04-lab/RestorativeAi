Restorative AI - rebuilt Streamlit version

Run:
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py

What this version does:
- uses the verified materials database directly
- removes the three inputs you wanted to eliminate
- calculates SSI, BRI, FSI, EDI, WCI
- recommends restoration type and ranks materials
- shows sources for the top material

Main files:
- app.py
- utils/engine.py
- data/materialmatch_site_optimized_database_v1.xlsx
