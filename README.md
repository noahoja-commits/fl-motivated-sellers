# FL Motivated Sellers Dashboard

Streamlit dashboard over Florida's statewide parcel (NAL) data, surfacing
absentee, out-of-state, long-held, trust-owned, and multi-property leads.

## Data

Ships with three pre-computed files in `data/`:
- `leads.csv` — scored absentee-equity leads
- `by_owner.parquet` — owners with 5+ parcels
- `by_address.parquet` — mailing addresses on 5+ parcels

These come from the upstream pipeline at `fl-llc-properties`.

## Run locally

```powershell
pip install -r requirements.txt
streamlit run app.py
```
