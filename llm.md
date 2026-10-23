# Tunnel Load Simulator: synthetic electricity demand of road tunnels (Python, Streamlit)

**Instructions for a language model using this software.** Read this file first; the machine-readable versions are `websemantic.json` (fields, units, bounds) and `ontology/` (RDF/Turtle and JSON-LD, QUDT units).

Software DOI (all versions): https://doi.org/10.5281/zenodo.20080042 · licence MIT · language Python · authors: Haytham El-Houari, Cyril Voyant

Article: H. El-Houari, C. Voyant. Tunnel Load Simulator: an open-source generator for synthetic road-tunnel electricity demand. SoftwareX 35 (2026) 102946, https://doi.org/10.1016/j.softx.2026.102946 (published).

## What it does
Generates the electricity demand of a road tunnel from its geometry, traffic, lighting and ventilation, with random pollution episodes and accidents, over Monte Carlo realizations.

## Why (from the article)
Tunnel operators rarely have detailed load measurements, especially at design stage; a transparent first-order generator gives orders of magnitude and their variability.

## Method
- Power = fixed base + auxiliaries (length, tubes, depth factor) + lighting (daylight, traffic, lighting type) + ventilation (traffic, slope, altitude, events), times a multiplicative noise
- Monte Carlo: each realization uses base_seed + run; outputs are medians and P10–P90 envelopes

## Entry points
- `from tunnel_load_simulator.simulator import TunnelConfig, simulate_one_realization, run_monte_carlo`: Python API
- `streamlit run app.py`: web application
- `examples/basic_example.py, examples/monte_carlo_example.py`: examples

## Inputs
| Field | Unit | Type | Bounds | Required | Default (proposal only) | Meaning |
|---|---|---|---|---|---|---|
| `length_m` | m (unit:M) | float | min_exclusive 0 | yes | 1500 | Longueur du tunnel commune aux tubes ; convertie en kilomètres pour les charges proportionnelles à la longueur. |
| `n_tubes` | count (unit:NUM) | int | min 1, max 8 | yes | 2 | Nombre de tubes ; multiplie les charges de ventilation et d’auxiliaires et contribue au nombre total de voies. |
| `n_lanes_per_tube` | count (unit:NUM) | int | min 1, max 6 | yes | 2 | Nombre de voies dans chaque tube ; l’éclairage est dimensionné sur n_tubes × n_lanes_per_tube. |
| `altitude_m` | m (unit:M) | float |  | yes | 300 | Altitude utilisée dans le facteur de ventilation : 1 + max(altitude_m − 500, 0)/6000. |
| `max_depth_m` | m (unit:M) | float |  | yes | 80 | Profondeur maximale utilisée pour les auxiliaires : facteur 1 + min(max_depth_m, 800)/6000. |
| `gradient_percent` | % (unit:PERCENT) | float |  | yes | 2.0 | Pente en pourcentage, utilisée dans le facteur de ventilation 1 + abs(gradient_percent)/20. |
| `tunnel_context` | as stated | category |  | yes | peri-urban | Catégorie appliquant un multiplicateur au profil de trafic ; urban ajoute également une composante de pointe en soirée. |
| `lighting_type` | as stated | category |  | yes | LED adaptive | Catégorie déterminant la puissance spécifique installée, le couplage au trafic et la fraction minimale d’éclairage. |
| `ventilation_type` | as stated | category |  | yes | longitudinal | Catégorie déterminant la charge de ventilation par kilomètre et par tube, modulée par trafic, altitude, pente et événements. |
| `aux_kw_per_km_tube` | as stated | float |  | yes | 35.0 | Charge de référence des auxiliaires en kW/(km·tube), multipliée par la longueur, les tubes et le facteur de profondeur. |
| `base_fixed_kw` | kW (unit:KiloW) | float |  | yes | 40.0 | Puissance fixe ajoutée une seule fois à la demande totale du tunnel, avant le bruit multiplicatif global. |
| `traffic_level` | 1 (unit:UNITLESS) | float |  | yes | 1.0 | Multiplicateur du profil synthétique de trafic normalisé ; l’indice final est borné entre 0 et 2 dans TLS. |
| `morning_peak_hour` | h (unit:HR) | float |  | yes | 8.0 | Heure du centre du pic gaussien du matin, comprise entre 0 inclus et 24 exclu. |
| `evening_peak_hour` | h (unit:HR) | float |  | yes | 18.0 | Heure du centre du pic gaussien du soir, comprise entre 0 inclus et 24 exclu. |
| `peak_width_h` | h (unit:HR) | float |  | yes | 1.4 | Écart-type temporel du pic gaussien du matin en heures ; TLS utilise 1,15 fois cette valeur pour le soir. |
| `traffic_sensitivity` | 1 (unit:UNITLESS) | float |  | yes | 0.65 | Coefficient du trafic dans la charge de ventilation : facteur 0,15 + traffic_sensitivity × traffic_index. |
| `noise_sigma` | 1 (unit:UNITLESS) | float | min 0 | yes | 0.06 | Écart-type du bruit gaussien relatif appliqué à la puissance totale ; la puissance résultante est ramenée au minimum à zéro. |
| `pollution_probability_per_day` | 1 (unit:UNITLESS) | float | min 0, max 1 | yes | 0.05 | Probabilité journalière de tirer un événement synthétique de pollution ; TLS tire ensuite son début et sa durée. |
| `accident_probability_per_day` | 1 (unit:UNITLESS) | float | min 0, max 1 | yes | 0.015 | Probabilité journalière de tirer un événement synthétique d’accident ; TLS tire ensuite son début et sa durée. |
| `pollution_sensitivity` | 1 (unit:UNITLESS) | float |  | yes | 0.55 | Coefficient de surcroît de ventilation lorsqu’un événement de pollution est actif : ajout à 1 + pollution_sensitivity × indicateur. |
| `accident_sensitivity` | 1 (unit:UNITLESS) | float |  | yes | 0.75 | Coefficient de surcroît de ventilation lorsqu’un accident est actif ; il s’ajoute au coefficient de pollution dans le même facteur. |
| `start_date` | as stated | date |  | yes | 2025-01-01 | Date de début au format YYYY-MM-DD ; détermine le calendrier, les jours de semaine et les saisons du modèle. |
| `n_days` | d (unit:DAY) | int | min 1 | yes | 7 | Nombre de jours entiers simulés à partir de start_date. |
| `freq_minutes` | min (unit:MIN) | int | min 1 | yes | 60 | Durée constante du pas interne ; l’énergie du pas vaut puissance × freq_minutes/60. |
| `n_runs` | count (unit:NUM) | int | min 1 | yes | 3 | Nombre de réalisations Monte Carlo indépendantes ; chaque réalisation utilise base_seed + son indice. |
| `base_seed` | as stated | int | min 0 | yes | 42 | Identifiant initial du générateur pseudo-aléatoire ; chaque réalisation utilise base_seed + run. |

## Outputs
| Output | Unit | Meaning |
|---|---|---|
| `representative.lighting_kw` | kW (unit:KiloW) | power: Puissance électrique calculée. |
| `representative.ventilation_kw` | kW (unit:KiloW) | power: Puissance électrique calculée. |
| `representative.auxiliary_kw` | kW (unit:KiloW) | power: Puissance électrique calculée. |
| `representative.power_kw` | kW (unit:KiloW) | power: Puissance électrique calculée. |
| `representative.energy_kwh` | as stated | energy: Énergie du pas : power_kw × freq_minutes / 60. |
| `representative.traffic_index` | 1 (unit:UNITLESS) | relative_traffic_index: Indice relatif de trafic ; pas un nombre de véhicules par jour. |
| `representative.pollution_event` | 1 (unit:UNITLESS) | event_flag: Indicateur binaire d'événement de pollution simulé. |
| `representative.accident_event` | 1 (unit:UNITLESS) | event_flag: Indicateur binaire d'accident simulé. |
| `kpis.total_mwh` | as stated | energy: Énergie de toute la période : somme des energy_kwh / 1000. |
| `kpis.annualized_mwh` | as stated | annualized_energy: Extrapolation : total_mwh × 365 / n_days ; pas une année observée. |
| `kpis.peak_kw` | kW (unit:KiloW) | power: Puissance électrique calculée. |
| `kpis.mean_kw` | kW (unit:KiloW) | power: Puissance électrique calculée. |
| `kpis.load_factor` | 1 (unit:UNITLESS) | power_ratio: mean_kw / peak_kw par réalisation ; sans dimension, 0,6 équivaut à 60 %. |
| `kpis.specific_kwh_m_year` | as stated | length_normalized_annualized_energy: annualized_mwh × 1000 / length_m ; ensemble des tubes, par mètre de longueur du tunnel, pas par mètre-tube. |
| `kpis.n_pollution_events` | 1 (unit:UNITLESS) | transition_count: Nombre de transitions 0 vers 1 dans la série pollution_event ; un événement actif au premier pas n'est pas compté. |
| `kpis.n_accident_events` | 1 (unit:UNITLESS) | transition_count: Nombre de transitions 0 vers 1 dans la série accident_event ; un événement actif au premier pas n'est pas compté. |
| `daily.energy_kwh` | as stated | energy: Somme des énergies des pas natifs de la journée, réalisation 0. |
| `daily.mean_kw` | kW (unit:KiloW) | power: Moyenne des puissances des pas natifs de la journée, réalisation 0. |

## How to read the results
- Defaults and qualitative conventions (for example « tunnel long ») are proposals: ask the user to accept them (WebSemantic contract of TLS).

## Rules
1. Never invent a value. Each input is given by the user, an exact unit conversion of a given value, or a documented default that the user explicitly accepts.
2. Use the canonical unit of each field; convert exactly and say so (for example 2 km -> 2000 m). Never mix a percentage and a fraction.
3. Ask when a request is ambiguous or a required input is missing; do not infer a number from a place name or a qualitative word.
4. Run the code unmodified. Report only what was actually computed, with quantity, unit and the input values used; never present a planned or unavailable run as a result.
5. State the scope and the limits below when you report a result.

## Scope
Supported: estimate the demand (energy, peak, load factor) of a tunnel under stated assumptions; compare lighting, ventilation and traffic configurations with controlled seeds.
Out of scope (refuse or redirect): certification; prediction of a specific real tunnel without calibration.

## Limits
- First-order generator to calibrate on measurements before engineering use; not a certification or a prediction of a real tunnel's consumption.

A full WebSemantic contract (qualitative conventions, SHACL shapes) exists for this software: https://github.com/cyrilvoyant/websemantic (pack tls: LLM-CONTRACT.md, ontology.ttl, shapes.ttl).

## Cite
Software: Haytham El-Houari, Cyril Voyant. Tunnel Load Simulator: synthetic electricity demand of road tunnels (Python, Streamlit). Zenodo. https://doi.org/10.5281/zenodo.20080042
Article: H. El-Houari, C. Voyant. Tunnel Load Simulator: an open-source generator for synthetic road-tunnel electricity demand. SoftwareX 35 (2026) 102946. https://doi.org/10.1016/j.softx.2026.102946

Guide with the theory and runnable examples: https://cyrilvoyant.github.io/ · author: https://person.cyrilvoyant.com/ · generated 2026-10-23, documentation only.
