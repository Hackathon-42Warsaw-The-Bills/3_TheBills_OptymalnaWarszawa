def prognozuj_populacje_klastra(
    x_obecna_populacja: int,
    kubatura_gunb_m3: float = 0.0,
    powierzchnia_wolna_mpzp_m2: float = 0.0,
    intensywnosc_mpzp: float = 1.6,
    stopa_demo_roczna: float = 0.0,
    horyzont_lat: int = 10
) -> dict:
    # 1. Baza demograficzna
    pop_baza = x_obecna_populacja * ((1 + stopa_demo_roczna) ** horyzont_lat)
    
    # 2. Nowi mieszkańcy z pozwoleń (GUNB)
    pum_gunb = kubatura_gunb_m3 * 0.26
    delta_gunb = (pum_gunb / 33.0) * 0.90
    
    # 3. Nowi mieszkańcy z rezerw MPZP
    pum_mpzp_max = powierzchnia_wolna_mpzp_m2 * intensywnosc_mpzp * 0.70
    delta_mpzp = (pum_mpzp_max / 33.0) * 0.35  # realizacja 35% rezerwy w 10 lat
    
    pop_calkowita_t10 = pop_baza + delta_gunb + delta_mpzp
    
    return {
        "populacja_obecna": x_obecna_populacja,
        "przyrost_gunb_pewny": int(round(delta_gunb)),
        "przyrost_mpzp_potencjalny": int(round(delta_mpzp)),
        "populacja_za_10_lat": int(round(pop_calkowita_t10)),
        "delta_calkowita": int(round(delta_gunb + delta_mpzp))
    }