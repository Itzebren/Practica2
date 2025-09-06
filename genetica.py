# ============================================================
# Algoritmo genético para encontrar al "individuo perfecto" (todos los genes = 9)
# ============================================================

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Set

# --- Soporte para graficar el árbol
import os, sys, shutil
import graphviz 

def _ensure_graphviz_on_path():
    if shutil.which("dot"):
        return
    # Rutas comunes de Windows (winget/MSI/Chocolatey/Conda)
    candidates = [
        r"C:\Program Files\Graphviz\bin",
        r"C:\Program Files (x86)\Graphviz\bin",
        r"C:\ProgramData\chocolatey\lib\graphviz\tools\graphviz\bin",
        os.path.expanduser(r"~\AppData\Local\Programs\Graphviz\bin"),
        os.path.join(sys.prefix, "Library", "bin"),
    ]
    for p in candidates:
        dot = os.path.join(p, "dot.exe")
        if os.path.isdir(p) and os.path.isfile(dot):
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
            break

_ensure_graphviz_on_path()

# ---------- Soporte para algoritmo Blossom ----------
try:
    import networkx as nx
    HAVE_NX = True
except Exception:
    HAVE_NX = False

# ----------------------- Parámetros --------------------------
N_INDIVIDUOS = 16          # individuos por generación
N_GENES = 20                # genes por individuo
VALORES = [1,2,3,4,5,6,7,8,9]

MAX_GENERACIONES = 1000     # límite de generaciones por corrida
IMPRIMIR_TODO = False       # True = imprime los 100; False = imprime solo lo que escojamos

# Mutación: probabilidad del 0.1 por gen, máx 2 genes
PROB_MUTACION = 0.10
MAX_GENES_MUTADOS = 2
CONSERVAR_9S = True     # opción de no tocar 9s

# Distribuciones para random.choices (más peso a valores altos)
PESOS_INI = [2, 2, 3, 2, 6, 1, 20, 10, 30]   
PESOS_MUT = [1, 1, 2, 2, 3, 9, 18, 28, 38]  

# Fitness de 180 que necesita tener el individuo perfectO :)
OBJETIVO = [9] * N_GENES

# -------------------- Utilidades de muestreo -----------------
def sample_choices(valores: List[int], pesos: List[int], evitar: Optional[int] = None) -> int:
    if evitar is None:
        return random.choices(valores, weights=pesos, k=1)[0]
    vals = [v for v in valores if v != evitar]
    ws   = [w for v, w in zip(valores, pesos) if v != evitar]
    # Defensa por si quedaran todos 0 
    if sum(ws) <= 0:
        return random.choice(vals)
    return random.choices(vals, weights=ws, k=1)[0]

# ----------------------- Estructura para los datos base --------------------------
@dataclass
class Individuo:
    id: int
    generacion: int
    padres: Optional[Tuple[int, int]]  
    genes: List[int]

# ------------------- Creación de población -------------------
def crear_poblacion_inicial(n: int) -> List[Individuo]:
    pobl = []
    for i in range(n):
        genes = [sample_choices(VALORES, PESOS_INI) for _ in range(N_GENES)]
        pobl.append(Individuo(id=i, generacion=0, padres=None, genes=genes))
    return pobl

# ------------------- Relaciones de parentesco ----------------
def son_hermanos(a: Individuo, b: Individuo) -> bool:
   
    return bool(a.padres and b.padres and (set(a.padres) & set(b.padres)))

def abuelos_de(ind: Individuo, idx: Dict[int, Individuo]) -> Set[int]:
    
    if not ind.padres:
        return set()
    res: Set[int] = set()
    for pid in ind.padres:
        p = idx.get(pid)
        if p and p.padres:
            res |= set(p.padres)
    return res

def son_primos(a: Individuo, b: Individuo, idx: Dict[int, Individuo]) -> bool:
   
    if not a.padres or not b.padres:
        return False
    return bool(abuelos_de(a, idx) & abuelos_de(b, idx))

def pareja_valida_generacional(a: Individuo, b: Individuo, idx: Dict[int, Individuo], gen_actual: int) -> bool:

    if gen_actual == 0:
        return True
    if son_hermanos(a, b):
        return False
    if gen_actual >= 2 and son_primos(a, b, idx):
        return False
    return True

def construir_indice(*pops: List[Individuo]) -> Dict[int, Individuo]:
    
    idx: Dict[int, Individuo] = {}
    for pob in pops:
        for ind in pob:
            idx[ind.id] = ind
    return idx

# ---------------------- Métricas de las generaciones ------------------
def fitness(ind: Individuo) -> int:
    return sum(ind.genes)

def promedio_fitness(poblacion: List[Individuo]) -> float:
    return sum(fitness(ind) for ind in poblacion) / len(poblacion)

def es_perfecto(ind: Individuo) -> bool:
    return ind.genes == OBJETIVO

def encontrar_perfecto(poblacion: List[Individuo]) -> Optional[Individuo]:
    for ind in poblacion:
        if es_perfecto(ind):
            return ind
    return None

def contar_9s_por_gen(poblacion: List[Individuo]) -> List[int]:
    return [sum(1 for ind in poblacion if ind.genes[g] == 9) for g in range(N_GENES)]

def contar_ge8_por_gen(poblacion: List[Individuo]) -> List[int]:
    return [sum(1 for ind in poblacion if ind.genes[g] >= 8) for g in range(N_GENES)]

# -------------------- Reproducción y mutación ----------------
def hijos_por_promedio(p1: Individuo, p2: Individuo) -> Tuple[List[int], List[int]]:

    give_ceil_to_h1 = (random.random() < 0.5)
    h1, h2 = [], []
    for g1, g2 in zip(p1.genes, p2.genes):
        s = g1 + g2
        if s % 2 == 0:
            v = s // 2
            h1.append(v); h2.append(v)
        else:
            lo, hi = s // 2, s // 2 + 1
            if give_ceil_to_h1:
                h1.append(hi); h2.append(lo)
            else:
                h1.append(lo); h2.append(hi)
    return h1, h2

def mutar_genes(genes: List[int]) -> List[int]:

    nuevos = genes[:]
    idxs = list(range(N_GENES))
    random.shuffle(idxs)
    mutados = 0
    for i in idxs:
        if mutados >= MAX_GENES_MUTADOS:
            break
        if CONSERVAR_9S and nuevos[i] == 9:
            continue
        if random.random() < PROB_MUTACION:
            nuevos[i] = sample_choices(VALORES, PESOS_MUT, evitar=nuevos[i])
            mutados += 1
    return nuevos

def crear_siguiente_generacion(poblacion: List[Individuo],
                               parejas: List[Tuple[Individuo, Individuo]],
                               gen_actual: int) -> List[Individuo]:
    nueva = []
    prox_id = max(ind.id for ind in poblacion) + 1
    for a, b in parejas:
        g1, g2 = hijos_por_promedio(a, b)
        hijo1 = Individuo(id=prox_id, generacion=gen_actual+1, padres=(a.id, b.id), genes=mutar_genes(g1)); prox_id += 1
        hijo2 = Individuo(id=prox_id, generacion=gen_actual+1, padres=(a.id, b.id), genes=mutar_genes(g2)); prox_id += 1
        nueva.extend([hijo1, hijo2])
    return nueva

# -------------------- Scoring para emparejar -----------------
def pesos_duales(poblacion: List[Individuo]) -> Tuple[List[float], List[float]]:

    c9  = contar_9s_por_gen(poblacion)      # 0..100
    c8p = contar_ge8_por_gen(poblacion)     
    w9 = [(N_INDIVIDUOS - c + 1)**2 for c in c9]
    w8 = [(N_INDIVIDUOS - c + 1)     for c in c8p]
    return w9, w8

def score_pareja_extremo(a: Individuo, b: Individuo, w9: List[float], w8: List[float]) -> int:

    s9_units = 0.0
    s8_units = 0.0
    for g, (x, y) in enumerate(zip(a.genes, b.genes)):
        t = x + y
        if t >= 18:
            s9_units += 2.0 * w9[g]
        elif t == 17:
            s9_units += 1.0 * w9[g]
        elif 15 <= t <= 16:
            s8_units += 1.0 * w8[g]
    return int(1_000_000 * s9_units + 1_000 * s8_units + (sum(a.genes) + sum(b.genes)))

# --------------- Emparejamiento (Blossom / greedy) -----------
def formar_parejas_optimas(poblacion: List[Individuo], idx: Dict[int, Individuo], gen_actual: int) -> List[Tuple[Individuo, Individuo]]:

    ids = [ind.id for ind in poblacion]
    id2 = {ind.id: ind for ind in poblacion}
    w9, w8 = pesos_duales(poblacion)

    # ----- Intento 1: Blossom (óptimo de máximo peso) -----
    if HAVE_NX:
        G = nx.Graph(); G.add_nodes_from(ids)
        for i in range(len(ids)):
            a = id2[ids[i]]
            for j in range(i + 1, len(ids)):
                b = id2[ids[j]]
                if pareja_valida_generacional(a, b, idx, gen_actual):
                    G.add_edge(a.id, b.id, weight=score_pareja_extremo(a, b, w9, w8))
        M = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True, weight='weight')
        if len(M) == len(ids) // 2:
            usados = set(); parejas = []
            for u, v in M:
                if u in usados or v in usados: continue
                usados.add(u); usados.add(v)
                parejas.append((id2[u], id2[v]))
            return parejas
        # si no hay matching perfecto (por restricciones), cae al fallback (opción alternativa)

    # ----- Intento 2: Greedy y pequeños reintentos -----
    # Verifica compatibilidad (respetando parentesco)
    vecinos: Dict[int, Set[int]] = {ind.id:set() for ind in poblacion}
    for i in range(len(ids)):
        a = id2[ids[i]]
        for j in range(i+1, len(ids)):
            b = id2[ids[j]]
            if pareja_valida_generacional(a, b, idx, gen_actual):
                vecinos[a.id].add(b.id); vecinos[b.id].add(a.id)

    def construir_matching() -> Optional[List[Tuple[int,int]]]:
        libres = set(ids)
        parejas_ids: List[Tuple[int,int]] = []

        def elegir_siguiente():
            # Atajo que lleva a una solución óptima: el de menor grado (menos opciones) para evitar callejones
            return min(libres, key=lambda u: (len(vecinos[u] & libres), random.random()))

        while libres:
            u = elegir_siguiente(); libres.remove(u)
            cands = list((vecinos[u] & libres) - {u})
            if not cands:
                return None
            # Ordena candidatos por score decreciente 
            cands.sort(key=lambda v: (score_pareja_extremo(id2[u], id2[v], w9, w8), random.random()), reverse=True)
            v = cands[0]
            if v not in libres:
                return None
            libres.remove(v)
            parejas_ids.append((u, v))
        return parejas_ids

    for _ in range(12):
        res = construir_matching()
        if res is not None and len(res)*2 == len(ids):
            return [(id2[a], id2[b]) for a,b in res]

    raise RuntimeError("No se pudo construir un emparejamiento válido con las restricciones.")

def formar_parejas(poblacion: List[Individuo], gen_actual: int, indice_global: Dict[int, Individuo]) -> List[Tuple[Individuo, Individuo]]:

    assert len(poblacion) % 2 == 0, "La población debe ser par."
    if gen_actual == 0:
        barajada = poblacion[:]
        random.shuffle(barajada)
        return [(barajada[i], barajada[i+1]) for i in range(0, len(barajada), 2)]
    return formar_parejas_optimas(poblacion, indice_global, gen_actual)

# ---------------------- Simulación: una corrida --------------
def imprimir_poblacion(pobl: List[Individuo], titulo: str = "", mostrar: int = 10):
    if titulo:
        print(f"\n=== {titulo} ===")
    total9 = sum(sum(1 for g in ind.genes if g == 9) for ind in pobl)
    total_ge8 = sum(sum(1 for g in ind.genes if g >= 8) for ind in pobl)
    print(f"Total: {len(pobl)} | prom_fit: {promedio_fitness(pobl):.2f} | 9s totales: {total9} | ≥8 totales: {total_ge8}")
    print("id | gen | madre | padre | genes")
    it = pobl if IMPRIMIR_TODO else pobl[:mostrar]
    for ind in it:
        m = ind.padres[0] if ind.padres else None
        p = ind.padres[1] if ind.padres else None
        print(f"{ind.id:4d} | {ind.generacion:3d} | {str(m):>5} | {str(p):>5} | {ind.genes}")

def simular(max_generaciones: int = MAX_GENERACIONES, verbose: bool = True):
    
    gen0 = crear_poblacion_inicial(N_INDIVIDUOS)
    poblaciones = [gen0]

    if verbose:
        imprimir_poblacion(gen0, "Generación 0 (fundadores)")

    if (p := encontrar_perfecto(gen0)) is not None:
        if verbose: print(" Individuo Perfecto en Generación 0:", p.id)
        return p, 0, poblaciones

    gen_actual = 0
    while gen_actual < max_generaciones:
        idx = construir_indice(*poblaciones)
        parejas = formar_parejas(poblaciones[-1], gen_actual, idx)
        assert len(parejas) * 2 == len(poblaciones[-1]), "Número de parejas inconsistente."
        hijos = crear_siguiente_generacion(poblaciones[-1], parejas, gen_actual)
        poblaciones.append(hijos)
        gen_actual += 1

        if verbose:
            best = max(hijos, key=fitness)
            best9 = sum(1 for g in best.genes if g == 9)
            total9 = sum(sum(1 for g in ind.genes if g == 9) for ind in hijos)
            total_ge8 = sum(sum(1 for g in ind.genes if g >= 8) for ind in hijos)
            print(f"\nGen {gen_actual}: best_fit={fitness(best)}/{N_GENES*9} | best 9s={best9}/{N_GENES} | total 9s={total9} | total ≥8={total_ge8} | prom={promedio_fitness(hijos):.2f}")
            imprimir_poblacion(hijos, f"Generación {gen_actual}")

        if (p := encontrar_perfecto(hijos)) is not None:
            if verbose:
                print(f"\n ¡Individuo Perfecto encontrado! id={p.id} en Generación {p.generacion}")
            return p, gen_actual, poblaciones

    if verbose:
        print("\n No apareció el individuo perfecto dentro del límite.")
    return None, gen_actual, poblaciones


# ====== DIAGRAMA DE ÁRBOL (Graphviz) ======
try:
    import graphviz
    _GRAPHVIZ_OK = True
except Exception:
    _GRAPHVIZ_OK = False

def _build_registry(poblaciones):
   
    idx = {}
    for cohorte in poblaciones:
        for ind in cohorte:
            idx[ind.id] = ind
    return idx

def _collect_ancestors(registry, nid, niveles, nodos, aristas):
   
    if nid is None or nid not in registry or nid in nodos or niveles < 0:
        return
    nodos.add(nid)
    ind = registry[nid]
    if ind.padres:
        for pid in ind.padres:
            if pid is not None and pid in registry:
                aristas.add((pid, nid))
                _collect_ancestors(registry, pid, niveles - 1, nodos, aristas)

def export_generation_tree(poblaciones, gen_idx, niveles=3, base_filename=None,
                           mostrar_9s=True, dibujar_parejas=True):
    
    if not _GRAPHVIZ_OK:
        print("[AVISO] Falta Graphviz (pip install graphviz) o el binario 'dot' en PATH.")
        return

    if gen_idx < 0 or gen_idx >= len(poblaciones):
        raise ValueError("gen_idx fuera de rango")

    registry = _build_registry(poblaciones)
    cohorte = poblaciones[gen_idx]
    if not base_filename:
        base_filename = f"gen_{gen_idx}"

    # --- nodos a incluir (cohorte + ancestros hasta 'niveles') ---
    nodos = set()
    aristas = set()
    for ind in cohorte:
        _collect_ancestors(registry, ind.id, niveles, nodos, aristas)

    # --- parejas en esta cohorte (si hay gen siguiente, se infieren por 'padres' de los hijos) ---
    parejas = set()
    if dibujar_parejas and gen_idx + 1 < len(poblaciones):
        for hijo in poblaciones[gen_idx + 1]:
            if hijo.padres:
                a, b = hijo.padres
                if a is not None and b is not None:
                    par = (min(a, b), max(a, b))
                    parejas.add(par)

    # --- preparar dibujo ---
    dg = graphviz.Digraph(comment=f"Generación {gen_idx}")
    dg.attr(rankdir="TB")
    dg.attr(ranksep="1.2", nodesep="0.6")
    dg.attr(dpi="140")

    # Agrupar por generación para rank=same
    gens = {}
    for nid in nodos:
        g = registry[nid].generacion
        gens.setdefault(g, []).append(nid)

    # Etiquetas de nodos
    def _label(nid):
        ind = registry[nid]
        if mostrar_9s:
            n9 = sum(1 for v in ind.genes if v == 9)
            return f"id={ind.id}\\ngen={ind.generacion}\\n9s={n9}/{len(ind.genes)}"
        else:
            return f"id={ind.id}\\ngen={ind.generacion}"

    # Nodos (resalta la cohorte viva)
    for nid in nodos:
        if registry[nid].generacion == gen_idx:
            dg.node(f"n{nid}", label=_label(nid), shape="box", style="filled", fillcolor="lightgrey")
        else:
            dg.node(f"n{nid}", label=_label(nid), shape="box")

    # Forzar layout horizontal por generación
    for g, ids in sorted(gens.items()):
        with dg.subgraph() as s:
            s.attr(rank="same")
            for nid in ids:
                s.node(f"n{nid}")

    # Aristas padre->hijo
    for (u, v) in aristas:
        dg.edge(f"n{u}", f"n{v}")

    # Parejas (línea discontinua) — solo si ambos padres están en el dibujo
    if parejas:
        for (u, v) in parejas:
            if u in nodos and v in nodos and registry[u].generacion == gen_idx and registry[v].generacion == gen_idx:
                dg.edge(f"n{u}", f"n{v}", dir="none", style="dashed", color="gray", label="pareja")

    # Guardar DOT y, si se puede, render a PNG
    dot_path = dg.save(filename=f"{base_filename}.dot")
    try:
        png_path = dg.render(filename=base_filename, format="png", cleanup=True)
        print(f"[OK] Árbol guardado: {png_path}  (DOT: {dot_path})")
    except Exception as e:
        print(f"[AVISO] No se pudo renderizar PNG (¿falta 'dot' en PATH?). DOT en: {dot_path}. Error: {e}")


        # ------------------------------ Main -------------------------
if __name__ == "__main__":
    perfecto, gen_encontrado, poblaciones = simular()
  
# 1) Dibuja la generación donde nació el individuo perfecto (con sus ancestros)
if perfecto:
    export_generation_tree(poblaciones, gen_idx=perfecto.generacion, niveles=3,
                           base_filename=f"arbol_gen_{perfecto.generacion}")