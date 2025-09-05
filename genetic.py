import random
from statistics import mean
from typing import List, Tuple, Optional

# ==========================
# Parámetros
# ==========================
NUM_INDIVIDUOS = 100
NUM_GENES = 20
GEN_MIN, GEN_MAX = 1, 9

ELITISM = 2                 # 2 de los mejores individuos de cada generación pasan directamente a la siguiente generación
PMUT_BASE = 0.20            # con 20 genes, esperamos 4 genes mutados por hijo
PMUT_MAX  = 0.25            # mutación máxima
PMUT_DOWN = 0.01            # pequeña probabilidad de bajar un gen para mantener diversidad genética en la población

ADAPT_WINDOW = 10           # si en 10 gens no mejora el mejor, sube mutación
ADAPT_STEP   = 0.02         # incremento/decremento adaptativo de mutación

MAX_GENERATIONS = 100000
RANDOM_SEED = 42            # semilla que fija el camino evolutivo que seguirá el algoritmo genético
                            # controla la secuencia de números aleatorios para inicializar la población, decidir cruces, mutaciones

if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

assert 0.0 <= PMUT_BASE <= PMUT_MAX <= 0.25, "La mutación debe ser <= 25%"

# ==========================
# Individuo y familia
# ==========================
class Individuo:
    _next_id = 0 #id de cada individuo para saber si son familia o no
    def __init__(self, genes: List[int], padre: Optional[int]=None, madre: Optional[int]=None):
        self.id = Individuo._next_id
        Individuo._next_id += 1
        self.genes = genes[:]
        registrar_familia(self.id, padre, madre)

    def fitness(self) -> int:
        return sum(self.genes)

parents_map = {}      # id -> (padre, madre), los hermonas comparten al menos un padre
grandparents_map = {} # id -> set(abuelos), los primos comparten abuelos

def registrar_familia(hijo_id: int, padre: Optional[int], madre: Optional[int]):
    parents_map[hijo_id] = (padre, madre) #guarda quiénes son los padres de ese hijo
    gp = set() #crea un conjunto vacío para ir llenando con los abuelos
    for p in (padre, madre):
        if p is not None:
            gp |= set(parents_map.get(p, (None, None)))
    gp.discard(None)
    grandparents_map[hijo_id] = gp #guarda el conjunto de abuelos

def individuo_aleatorio() -> Individuo:
    genes = [random.randint(GEN_MIN, GEN_MAX) for _ in range(NUM_GENES)]
    return Individuo(genes)

def fitness(ind: Individuo) -> int:
    return ind.fitness()

# ==========================
# Parentesco
# ==========================
def are_siblings(a: Individuo, b: Individuo) -> bool:
    pa = set(parents_map.get(a.id, (None, None))); pa.discard(None)
    pb = set(parents_map.get(b.id, (None, None))); pb.discard(None)
    return len(pa & pb) > 0

def are_cousins(a: Individuo, b: Individuo) -> bool:
    ga = grandparents_map.get(a.id, set())
    gb = grandparents_map.get(b.id, set())
    return len(ga) > 0 and len(gb) > 0 and len(ga & gb) > 0

def related_prohibited(a: Individuo, b: Individuo) -> bool:
    return are_siblings(a, b) or are_cousins(a, b)

# ==========================
# Cruce y mutación
# ==========================
def promedio_gen(a: int, b: int) -> int:
    s = (a + b) / 2.0
    base = int(s)
    frac = s - base
    # redondeo hacia arriba con 50% cuando hay fracción
    if frac > 0 and random.random() < 0.5:
        base += 1
    return max(GEN_MIN, min(GEN_MAX, base))

def cruzar(p1: Individuo, p2: Individuo) -> Tuple[Individuo, Individuo]:
    g1 = [promedio_gen(a, b) for a, b in zip(p1.genes, p2.genes)]
    g2 = [promedio_gen(a, b) for a, b in zip(p1.genes, p2.genes)]
    h1 = Individuo(g1, p1.id, p2.id)
    h2 = Individuo(g2, p1.id, p2.id)
    return h1, h2

#Gen = 6 → muta → sube a 7 (+ probabilidad).
#Gen = 9 → rara vez baja a 8.
def mutar_gen_sesgado(g: int, pm: float) -> int:
    if random.random() < pm:
        dist = GEN_MAX - g
        if dist > 0:
            # más lejos de 9 => mayor probabilidad de subir +1
            p_up = min(1.0, 0.15 + 0.10 * dist)  # ~0.25..1.0
            if random.random() < p_up:
                g = min(GEN_MAX, g + 1)
            else:
                if random.random() < PMUT_DOWN:
                    g = max(GEN_MIN, g - 1)
        else:
            if random.random() < PMUT_DOWN:
                g = max(GEN_MIN, g - 1)
    return g

#Antes: [7, 6, 9, 5, …]
#Después de mutar: [7, 7, 9, 6, …]
def mutar(ind: Individuo, pm: float) -> Individuo:
    ind.genes = [mutar_gen_sesgado(g, pm) for g in ind.genes]
    return ind

# ==========================
# Emparejamiento
# Backtracking
# ==========================
def compatibles(poblacion: List[Individuo]):
    """Devuelve ids, by_id y compat: dict id -> lista de ids compatibles."""
    ids = [ind.id for ind in poblacion]
    by_id = {ind.id: ind for ind in poblacion}
    compat = {i: [] for i in ids}
    for i in ids:
        a = by_id[i]
        for j in ids:
            if j == i:
                continue
            b = by_id[j]
            if not related_prohibited(a, b):
                compat[i].append(j)
    return ids, by_id, compat

def emparejar_sin_familia(poblacion: List[Individuo]) -> List[Tuple[Individuo, Individuo]]:
    """
    Encuentra N/2 parejas válidas si existe matching perfecto.
    Heurísticas:
      - Grado mínimo primero (menos opciones primero).
      - Entre candidatos, prioriza fitness combinado.
      - Backtracking si se atasca.
    """
    ids, by_id, compat = compatibles(poblacion)

    #si alguien no tiene opciones, no hay matching
    for i in ids:
        if len(compat[i]) == 0:
            raise RuntimeError("No existe matching válido: algún individuo no tiene pareja compatible (hermanos/primos).")

    # ordenar por grado (menos opciones primero)
    orden = sorted(ids, key=lambda k: len(compat[k]))

    usados = set() # IDs ya emparejados
    parejas: List[Tuple[Individuo, Individuo]] = [] 

    def candidatos_ordenados(i):
        ci = compat[i]
        # solo no usados, en orden de mayor fitness combinado
        return sorted(
            (j for j in ci if j not in usados and j != i),
            key=lambda j: by_id[i].fitness() + by_id[j].fitness(),
            reverse=True
        )

    def bt(idx: int) -> bool:
        if idx == len(orden):
            return True
        i = orden[idx]
        if i in usados:
            return bt(idx + 1)

        for j in candidatos_ordenados(i):
            if j in usados:
                continue
            a, b = by_id[i], by_id[j]
            # redundante (ya filtrado), pero seguro
            if related_prohibited(a, b):
                continue
            usados.add(i); usados.add(j)
            parejas.append((a, b))

            if bt(idx + 1):
                return True

            # backtrack
            parejas.pop()
            usados.remove(j); usados.remove(i)

        return False

    # Intentar algunas permutaciones suaves del orden (bloques barajados)
    intentos = 6
    for _ in range(intentos):
        parejas.clear()
        usados.clear()
        # reordenación suave por bloques para evitar peores casos
        nuevo_orden = []
        start = 0
        while start < len(orden):
            end = min(start + 6, len(orden))
            bloque = orden[start:end]
            random.shuffle(bloque)
            nuevo_orden.extend(bloque)
            start = end
        orden = nuevo_orden

        if bt(0):
            if len(parejas) * 2 == len(ids):
                return parejas

    raise RuntimeError("No se pudo construir emparejamientos válidos tras varios intentos (sin hermanos/primos).")

# ==========================
# Main
# ==========================
def inicializar_poblacion() -> List[Individuo]:
    return [individuo_aleatorio() for _ in range(NUM_INDIVIDUOS)]

def ejecutar():
    poblacion = inicializar_poblacion()
    objetivo = NUM_GENES * GEN_MAX
    gen = 0
    pmut = PMUT_BASE
    best_history = []
    best_ever = -1

    while gen < MAX_GENERATIONS:
        gen += 1
        poblacion.sort(key=fitness, reverse=True)
        best = poblacion[0]
        best_fit = best.fitness()
        avg_fit = mean(ind.fitness() for ind in poblacion)
        frac_nines = sum(g == GEN_MAX for ind in poblacion for g in ind.genes) / (NUM_INDIVIDUOS * NUM_GENES)

        print(f"Gen {gen:4d} | Mejor={best_fit:3d}/{objetivo} | Prom={avg_fit:6.2f} | %9s={(100*frac_nines):5.2f}% | pmut={pmut:.3f}")

        # criterio de paro: individuo perfecto
        if best_fit == objetivo:
            print("\n¡Individuo perfecto encontrado!")
            print(f"Generación: {gen}")
            print("Individuo (id, genes):", best.id, best.genes)
            return gen, best

        # Mutación adaptativa (≤ 0.25)
        best_history.append(best_fit)
        if best_fit > best_ever:
            best_ever = best_fit
        if len(best_history) >= ADAPT_WINDOW:
            # si no mejoró el mejor en la ventana, subir pmut; si sí, bajar hacia base
            if max(best_history[-ADAPT_WINDOW:]) < best_ever:
                pmut = min(PMUT_MAX, pmut + ADAPT_STEP)
            else:
                pmut = max(PMUT_BASE, pmut - ADAPT_STEP / 2)

        # Emparejamiento robusto (evita hermanos/primos)
        parejas = emparejar_sin_familia(poblacion)

        # Reproducción (2 hijos por pareja) + mutación
        hijos: List[Individuo] = []
        for p1, p2 in parejas:
            h1, h2 = cruzar(p1, p2)
            mutar(h1, pmut); mutar(h2, pmut)
            hijos.append(h1); hijos.append(h2)

        # Reemplazo con elitismo bajo
        if ELITISM > 0:
            elites = sorted(poblacion, key=fitness, reverse=True)[:ELITISM]
            hijos.sort(key=fitness)  # peores primero
            for i in range(min(ELITISM, len(hijos))):
                hijos[i] = elites[i]

        # Inmigración ocasional para  + diversidad:
        # if gen % 40 == 0:
        #     for _ in range(2):
        #         hijos[ random.randrange(len(hijos)) ] = individuo_aleatorio()

        poblacion = hijos

    print("\nNo se encontró individuo perfecto.")
    poblacion.sort(key=fitness, reverse=True)
    print("Mejor hasta ahora:", poblacion[0].id, poblacion[0].genes, "| fitness =", poblacion[0].fitness())
    return None, poblacion[0]

if __name__ == "__main__":
    ejecutar()
