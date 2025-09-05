import random


# Parámetros

NUM_PAREJAS = 50          # 50 parejas = 100 individuos
NUM_GENES = 20            # cada individuo tiene 20 genes
PROB_MUT = 0.2            # probabilidad de mutación (20%)
MAX_GEN = 500000             # número máximo de generaciones
VALOR_OBJETIVO = [9]*NUM_GENES  # individuo "perfecto"



# Estructuras globales
id_counter = 0             # contador para dar id únicos
padres = {}                # diccionario: id -> (padre, madre)


# Generación de individuos

def nuevo_id():
    global id_counter
    id_counter += 1
    return id_counter

def generar_individuo():
    ind = {
        "id": nuevo_id(),
        "genes": [random.randint(0, 9) for _ in range(NUM_GENES)]
    }
    padres[ind["id"]] = (None, None)  # sin padres en la generación inicial
    return ind

def generar_poblacion():
    return [generar_individuo() for _ in range(NUM_PAREJAS*2)]

# Función de aptitud 

def evaluar(individuo):
    return sum(1 for i, g in enumerate(individuo["genes"]) if g == VALOR_OBJETIVO[i])

def es_perfecto(individuo):
    return individuo["genes"] == VALOR_OBJETIVO

# Verificación de parentesco

def son_parientes(ind1, ind2):
    p1 = padres[ind1["id"]]
    p2 = padres[ind2["id"]]

    # Hermanos: comparten al menos un padre
    if p1[0] is not None and (p1[0] in p2 or p1[1] in p2):
        return True
    if p1[1] is not None and (p1[1] in p2):
        return True

    # Primos: los padres de uno son hermanos de los padres del otro
    padres1 = [x for x in p1 if x is not None]
    padres2 = [x for x in p2 if x is not None]

    for pa1 in padres1:
        for pa2 in padres2:
            if pa1 is not None and pa2 is not None:
                abu1 = padres[pa1]
                abu2 = padres[pa2]
                # Si comparten abuelo → entonces son primos
                if abu1[0] is not None and (abu1[0] in abu2 or abu1[1] in abu2):
                    return True
                if abu1[1] is not None and (abu1[1] in abu2):
                    return True
    return False


# Reproducción 
def reproducir(padre, madre):
    punto = random.randint(1, NUM_GENES-2)
    hijo1_genes = padre["genes"][:punto] + madre["genes"][punto:]
    hijo2_genes = madre["genes"][:punto] + padre["genes"][punto:]

    hijo1 = {"id": nuevo_id(), "genes": hijo1_genes}
    hijo2 = {"id": nuevo_id(), "genes": hijo2_genes}

    # Guardar genealogía
    padres[hijo1["id"]] = (padre["id"], madre["id"])
    padres[hijo2["id"]] = (padre["id"], madre["id"])

    return hijo1, hijo2


# Mutación
def mutar(individuo, prob=PROB_MUT):
    if random.random() < prob:
        pos = random.randint(0, NUM_GENES-1)
        cambio = random.choice([-1, 1])
        individuo["genes"][pos] = max(0, min(9, individuo["genes"][pos] + cambio))
    return individuo


# Algoritmo genético principal

def algoritmo_genetico():
    poblacion = generar_poblacion()

    for generacion in range(1, MAX_GEN+1):
        # Evaluar aptitud
        mejor = max(poblacion, key=evaluar)
        print(f"Generación {generacion} | Mejor: {mejor['genes']} | Fitness: {evaluar(mejor)}")

        # Revisar perfección
        if es_perfecto(mejor):
            print("¡Individuo perfecto encontrado en generación", generacion, "!")
            break

        # Nueva población
        nueva_poblacion = []
        while len(nueva_poblacion) < len(poblacion):
            padre, madre = random.sample(poblacion, 2)
            if son_parientes(padre, madre):
                continue  # evitar hermanos/primos

            hijo1, hijo2 = reproducir(padre, madre)
            hijo1 = mutar(hijo1)
            hijo2 = mutar(hijo2)
            nueva_poblacion.extend([hijo1, hijo2])

        poblacion = nueva_poblacion


algoritmo_genetico()
