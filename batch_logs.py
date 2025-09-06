#!/usr/bin/env python3
#
# Qué hace:
# - Ejecuta la simulación N veces (por defecto 50).
# - Usa una semilla distinta en cada corrida.
# - Captura TODA la salida que el módulo imprime (verbose=True) y la guarda en
#   un .txt por corrida dentro de una carpeta.
# - Escribe un summary.csv con: run, seed, found, generations, log_path
#
# Uso:
#   python3 batch_logs.py --modulo genetica --intentos 100 --max-generaciones 15 --outdir runs_100


import argparse
import importlib
import random
import secrets
import os
import io
import csv
import contextlib
from typing import Optional

def main():
    parser = argparse.ArgumentParser(description="Correr el algoritmo genético muchas veces y guardar las impresiones en archivos.")
    parser.add_argument("--modulo", required=True, help="Nombre del módulo algoritmo genético (sin .py), genetica")
    parser.add_argument("--intentos", type=int, default=50, help="Número de corridas (default: 100)")
    parser.add_argument("--max-generaciones", type=int, default=1000, help="Límite de generaciones por corrida (default: 15)")
    parser.add_argument("--outdir", type=str, default="runs", help="Carpeta de salida para logs y summary.csv (default: runs)")
    parser.add_argument("--semilla-base", type=int, default=None, help="Si se da, usa semilla base+i para la corrida i (en vez de secrets)")
    args = parser.parse_args()

    # Importación del módulo del algoritmo genético
    try:
        ga = importlib.import_module(args.modulo)
    except Exception as e:
        raise SystemExit(f"No pude importar el módulo '{args.modulo}'. Error: {e}")

    # Verificar que exista la función simular(max_generaciones, verbose)
    if not hasattr(ga, "simular"):
        raise SystemExit("El módulo no tiene la función simular(max_generaciones=..., verbose=...)")

    # Preparar carpeta de salida
    os.makedirs(args.outdir, exist_ok=True)
    summary_path = os.path.join(args.outdir, "summary.csv")

    # Correr intentos
    rows = []
    for i in range(1, args.intentos + 1):
        # Semilla por corrida
        seed = (args.semilla_base + i - 1) if args.semilla_base is not None else secrets.randbits(64)
        random.seed(seed)

        # Capturar toda la salida del módulo
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                perfecto, gen, _poblaciones = ga.simular(max_generaciones=args.max_generaciones, verbose=True)
        except Exception as e:
            # En caso de error, guardar el traceback en el log
            contenido = buffer.getvalue()
            log_path = os.path.join(args.outdir, f"run_{i:03d}.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"[RUN #{i}] seed={seed}\n")
                f.write("ERROR DURANTE LA CORRIDA:\n")
                f.write(repr(e) + "\n\n")
                f.write("=== SALIDA CAPTURADA ANTES DEL ERROR ===\n")
                f.write(contenido)
            rows.append({"run": i, "seed": seed, "found": False, "generations": "", "log_path": log_path, "error": repr(e)})
            continue

        # Guardar el log completo de esa corrida
        contenido = buffer.getvalue()
        log_path = os.path.join(args.outdir, f"run_{i:03d}.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"[RUN #{i}] seed={seed}\n")
            f.write(contenido)

        found = (perfecto is not None)
        rows.append({"run": i, "seed": seed, "found": found, "generations": gen if found else "", "log_path": log_path, "error": ""})

    # Escribir el resumen CSV
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run", "seed", "found", "generations", "log_path", "error"])
        w.writeheader()
        w.writerows(rows)

    # Mensaje final: dónde quedó todo
    print(f"\nListo. Guardé {len(rows)} logs en: {os.path.abspath(args.outdir)}")
    print(f"Resumen: {os.path.abspath(summary_path)}")
    # Sugerencia: mostrar el mejor (si hubo alguno)
    exitos = [r for r in rows if r["found"]]
    if exitos:
        best = min(exitos, key=lambda r: int(r["generations"]))
        print(f"Mejor corrida: run #{best['run']}  seed={best['seed']}  generaciones={best['generations']}")
        print(f"Log: {best['log_path']}")
    else:
        print("No se encontró el individuo perfecto dentro del límite de generaciones en estas corridas.")

if __name__ == "__main__":
    main()