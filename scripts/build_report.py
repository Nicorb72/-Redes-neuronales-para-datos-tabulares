"""Genera el blog desde Markdown y tablas derivadas de artefactos guardados."""
import argparse
import html
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlparse

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/technical-report'


def table(title, header, rows):
    return title + '\n\n| ' + ' | '.join(header) + ' |\n|' + '|'.join(['---'] * len(header)) + '|\n' + '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in rows) + '\n'


def evidence_blocks():
    evidence = json.loads((REPORT / 'evidence.json').read_text(encoding='utf-8'))
    for path, expected in evidence['sources'].items():
        artifact = ROOT / path
        payload = artifact.read_bytes()
        if artifact.suffix.lower() in {'.json', '.csv'}:
            payload = payload.replace(b'\r\n', b'\n')
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f'Evidencia desactualizada: {path}; ejecutar report_evidence.py.')
    pd = json.loads((ROOT / 'reports/pd_evaluation/resultados.json').read_text(encoding='utf-8'))
    lgd = json.loads((ROOT / 'artifacts/models/lgd_benchmark_results.json').read_text(encoding='utf-8'))
    rows = []
    for name, label in [('auc', 'AUC'), ('gini', 'Gini'), ('ks', 'KS'), ('brier', 'Brier'), ('log_loss', 'Log-loss')]:
        stats = evidence['pd_test_30_seeds'][name]
        higher = name in {'auc', 'gini', 'ks'}
        rows.append([label, f"{pd['logistic']['test'][name]:.5f}", f"{stats['mean']:.5f} ± {stats['std']:.5f}", f"{stats['max' if higher else 'min']:.5f}", f"{stats['min' if higher else 'max']:.5f}"])
    blocks = {'PD_METRICS': table('**Tabla 4. PD en prueba temporal 2014, treinta semillas.** Fuente: `metricas_por_semilla.csv` y `resultados.json`, derivados en [evidence.json](evidence.json).', ['Métrica', 'Logística', 'MLP media ± DE', 'MLP mejor', 'MLP peor'], rows)}
    stats = lgd['mlp_30_semillas']
    base = lgd['baseline']
    blocks['LGD_METRICS'] = table('**Tabla 5. Resultados LGD respaldados por el JSON versionado.** Fuente: [lgd_benchmark_results.json](../../artifacts/models/lgd_benchmark_results.json). La DE corresponde a las cinco corridas registradas.', ['Modelo', 'Parámetros', 'MAE media ± DE', 'MAE mín–máx', 'RMSE media ± DE', 'RMSE mín–máx'], [
        ['Lineal recortado', base['parametros'], f"{base['test_mae']:.5f}", '—', f"{base['test_rmse']:.5f}", '—'],
        [f"MLP ({stats['n_semillas']} semillas)", stats['parametros'], f"{stats['mae']['media']:.5f} ± {stats['mae']['std']:.5f}", f"{stats['mae']['min']:.5f}–{stats['mae']['max']:.5f}", f"{stats['rmse']['media']:.5f} ± {stats['rmse']['std']:.5f}", f"{stats['rmse']['min']:.5f}–{stats['rmse']['max']:.5f}"]])
    blocks['COMPUTE'] = table('**Tabla 7. Presupuesto y complejidad por familia.** Fuente: resumen de búsqueda PD, treinta corridas PD y JSON LGD versionados.', ['Familia / fase', 'Configuraciones o corridas', 'Tiempo registrado', 'Parámetros finales'], [
        ['Logística PD', '1 ajuste', f"{pd['logistic']['fit_seconds']:.3f} s", '45'],
        ['MLP PD: búsqueda', '180 ensayos + 25 corridas de robustez', '1.994,85 s oficiales', '7.105'],
        ['MLP PD: evaluación final', '30 semillas de una configuración', f"{evidence['pd_fit_seconds_total']:.2f} s de ajuste", '7.105'],
        ['Lineal LGD', '1 ajuste', f"{base['tiempo_ajuste_sec']:.3f} s", base['parametros']],
        ['MLP LGD: archivo guardado', f"{lgd['busqueda_hiperparametros']['presupuesto_configuraciones']} configuraciones + {stats['n_semillas']} semillas", f"{lgd['busqueda_hiperparametros']['tiempo_total_busqueda_sec']:.2f} s búsqueda + {stats['tiempo_ajuste_sec']['total']:.2f} s ajustes", stats['parametros']]])
    rows = []
    for i, item in enumerate(evidence['hypothesis_scenarios'], 1):
        direction = 'menor' if item['mean_pd_change_pp'] < 0 else 'mayor'
        rows.append([f"H{i}: `{item['variable']}`", f"{item['q25_raw']:,.2f} → {item['q75_raw']:,.2f}", f"{item['mean_pd_change_pp']:+.3f}", f"{item['logistic_coefficient_standardized']:+.4f}", f"Compatible: PD media {direction}; no causal"])
    blocks['HYPOTHESES'] = table('**Tabla 8. Contraste de hipótesis mediante escenarios pareados P25→P75.** Fuente: inferencia nueva del checkpoint final, validación 2013, 5.000 filas, semilla 42; [evidence.json](evidence.json). Cambio PD en puntos porcentuales; coeficiente logístico por unidad estandarizada.', ['Hipótesis', 'Valores en unidades originales', 'Cambio medio PD (pp)', 'Coeficiente logístico', 'Conclusión acotada'], rows)
    return blocks


def refresh(source, check):
    for key, body in evidence_blocks().items():
        pattern = rf'<!-- BEGIN {key} -->.*?<!-- END {key} -->'
        replacement = f'<!-- BEGIN {key} -->\n{body}<!-- END {key} -->'
        if len(re.findall(pattern, source, re.S)) != 1:
            raise ValueError(f'Bloque ausente o repetido: {key}')
        updated = re.sub(pattern, lambda _: replacement, source, flags=re.S)
        if check and updated != source:
            raise ValueError(f'Tabla desactualizada: {key}; ejecutar --refresh-evidence.')
        source = updated
    return source


def render(source):
    maths = []
    def protect(match):
        display = match.group(1) is not None
        value = match.group(1) if display else match.group(2)
        token = f'REPORTMATHTOKEN{len(maths)}END'
        body = html.escape(value.strip())
        markup = '<div class="math-display">\\[' + body + '\\]</div>' if display else '<span class="math-inline">\\(' + body + '\\)</span>'
        maths.append((token, markup, display))
        return '\n\n' + token + '\n\n' if display else token
    # No interpretar $env ni otros símbolos dentro de bloques de código.
    chunks = re.split(r'(```.*?```)', source, flags=re.S)
    for index in range(0, len(chunks), 2):
        chunks[index] = re.sub(r'\$\$(.*?)\$\$|\$([^$\n]+)\$', protect, chunks[index], flags=re.S)
    body = MarkdownIt('commonmark', {'html': True}).enable('table').render(''.join(chunks))
    for token, markup, display in maths:
        body = body.replace(f'<p>{token}</p>', markup) if display else body.replace(token, markup)
    body = re.sub(r'<p>(<strong>Tabla \d+\..*?)</p>\s*<table>', lambda m: '<table><caption>' + m.group(1) + '</caption>', body, flags=re.S)
    body = re.sub(r'<table>.*?</table>', lambda m: '<div class="table-wrap">' + m.group() + '</div>', body, flags=re.S)
    entries = []
    def heading(match):
        index = len(entries) + 1
        title = match.group(1)
        entries.append(f'<li><a href="#seccion-{index}">{title}</a></li>')
        return f'<h2 id="seccion-{index}">{title}</h2>'
    body = re.sub(r'<h2>(.*?)</h2>', heading, body)
    nav = '<nav aria-label="Contenido"><strong>Contenido del reporte</strong><ol>' + ''.join(entries) + '</ol></nav>'
    body = re.sub(r'(<h2 id="seccion-1">)', lambda m: nav + m.group(1), body, count=1)
    return (REPORT / 'template.html').read_text(encoding='utf-8').replace('<!-- REPORT_BODY -->', body)


def validate(document):
    for target in re.findall(r'(?:href|src)="([^"]+)"', document):
        target = html.unescape(target)
        url = urlparse(target)
        if url.scheme or url.netloc or not url.path:
            continue
        if not (REPORT / unquote(url.path)).resolve().exists():
            raise ValueError(f'Referencia local inexistente: {target}')
    for label, pattern in [('Tabla', r'<caption>\s*<strong>Tabla (\d+)\.'), ('Figura', r'<figcaption>\s*<strong>Figura (\d+)\.')]:
        numbers = list(map(int, re.findall(pattern, document)))
        if numbers != list(range(1, len(numbers) + 1)):
            raise ValueError(f'Numeración no secuencial: {label} {numbers}')
    if 'REPORTMATHTOKEN' in document:
        raise ValueError('Fórmula sin restaurar.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Comprueba sin escribir archivos.')
    parser.add_argument('--refresh-evidence', action='store_true', help='Actualiza tablas desde evidencia guardada.')
    args = parser.parse_args()
    if args.check and args.refresh_evidence:
        parser.error('--check y --refresh-evidence son excluyentes.')
    path = REPORT / 'index.md'
    source = path.read_text(encoding='utf-8')
    source = refresh(source, check=not args.refresh_evidence)
    document = render(source)
    validate(document)
    target = REPORT / 'index.html'
    if args.check:
        if target.read_text(encoding='utf-8') != document:
            sys.exit('HTML desactualizado: ejecutar python scripts/build_report.py.')
        print('OK: HTML sincronizado, tablas derivadas y referencias locales válidas.')
    else:
        if args.refresh_evidence:
            path.write_text(source, encoding='utf-8')
        target.write_text(document, encoding='utf-8')
        print(target)


if __name__ == '__main__':
    main()
