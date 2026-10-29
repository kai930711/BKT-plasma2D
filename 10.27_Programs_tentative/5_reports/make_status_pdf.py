from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem, KeepTogether

FONT = '/System/Library/Fonts/Supplemental/'
pdfmetrics.registerFont(TTFont('Arial', FONT + 'Arial.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Bold', FONT + 'Arial Bold.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Italic', FONT + 'Arial Italic.ttf'))
pdfmetrics.registerFontFamily('Arial', normal='Arial', bold='Arial-Bold', italic='Arial-Italic', boldItalic='Arial-Bold')

OUT = '/Users/kai/BKT-plasma2D/10.27_Wide_SQQ_Scan/Status_and_Limitations.pdf'
INK, MUTED, ACCENT, RULE = colors.HexColor('#1d2330'), colors.HexColor('#5b6475'), colors.HexColor('#7a2e8f'), colors.HexColor('#cfd3dc')

title = ParagraphStyle('t', fontName='Arial-Bold', fontSize=19, leading=28, textColor=INK, spaceAfter=6)
sub = ParagraphStyle('s', fontName='Arial', fontSize=9.5, leading=13, textColor=MUTED, spaceAfter=10)
h1 = ParagraphStyle('h1', fontName='Arial-Bold', fontSize=13, leading=17, textColor=ACCENT, spaceBefore=12, spaceAfter=4, keepWithNext=1)
h2 = ParagraphStyle('h2', fontName='Arial-Bold', fontSize=10.5, leading=14, textColor=INK, spaceBefore=7, spaceAfter=2, keepWithNext=1)
body = ParagraphStyle('b', fontName='Arial', fontSize=10, leading=14, textColor=INK, spaceAfter=3)
cell = ParagraphStyle('c', parent=body, fontSize=9, leading=12, spaceAfter=0)
cellb = ParagraphStyle('cb', parent=cell, fontName='Arial-Bold')


def bullets(items):
    return ListFlowable([ListItem(Paragraph(t, body), leftIndent=14, bulletColor=ACCENT) for t in items],
                        bulletType='bullet', bulletFontSize=7, leftIndent=14, bulletOffsetY=-1)


s = []
s.append(Paragraph('BKT and liquid–gas transitions from S<sub>QQ</sub>', title))
s.append(Paragraph('Progress and limitations, 28 October 2026. 2D two-component log plasma, Monte Carlo.', sub))

s.append(Paragraph('Where we are', h1))
s.append(Paragraph('Done', h2))
s.append(bullets([
    '<b>S<sub>QQ</sub> pipeline at N = 50.</b> The 9.21 scan covers 10 densities, T* from 0.15 to 0.35 and E0* from 0 to 0.5, with 121 runs per density.',
    '<b>Wide pilot at ρ* = 0.05.</b> T* runs from 0.05 to 0.55 and E0* from 0 to 1.0, with hot and cold starts on a coarse 6 × 5 grid.',
    '<b>Two transition lines, both defined from S<sub>QQ</sub> alone.</b> The BKT line is where 1/ε crosses 4T*. '
    'The liquid–gas line is the E0* step where 1/ε(k<sub>min</sub>) rises fastest at fixed T*. At ρ* = 0.05 the second line lands on the same step as the steepest S<sub>NN</sub> jump.',
    '<b>Interactive page,</b> published as a new artifact: box animation, state table and cut charts.',
    '<b>Cross-checks.</b> The pilot S<sub>QQ</sub>(k) was compared with the reference figure. The potential was confirmed from the code: log-Coulomb, plus the cosh attraction, plus a hard core, with K = 1 in code units.',
    '<b>New data received.</b> N = 100 and N = 200 are in Downloads/data_n100_200. They are surveyed and complete, but not yet processed.',
]))

s.append(Paragraph('Data in hand', h2))
rows = [
    [Paragraph(x, cellb) for x in ('Set', 'N', 'Density ρ*', 'E0*', 'T*', 'Runs')],
    [Paragraph(x, cell) for x in ('9.21 scan', '50', '10 values, 0.001 to 0.2', '0 to 0.5, step 0.05', '0.15 to 0.35, step 0.02', '121 per density')],
    [Paragraph(x, cell) for x in ('10.27 pilot', '50', '0.05', '0 to 1.0, 5 values', '0.05 to 0.55, 6 values', '60 (hot and cold)')],
    [Paragraph(x, cell) for x in ('New: original', '100 and 200', '10 values, one batch each', '0 to 0.5, step 0.05', '0.15 to 0.35, step 0.02', '121 per batch')],
    [Paragraph(x, cell) for x in ('New: bkt', '100 and 200', '0.001 only', '0 to 0.15, step 0.05', '0.15 to 0.35, step 0.01', '84 each')],
]
t = Table(rows, colWidths=[0.95 * inch, 0.8 * inch, 1.5 * inch, 1.35 * inch, 1.45 * inch, 1.0 * inch], repeatRows=1)
t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#eceef3')), ('LINEBELOW', (0, 0), (-1, -1), 0.4, RULE),
                       ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4)]))
s.append(t)
s.append(Spacer(1, 4))
s.append(Paragraph('Disk size: N = 50 is 1.5 GB, N = 100 is 2.1 GB and N = 200 is 4.0 GB unpacked (0.8 GB and 1.2 GB as tar.gz).', body))

s.append(Paragraph('Limitations', h1))
s.append(Paragraph('Data', h2))
s.append(bullets([
    '<b>Nothing below T* = 0.15 is usable.</b> The pilot’s T* = 0.05 row is unequilibrated. This is where the BKT and liquid–gas lines should meet, so the junction is unresolved.',
    '<b>The new N = 100/200 data does not fix that.</b> It has the same T* range as before, so it only refines lines we already have.',
    '<b>The grid is coarse.</b> The steps are 0.05 in E0* and 0.02 in T*, so each transition is a bracket, not a located point. The pilot is coarser still.',
]))
s.append(Paragraph('BKT line', h2))
s.append(bullets([
    '<b>Most drawn points are not real BKT points.</b> At ρ* ≥ 0.03 there are no filled (homogeneous) points. Beyond E0* ≈ 0.30 every crossing is inside a droplet, except at ρ* = 0.001.',
    '<b>The current page still draws them.</b> It has not been changed yet. The fix is to cap the line at E0* ≈ 0.30 and draw only the filled points.',
]))
s.append(Paragraph('Liquid–gas line', h2))
s.append(bullets([
    '<b>It is a location of fastest change, not proof of a first-order transition.</b> Confirming one needs bimodality or hysteresis, or the S<sub>NN</sub> Porod tail.',
    '<b>The finite box gives a rise, not the bulk drop.</b> The condensed phase is a neutral droplet, so 1/ε(k<sub>min</sub>) rises. The bulk liquid in the reference figure drops instead.',
    '<b>Overscreening is not reproduced.</b> The dip of 1/ε below 0 near ka ≈ 3.4 did not appear. The minimum is about 0.6 to 0.7.',
    '<b>Low density is unreliable.</b> At ρ* ≤ 0.01 the lines are scattered.',
]))
s.append(Paragraph('Method', h2))
s.append(bullets([
    '<b>The shape classifier’s cutoff is soft.</b> The 0.4 cutoff between crossover and condensed is not sharp.',
    '<b>Hot and cold starts disagree in places.</b> In the pilot, 6 of 30 cells differ by more than 0.15 in 1/ε.',
]))
s.append(Paragraph('Physics still open', h2))
s.append(bullets([
    '<b>The boundary condition is unchecked.</b> The Ewald dipole term fixes it. It has not been compared with Elio’s group (vacuum versus tin-foil).',
    '<b>K = 1 is inferred from how the code uses T*,</b> not written anywhere in the code. The BKT point at T* ≈ 0.24 supports it.',
]))

s.append(Paragraph('Pending notebook fixes (9.21_Density Correlation)', h1))
s.append(bullets([
    'VS Code needs “Revert File” for S_NN_Liquid_Gas.ipynb and Dielectric_Phase_Diagram.ipynb.',
    'The Porod text in section 4 is wrong.',
    'Step A, S(k) out to kσ ≈ 10, is waiting for approval.',
    'The colour scale should be centred at 4, not 3.',
]))

s.append(Paragraph('Next', h1))
s.append(Paragraph('The most valuable step is the low-T* scan. The choices still needed are the densities, the T* and E0* ranges, the run length, and hot and cold starts. '
                   'Processing N = 100 and 200 is optional.', body))


def footer(c, d):
    c.saveState(); c.setFont('Arial', 8); c.setFillColor(MUTED)
    c.drawString(0.8 * inch, 0.5 * inch, 'BKT-plasma2D · progress and limitations')
    c.drawRightString(letter[0] - 0.8 * inch, 0.5 * inch, f'Page {d.page}')
    c.restoreState()


kept, i = [], 0
while i < len(s):                                   # bind every heading to the flowable after it
    if isinstance(s[i], Paragraph) and s[i].style.name in ('h1', 'h2') and i + 1 < len(s):
        kept.append(KeepTogether([s[i], s[i + 1]])); i += 2
    else:
        kept.append(s[i]); i += 1
s = kept

doc = SimpleDocTemplate(OUT, pagesize=letter, leftMargin=0.8 * inch, rightMargin=0.8 * inch, topMargin=0.75 * inch, bottomMargin=0.8 * inch,
                        title='Progress and limitations', author='kai930711')
doc.build(s, onFirstPage=footer, onLaterPages=footer)
print('wrote', OUT)
