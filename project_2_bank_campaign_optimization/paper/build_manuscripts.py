"""Typeset two manuscripts from one checked prose source and repository outputs."""
from pathlib import Path
import re
import pandas as pd
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "outputs"
SRC = (HERE / "manuscript_source.md").read_text()
info = pd.read_csv(OUT / "information_set_comparison.csv")
info_pairs = pd.read_csv(OUT / "information_set_pairwise_bootstrap.csv")
leakage = pd.read_csv(OUT / "leakage_audit.csv")
selection = pd.read_csv(OUT / "model_selection_summary.csv")
models = pd.read_csv(OUT / "model_metrics.csv")
budget = pd.read_csv(OUT / "budget_metrics.csv")

def set_cell(cell, value):
    cell.text = str(value)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    for p in cell.paragraphs:
        p.paragraph_format.space_after = Pt(0)
        for run in p.runs:
            run.font.name = 'Times New Roman'
            run.font.size = Pt(12)
    tcPr = cell._tc.get_or_add_tcPr()
    mar = OxmlElement('w:tcMar')
    for edge in ('top', 'left', 'bottom', 'right'):
        x = OxmlElement('w:'+edge)
        x.set(qn('w:w'), '55' if edge in ('top','bottom') else '70')
        x.set(qn('w:type'), 'dxa')
        mar.append(x)
    tcPr.append(mar)

def set_cell_width(cell, inches):
    tcPr = cell._tc.get_or_add_tcPr()
    tcW = tcPr.first_child_found_in("w:tcW")
    if tcW is None:
        tcW = OxmlElement('w:tcW')
        tcPr.append(tcW)
    tcW.set(qn('w:w'), str(int(inches * 1440)))
    tcW.set(qn('w:type'), 'dxa')

def add_table(doc, caption, headers, rows, widths):
    cp=doc.add_paragraph(style='Caption')
    cp.add_run(caption)
    cp.paragraph_format.keep_with_next=True
    t=doc.add_table(rows=1, cols=len(headers))
    t.autofit=False
    for i,h in enumerate(headers):
        t.columns[i].width=Inches(widths[i])
        set_cell(t.rows[0].cells[i],h)
        set_cell_width(t.rows[0].cells[i], widths[i])
        for r in t.rows[0].cells[i].paragraphs[0].runs: r.bold=True
    for row in rows:
        cells=t.add_row().cells
        for i,v in enumerate(row):
            set_cell(cells[i],v)
            set_cell_width(cells[i], widths[i])
    for tr in t.rows:
        for cell in tr.cells:
            tcPr=cell._tc.get_or_add_tcPr()
            borders=OxmlElement('w:tcBorders')
            for side in ('top','bottom','left','right'):
                el=OxmlElement('w:'+side)
                el.set(qn('w:val'),'single');el.set(qn('w:sz'),'3');el.set(qn('w:color'),'D9D9D9')
                borders.append(el)
            tcPr.append(borders)
    for tr in t.rows[:-1]:
        for cell in tr.cells:
            for p in cell.paragraphs: p.paragraph_format.keep_with_next=True
    t.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
    doc.add_paragraph().paragraph_format.space_after=Pt(0)

def exhibit(doc, marker):
    if marker=='[TABLE 1]':
        levels = {r.Information_Set:r for _,r in info.iterrows()}
        invalid = leakage[leakage.Deployable == False].iloc[0]
        def pair(ref, comp, metric):
            r=info_pairs[(info_pairs.Reference_Set==ref)&(info_pairs.Comparison_Set==comp)&(info_pairs.Metric==metric)].iloc[0]
            return r.Estimate, r.CI95_Lower, r.CI95_Upper
        rows=[]
        rows.append(['Strict planning','List selection',f"{levels['Strict planning (primary)'].Holdout_PR_AUC:.3f}",f"{100*levels['Strict planning (primary)'].Top20_Capture:.1f}%",'Reference'])
        specs=[
            ('Planning + macro','Planning plus macro context','Planning if release vintages available','Strict planning (primary)'),
            ('Operational pre-contact','Operational pre-contact','Scheduled, before call','Strict planning (primary)'),
            ('Operational + macro','Operational plus macro context','Later pre-call decision','Strict planning (primary)'),
        ]
        for label,key,status,ref in specs:
            pr=pair(ref,key,'PR_AUC'); cap=pair(ref,key,'Top20_Capture')
            rows.append([label,status,f"{levels[key].Holdout_PR_AUC:.3f}",f"{100*levels[key].Top20_Capture:.1f}%",f"+{pr[0]:.3f} [{pr[1]:.3f}, {pr[2]:.3f}]\n+{100*cap[0]:.1f} pp [{100*cap[1]:.1f}, {100*cap[2]:.1f}]"])
        key='Invalid: operational plus macro plus duration'; ref='Operational plus macro context'
        pr=pair(ref,key,'PR_AUC'); cap=pair(ref,key,'Top20_Capture')
        rows.append(['Invalid + duration','After focal call',f"{invalid.PR_AUC:.3f}",f"{100*invalid.Top20_Capture:.1f}%",f"+{pr[0]:.3f} [{pr[1]:.3f}, {pr[2]:.3f}]\n+{100*cap[0]:.1f} pp [{100*cap[1]:.1f}, {100*cap[2]:.1f}]"])
        add_table(doc,'Table 1. Decision-time information sets and paired performance differences',
            ['Information set','Decision status','PR-AUC','Top-20% capture','Paired ΔPR-AUC; Δcapture [95% CI]'],rows,[1.25,1.25,.70,.85,2.45])
        p=doc.add_paragraph('Note. Valid specifications are compared with strict planning; the invalid duration specification is compared with operational plus macro. Differences are comparison minus reference from 1,000 paired bootstrap resamples. pp = percentage points.');p.style='Note'
    elif marker=='[TABLE 2]':
        s={r.Model:r for _,r in selection.iterrows()}
        m={r.Model:r for _,r in models.iterrows()}
        names=['Dummy Baseline','Logistic Regression','Random Forest','Histogram Gradient Boosting','Random Forest (isotonic calibration)']
        rows=[]
        for n in names:
            cv = '—' if n not in s else f'{s[n].PR_AUC_Mean:.3f}'
            x=m[n]
            rows.append([n,cv,f'{x.PR_AUC:.3f}',f'{x.ROC_AUC:.3f}',f'{x.Brier_Score:.3f}'])
        add_table(doc,'Table 2. Training selection and development-holdout results',
            ['Model','CV PR-AUC','Holdout PR-AUC','ROC-AUC','Brier score'],rows,[2.40,.95,1.20,.85,1.10])
        p=doc.add_paragraph('Note. CV denotes five-fold training-only mean. Uncalibrated class-weighted models have poorly calibrated raw probabilities. Isotonic calibration is chosen on training data and defines the primary model.')
        p.style='Note'
    elif marker=='[TABLE 3]':
        rows=[]
        for q in [.1,.2,.3,.4]:
            r=budget.iloc[(budget.contact_share-q).abs().argmin()]
            rows.append([f'{q:.0%}',f'{int(r.contact_records):,}',str(int(r.subscribers_captured)),f'{100*r.responder_capture:.1f}%',f'{r.lift:.2f}',f'{r.contacts_per_observed_subscriber:.2f}'])
        add_table(doc,'Table 3. Observed targeting outcomes for strict planning model',
            ['Budget','Contacts','Observed\nsubscribers','Capture','Lift','Contacts per\nobserved subscriber'],rows,[.85,1.05,1.15,1.05,.65,1.75])
        p=doc.add_paragraph('Note. All figures describe observed holdout contact records. At the 20% budget, random selection expects 185.7 observed subscribers.');p.style='Note'
    elif marker.startswith('[FIGURE'):
        n=int(marker[8])
        name={1:'information_set_and_leakage.png',2:'cumulative_gains.png'}[n]
        caption={1:'Figure 1. Apparent PR-AUC under different information sets; the duration model is invalid for pre-contact selection.',2:'Figure 2. Cumulative observed subscriber capture under strict planning, compared with random selection.'}[n]
        p=doc.add_paragraph(style='Caption');p.add_run(caption);p.paragraph_format.keep_with_next=True
        pic=doc.add_paragraph();pic.alignment=WD_ALIGN_PARAGRAPH.CENTER
        pic.paragraph_format.space_after=Pt(5)
        pic.add_run().add_picture(str(OUT/'charts'/name),width=Inches(6.15))

def emit_inline(p, s):
    for part in re.split(r'(\*\*.*?\*\*|`[^`]+`|\*[^*]+\*)',s):
        if not part: continue
        bold=part.startswith('**') and part.endswith('**')
        italic=part.startswith('*') and part.endswith('*') and not bold
        text=part[2:-2] if bold else part[1:-1] if italic or part.startswith('`') else part
        run=p.add_run(text);run.bold=bold;run.italic=italic

def build(master):
    doc=Document()
    sec=doc.sections[0]
    sec.page_width=Inches(8.5);sec.page_height=Inches(11)
    sec.top_margin=sec.bottom_margin=Inches(1)
    sec.left_margin=sec.right_margin=Inches(1)
    styles=doc.styles
    norm=styles['Normal'];norm.font.name='Times New Roman';norm.font.size=Pt(12)
    norm.paragraph_format.line_spacing=1.0
    norm.paragraph_format.space_after=Pt(0 if master else 3)
    norm.paragraph_format.first_line_indent=Inches(.18)
    for key in ['Title','Heading 1','Caption']:
        st=styles[key];st.font.name='Times New Roman';st.font.color.rgb=None
    styles['Title'].font.size=Pt(14);styles['Title'].font.bold=True
    styles['Title'].paragraph_format.space_after=Pt(6)
    # Word's built-in Title style can carry a blue bottom rule through its template.
    stpr=styles['Title']._element.get_or_add_pPr()
    for old in stpr.findall(qn('w:pBdr')): stpr.remove(old)
    border=OxmlElement('w:pBdr')
    bottom=OxmlElement('w:bottom');bottom.set(qn('w:val'),'nil');border.append(bottom)
    stpr.append(border)
    styles['Heading 1'].font.size=Pt(12);styles['Heading 1'].font.bold=True
    styles['Heading 1'].paragraph_format.space_before=Pt(7)
    styles['Heading 1'].paragraph_format.space_after=Pt(3)
    styles['Caption'].font.size=Pt(12);styles['Caption'].font.bold=True
    styles['Caption'].paragraph_format.space_before=Pt(4)
    styles['Caption'].paragraph_format.space_after=Pt(3)
    if 'Note' not in [x.name for x in styles]: styles.add_style('Note',1)
    styles['Note'].font.name='Times New Roman';styles['Note'].font.size=Pt(12)
    styles['Note'].paragraph_format.space_after=Pt(5)
    doc.core_properties.title='Decision Time Information and the Validity of Response Models for Constrained Marketing'
    doc.core_properties.author='Rong Zhao' if master else ''
    doc.core_properties.last_modified_by='Rong Zhao' if master else ''
    doc.core_properties.comments=''
    paras=SRC.split('\n\n')
    in_references=False
    for raw in paras:
        line=raw.strip()
        if not line: continue
        if line.startswith('# '):
            p=doc.add_paragraph(style='Title');p.alignment=WD_ALIGN_PARAGRAPH.CENTER;emit_inline(p,line[2:])
            border=OxmlElement('w:pBdr');bottom=OxmlElement('w:bottom');bottom.set(qn('w:val'),'nil');border.append(bottom);p._element.get_or_add_pPr().append(border)
            if master:
                p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.first_line_indent=Inches(0);p.paragraph_format.space_after=Pt(8)
                p.add_run('Rong Zhao\nTemple University, Fox School of Business')
        elif line.startswith('## '):
            if master and line == '## 8. Discussion and Limitations':
                doc.add_heading('Additional Research Interpretation',level=1)
                doc.add_paragraph('The empirical contrast can also be understood as a choice of estimand. Planning scores represent conditional variation in observed subscription among contact records using X available before list construction. An operational score instead conditions on parts of the bank’s own scheduling and contact process. Its gain need not represent more enduring knowledge of customer preferences: channel and timing may reflect the bank’s policy, while macro indicators may reflect calendar regimes shared by many records. Evaluation across random partitions of the same historical campaigns cannot distinguish stable individual sorting from favorable interpolation across periods. A prospective campaign-level split with release-vintage data would test whether those signals continue to rank customers when the operating context changes. This is why the reported information-set differences are informative as a diagnosis of the retrospective evidence but not a validated gain from procuring each new field.')
                doc.add_paragraph('This distinction bears on what counts as a managerial benchmark. With a fixed budget, the cumulative gain curve is the relevant descriptive object: it states how many subscribers appear in the highest-ranked fraction of historically contacted records. With heterogeneous customer value or contact costs, the preferred ranking might change even if PR-AUC were constant. With an untreated comparison group, the preferred ranking might change again because a likely subscriber might have subscribed without this contact. These are separate changes to the policy objective, not minor adjustments to the threshold of the same response model. The present work deliberately holds its objective to observed response concentration so that it can state precisely what is and is not measured.')
                doc.add_paragraph('The paper thus offers a portable design principle with a deliberately narrow empirical claim. Before choosing an estimator, an analyst should record the action, decision time, eligible population, feature release time, and observed outcome. The analyst can then report metrics for the resulting information set and budget, alongside clearly labeled scenarios for later decisions. Such an audit does not require a novel classifier. Its value lies in preventing a strong retrospective score from being misreported as evidence for a policy that could never have used those inputs. The historical bank data make this failure mode transparent, but they cannot tell us how often it occurs in present-day campaigns.')
            doc.add_heading(line[3:],level=1)
            if line == '## References': in_references=True
        elif line.startswith('**['): exhibit(doc,line.strip('*'))
        else:
            p=doc.add_paragraph()
            if line.startswith('**Abstract') or line.startswith('**Keywords') or in_references:
                p.paragraph_format.first_line_indent=Inches(0)
            if line.startswith('**Abstract'):
                p.paragraph_format.space_after=Pt(5)
            if in_references:
                p.paragraph_format.left_indent=Inches(.18)
                p.paragraph_format.first_line_indent=Inches(-.18)
                p.paragraph_format.space_after=Pt(1)
            emit_inline(p,line.replace('\n',' '))
    path=HERE/('Rong_Zhao_Writing_Sample.docx' if master else 'AMS_2027_Blind_Manuscript.docx')
    doc.save(path)
    return path

if __name__=='__main__':
    for mode in (False,True): print(build(mode))
