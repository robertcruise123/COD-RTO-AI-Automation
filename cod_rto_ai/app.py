import streamlit as st
import pandas as pd
import numpy as np
import json, re
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split

st.set_page_config(page_title='COD Risk Desk', page_icon='📦', layout='wide')
ROOT=Path(__file__).parent
DEFAULTS={'ship_max':0.18,'confirm_max':0.35,'nudge_max':0.55,'prepaid_discount_pct':3,'prepaid_discount_cap':30,'confirmation_timeout_hours':24,'auto_cancel_enabled':False,'high_risk_pincode_block':False}

def load_config():
    p=ROOT/'config.json'
    try: return {**DEFAULTS, **json.loads(p.read_text())}
    except Exception: return DEFAULTS.copy()

def clean_status(s):
    x=str(s).strip().upper()
    if x=='DELIVERED': return 0
    if x=='RTO': return 1
    return np.nan

def features(df):
    x=df.copy()
    x['order_datetime']=pd.to_datetime(x.get('order_datetime'), errors='coerce')
    x['order_hour']=x.order_datetime.dt.hour.fillna(-1)
    x['order_dayofweek']=x.order_datetime.dt.dayofweek.fillna(-1)
    x['address_text']=x['address'].fillna('').astype(str).str.strip().str.lower()
    x['address_len']=x.address_text.str.len()
    x['address_word_count']=x.address_text.str.split().str.len()
    x['address_has_landmark']=x.address_text.str.contains(r'mandir|temple|school|bank|near|opposite|opp|chowk|market|landmark|ke saamne|ke pass|road|nagar|sector', regex=True).astype(int)
    x['address_suspicious']=x.address_text.isin(['','.','-','na','n/a','test','unknown','asdf']) | (x.address_len<8)
    x['name_text']=x['customer_name'].fillna('').astype(str).str.strip().str.lower()
    x['name_len']=x.name_text.str.len()
    x['name_suspicious']=x.name_text.str.contains(r'asdf|qwerty|test|fake|unknown|dummy|abc123|^\d+$', regex=True).astype(int)
    x['phone_digits']=x['phone'].fillna('').astype(str).str.replace(r'\D','',regex=True)
    x['phone_len']=x.phone_digits.str.len()
    x['phone_valid_length']=(x.phone_len==10).astype(int)
    x['prior_rto_rate']=x['customer_prior_rto'].fillna(0)/(x['customer_prior_orders'].fillna(0).clip(lower=0)+1)
    x['pincode']=x['pincode'].fillna(-1).astype(str).str.replace(r'\.0$','',regex=True)
    x['order_value']=pd.to_numeric(x['order_value'],errors='coerce')
    x['num_items']=pd.to_numeric(x['num_items'],errors='coerce')
    x['customer_prior_orders']=pd.to_numeric(x['customer_prior_orders'],errors='coerce')
    x['customer_prior_rto']=pd.to_numeric(x['customer_prior_rto'],errors='coerce')
    x['pincode_missing']=(x['pincode'].isin(['-1','nan','None',''])).astype(int)
    return x

NUM=['order_value','num_items','customer_prior_orders','customer_prior_rto','prior_rto_rate','order_hour','order_dayofweek','address_len','address_word_count','address_has_landmark','address_suspicious','name_len','name_suspicious','phone_len','phone_valid_length','pincode_missing']
CAT=['city','state','utm_source','courier','pincode']
def train_model(raw):
    d=raw.copy(); d['label']=d['final_status'].map(clean_status)
    d=d[d.label.notna()].copy()
    if len(d)<30 or d.label.nunique()<2: return None, None, d
    X=features(d); y=d.label.astype(int)
    # Time-ordered holdout avoids training on later records when timestamps are available.
    order=pd.to_datetime(d.order_datetime,errors='coerce')
    ix=order.sort_values().index
    cut=max(1,int(len(ix)*0.8)); train_idx=ix[:cut]; test_idx=ix[cut:]
    if y.loc[train_idx].nunique()<2 or y.loc[test_idx].nunique()<2:
        train_idx,test_idx=train_test_split(d.index,test_size=.2,random_state=42,stratify=y)
    prep=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUM),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),CAT)])
    model=Pipeline([('prep',prep),('clf',LogisticRegression(max_iter=1500,class_weight='balanced',C=.5))])
    model.fit(X.loc[train_idx,NUM+CAT],y.loc[train_idx])
    prob=model.predict_proba(X.loc[test_idx,NUM+CAT])[:,1]
    metrics={'test_rows':len(test_idx),'train_rows':len(train_idx),'roc_auc':float(roc_auc_score(y.loc[test_idx],prob)),'pr_auc':float(average_precision_score(y.loc[test_idx],prob)),'brier':float(brier_score_loss(y.loc[test_idx],prob))}
    # Retrain on all labeled history for production scoring.
    model.fit(X[NUM+CAT],y)
    return model,metrics,d

def action_for(p, row, cfg):
    if row.get('address_suspicious',False) or row.get('name_suspicious',0)==1 or row.get('phone_valid_length',1)==0 or row.get('order_value',1)<=0:
        return 'Ship after confirmation', 'Contact details or address need verification before dispatch.'
    if cfg['high_risk_pincode_block'] and float(p)>=cfg['nudge_max']:
        return ('Hold / cancel' if cfg['auto_cancel_enabled'] else 'Ship after confirmation'), 'Configured high-risk threshold reached; review the order before dispatch.'
    if p < cfg['ship_max']: return 'Ship as-is', f'Estimated RTO risk is low ({p:.0%}), below the ship threshold.'
    if p < cfg['confirm_max']: return 'Ship after confirmation', f'Estimated RTO risk is moderate ({p:.0%}); confirm customer intent on WhatsApp.'
    if p < cfg['nudge_max']: return 'Nudge to prepaid', f'Estimated RTO risk is elevated ({p:.0%}); offer prepaid payment with a small discount.'
    if cfg['auto_cancel_enabled']: return 'Hold / cancel', f'Estimated RTO risk is high ({p:.0%}); hold for manual review or cancel under the configured policy.'
    return 'Ship after confirmation', f'Estimated RTO risk is high ({p:.0%}); avoid automatic cancellation and escalate for human review.'

def risk_cost(order_value, p):
    # RTO shipping + handling + expected damage cost. Damage cost is 8% x ₹180.
    rto_cost=65+65+25+0.08*180
    # Contribution margin foregone on a failed delivery is modeled separately as risk context, not added to direct logistics loss.
    return p*rto_cost


st.title('📦 COD Risk Desk')
st.caption('Decision support for COD orders · prototype trained on the supplied historical sample')
cfg=load_config()
data_path=ROOT/'orders_sample.csv'
if not data_path.exists(): st.error('orders_sample.csv is missing. Put it next to app.py.'); st.stop()
raw=pd.read_csv(data_path)
model,metrics,labeled=train_model(raw)
if model is None: st.error('Not enough labeled data to train.'); st.stop()
Xhist=features(labeled); hist_probs=model.predict_proba(Xhist[NUM+CAT])[:,1]
labeled=labeled.copy(); labeled['risk_probability']=hist_probs
labeled['recommended_action']=[action_for(p,features(pd.DataFrame([r])).iloc[0].to_dict(),cfg)[0] for p,(_,r) in zip(hist_probs,labeled.iterrows())]

with st.sidebar:
    st.header('Decision thresholds')
    st.caption('Adjust without editing code. Changes apply immediately to recommendations.')
    cfg['ship_max']=st.slider('Ship as-is below risk',0.01,.50,float(cfg['ship_max']),.01)
    cfg['confirm_max']=st.slider('Confirm below risk',cfg['ship_max']+.01,.70,max(cfg['confirm_max'],cfg['ship_max']+.01),.01)
    cfg['nudge_max']=st.slider('Prepaid nudge below risk',cfg['confirm_max']+.01,.95,max(cfg['nudge_max'],cfg['confirm_max']+.01),.01)
    cfg['prepaid_discount_pct']=st.slider('Prepaid discount (%)',0,10,int(cfg['prepaid_discount_pct']))
    cfg['prepaid_discount_cap']=st.number_input('Discount cap (₹)',min_value=0,max_value=100,value=int(cfg['prepaid_discount_cap']))
    cfg['confirmation_timeout_hours']=st.selectbox('Confirmation timeout (hours)',[6,12,24,36,48],index=[6,12,24,36,48].index(int(cfg['confirmation_timeout_hours'])))
    cfg['auto_cancel_enabled']=st.toggle('Allow auto-cancel for high risk',value=cfg['auto_cancel_enabled'])
    cfg['high_risk_pincode_block']=st.toggle('Use high-risk threshold rule',value=cfg['high_risk_pincode_block'])
    if st.button('Save thresholds to config.json'):
        (ROOT/'config.json').write_text(json.dumps(cfg,indent=2)); st.success('Saved configuration.')

pages=st.tabs(['Morning dashboard','Score a new order','WhatsApp simulator','Historical data & model'])
with pages[0]:
    st.subheader('Yesterday at a glance')
    date=pd.to_datetime(labeled.order_datetime,errors='coerce').dt.date
    latest=date.max(); yesterday=latest # use latest complete sample day as proxy, clearly labeled
    snap=labeled[date==yesterday]
    if snap.empty: snap=labeled.tail(0)
    buckets=['Ship as-is','Ship after confirmation','Nudge to prepaid','Hold / cancel']
    cols=st.columns(4)
    for c,b in zip(cols,buckets): c.metric(b,int((snap.recommended_action==b).sum()))
    st.caption(f"Sample's latest date: {yesterday}. This is a historical proxy, not live yesterday's Shopify feed.")
    st.divider()
    a,b,c=st.columns(3)
    actual_rto=float(labeled.label.mean())
    avg_order=float(labeled.order_value.mean())
    direct_cost=65+65+25+.08*180
    a.metric('Observed sample RTO rate',f'{actual_rto:.1%}')
    b.metric('Direct cost per RTO',f'₹{direct_cost:.2f}')
    c.metric('Expected direct RTO cost / order',f'₹{actual_rto*direct_cost:.2f}')
    st.markdown('#### Estimated financial impact (illustrative)')
    st.info('Dashboard savings are a scenario estimate, not a causal claim. The prototype assumes that a prevented RTO avoids ₹169.40 direct loss; cancelled genuine orders lose an estimated 62% gross contribution. Change the assumptions before operational use.')
    avoided_rate=st.slider('Assumed RTOs prevented among intervention orders',0.0,1.0,.50,.05,key='avoided')
    genuine_cancel_rate=st.slider('Assumed genuine share of held/cancelled orders',0.0,1.0,.50,.05,key='genuine')
    interventions=labeled[labeled.recommended_action.isin(['Ship after confirmation','Nudge to prepaid','Hold / cancel'])]
    cancelled=labeled[labeled.recommended_action=='Hold / cancel']
    saved=len(interventions)*actual_rto*avoided_rate*direct_cost
    lost=float((cancelled.order_value*0.62).sum()*genuine_cancel_rate)
    disc=(labeled[labeled.recommended_action=='Nudge to prepaid'].order_value.clip(upper=10000)*cfg['prepaid_discount_pct']/100).clip(upper=cfg['prepaid_discount_cap']).sum()
    net=saved-lost-disc
    x,y,z=st.columns(3); x.metric('Estimated RTO cost avoided',f'₹{saved:,.0f}'); y.metric('Potential genuine-order contribution lost',f'₹{lost:,.0f}'); z.metric('Net impact after discounts',f'₹{net:,.0f}')
    st.caption('Estimates use historical sample rows and editable assumptions. They are not proof that an intervention caused an order to be saved.')
    st.markdown('#### Recommended-action mix in full sample')
    st.bar_chart(labeled.recommended_action.value_counts().reindex(buckets,fill_value=0))
    st.markdown('#### Retrospective replay: last calendar month present in the CSV')
    month_key=pd.to_datetime(labeled.order_datetime,errors='coerce').dt.to_period('M')
    latest_month=month_key.max()
    month_rows=labeled[month_key==latest_month]
    st.write(f'{latest_month}: {len(month_rows):,} labeled historical orders. Actions below are an exploratory replay using a model refit on the full labeled dataset, so they are **in-sample** and must not be treated as unbiased performance or proven savings.')
    st.dataframe(month_rows.recommended_action.value_counts().reindex(buckets,fill_value=0).rename_axis('recommended action').reset_index(name='orders'),use_container_width=True,hide_index=True)

with pages[1]:
    st.subheader('Score a new COD order')
    with st.form('score_form'):
        c1,c2,c3=st.columns(3)
        name=c1.text_input('Customer name','Priya Sharma'); phone=c2.text_input('Phone','9876543210'); value=c3.number_input('Order value (₹)',min_value=0,value=649)
        address=st.text_area('Address','House 12, Station Road, near SBI Bank'); city=c1.text_input('City','Siliguri'); state=c2.text_input('State','West Bengal'); pin=c3.text_input('Pincode','734001')
        c1,c2,c3=st.columns(3); items=c1.text_input('Items','Shampoo'); num_items=c2.number_input('Number of items',1,20,1); source=c3.selectbox('Acquisition source',sorted(raw.utm_source.dropna().unique()))
        courier=c1.selectbox('Courier',sorted(raw.courier.dropna().unique())); prior=c2.number_input('Prior orders',0,100,0); prior_rto=c3.number_input('Prior RTOs',0,100,0)
        submitted=st.form_submit_button('Score order')
    if submitted:
        new=pd.DataFrame([{'order_id':'NEW','order_datetime':pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),'customer_name':name,'phone':phone,'address':address,'city':city,'state':state,'pincode':pin,'order_value':value,'items':items,'num_items':num_items,'customer_prior_orders':prior,'customer_prior_rto':prior_rto,'utm_source':source,'courier':courier,'payment_method':'COD','final_status':'PENDING'}])
        fx=features(new); p=float(model.predict_proba(fx[NUM+CAT])[:,1][0]); action,reason=action_for(p,fx.iloc[0].to_dict(),cfg)
        c1,c2,c3=st.columns(3); c1.metric('Estimated RTO risk',f'{p:.1%}'); c2.metric('Recommended action',action); c3.metric('Expected direct RTO cost',f'₹{risk_cost(value,p):.2f}')
        st.success(reason)
        st.caption('Model risk is a ranking estimate trained on a small historical sample, not a guarantee. Address/name checks here are deterministic heuristics; no external LLM is called in this offline demo.')
        if action in ['Ship after confirmation','Nudge to prepaid']:
            st.markdown('**Next step:** use the WhatsApp simulator tab to test replies.')

with pages[2]:
    st.subheader('WhatsApp flow mock')
    kind=st.radio('Flow',['Ship after confirmation','Nudge to prepaid'],horizontal=True)
    language=st.radio('Language',['Hinglish','English'],horizontal=True)
    discount=min(int(649*cfg['prepaid_discount_pct']/100),cfg['prepaid_discount_cap'])
    if kind=='Ship after confirmation':
        msg=('Namaste! Aapka ₹649 ka order confirm karna hai? Dispatch ke liye YES reply karein. Cancel karna ho to NO reply karein. 24 ghante mein reply na aane par order hold rahega.' if language=='Hinglish' else 'Hi! Please confirm your ₹649 order. Reply YES to confirm dispatch or NO to cancel. If we do not hear back within 24 hours, the order will be held for review.')
    else:
        msg=(f'Namaste! Aapke order par prepaid karne par ₹{discount} ki chhoot milegi. Online pay karne ke liye secure payment link use karein. COD rakhna hai to COD reply karein.' if language=='Hinglish' else f'Hi! Pay online for your order and get ₹{discount} off. Use the secure payment link at checkout. Reply COD if you prefer to keep cash on delivery.')
    st.chat_message('assistant').write(msg)
    reply=st.selectbox('Simulate customer reply',['YES','NO','COD','PAID','Maybe','Where is my order?','No reply (24 hours elapsed)'])
    if st.button('Process reply'):
        if kind=='Ship after confirmation':
            if reply=='YES': out='Confirmed → ship order; log timestamp and confirmation source.'
            elif reply=='NO': out='Customer declined → cancel and log reason; do not dispatch.'
            elif reply=='No reply (24 hours elapsed)': out='Timeout → hold, send one reminder if policy allows, then human review; no automatic cancellation by default.'
            else: out='Unrecognized reply → ask “Please reply YES to confirm or NO to cancel”; keep order on hold.'
        else:
            if reply in ['PAID','YES']: out='Verify payment webhook/status before changing payment method; then mark prepaid and dispatch.'
            elif reply=='COD': out='Customer prefers COD → return to risk decision; confirmation flow may still be required.'
            elif reply=='No reply (24 hours elapsed)': out='No conversion → retain COD decision; hold/review only if risk policy requires it.'
            else: out='Unrecognized reply → send a short clarification; never treat an ambiguous message as payment confirmation.'
        st.info(out)
    st.caption('Production requirements: approved WhatsApp template, consent/opt-out handling, idempotent webhook, verified payment status, audit log and retry controls.')

with pages[3]:
    st.subheader('Historical sample & model diagnostics')
    st.write(f"Rows: {len(raw):,} · Labeled outcomes used: {len(labeled):,} · Date range: {pd.to_datetime(raw.order_datetime,errors='coerce').min().date()} to {pd.to_datetime(raw.order_datetime,errors='coerce').max().date()}")
    if metrics:
        c1,c2,c3,c4=st.columns(4); c1.metric('Time holdout rows',metrics['test_rows']); c2.metric('ROC-AUC',f"{metrics['roc_auc']:.3f}"); c3.metric('PR-AUC',f"{metrics['pr_auc']:.3f}"); c4.metric('Brier score',f"{metrics['brier']:.3f}")
    st.warning('Metrics are a single small time-based holdout and may be unstable. The dataset has duplicate order IDs/timestamps and inconsistent outcome casing; labels are normalized and unlabeled IN_TRANSIT is excluded. Validate with a later untouched cohort before deployment.')
    status=raw.final_status.astype(str).str.strip().str.upper().value_counts().rename_axis('status').reset_index(name='rows')
    st.dataframe(status,use_container_width=True)
    st.dataframe(labeled[['order_id','order_datetime','order_value','customer_prior_orders','customer_prior_rto','risk_probability','recommended_action','final_status']].sort_values('risk_probability',ascending=False),use_container_width=True,hide_index=True)
    st.download_button('Download scored historical sample CSV',labeled.to_csv(index=False).encode(),file_name='scored_orders.csv',mime='text/csv')
