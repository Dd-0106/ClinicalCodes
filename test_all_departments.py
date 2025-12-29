"""
Test comprehensive medical extraction across ALL departments
"""
import requests
import json

def test_department(name, report):
    """Test a single department case"""
    print(f"\n{'='*80}")
    print(f"🏥 TESTING: {name}")
    print(f"{'='*80}")
    print(f"Medical Report:\n{report[:200]}...")
    print(f"{'='*80}")
    
    try:
        url = "http://localhost:8000/api/code"
        payload = {"report_text": report}
        
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✓ Extracted Statements: {len(result['extracted_statements'])}")
            print(f"✓ Generated Codes: {result['total_codes']}")
            print(f"✓ Average Confidence: {result['avg_confidence']}%")
            
            if result['extracted_statements']:
                print(f"\n📋 EXTRACTED STATEMENTS:")
                for i, stmt in enumerate(result['extracted_statements'][:10], 1):
                    print(f"  {i}. {stmt}")
            
            if result['codes']:
                symptoms = [c for c in result['codes'] if c['category'] == 'symptom']
                diagnoses = [c for c in result['codes'] if c['category'] in ['primary_diagnosis', 'secondary_diagnosis']]
                procedures = [c for c in result['codes'] if c['category'] == 'procedure']
                
                if symptoms:
                    print(f"\n🔵 SYMPTOMS ({len(symptoms)}):")
                    for code in symptoms:
                        print(f"  {code['code']} - {code['description'][:60]}")
                
                if diagnoses:
                    print(f"\n🔴 DIAGNOSES ({len(diagnoses)}):")
                    for code in diagnoses:
                        print(f"  {code['code']} - {code['description'][:60]}")
                
                if procedures:
                    print(f"\n🟢 PROCEDURES ({len(procedures)}):")
                    for code in procedures:
                        print(f"  {code['code']} - {code['description'][:60]}")
            else:
                print(f"\n❌ NO CODES GENERATED!")
            
            return result['total_codes']
        else:
            print(f"❌ API Error: {response.status_code}")
            return 0
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 0

def main():
    """Test all department cases"""
    
    test_cases = [
        ("ORTHOPEDICS - Low Back Pain", """
PATIENT PRESENTED WITH C/O-LOW BACK PAIN SINCE 5 DAYS ASSOCIATED WITH RESTRICTED MOVEMENTS. NO RADIATION OF PAIN. NO H/O-TRAUMA.
EXAMINATION WAS DONE. CONSERVATIVE MANAGEMENT ADVISED.
TREATMENT GIVEN
TAB-ETORICOXIB 90 MG OD
TAB-MYORIL 4 MG BD
TAB-PANTOP 40 MG OD
HOT FERMENTATION ADVISED
BED REST ADVISED
"""),
        
        ("NEUROLOGY - Seizure", """
PATIENT GOT ADMITTED WITH H/O-GENERALIZED TONIC CLONIC SEIZURE LASTING 2-3 MINUTES FOLLOWED BY POST ICTAL CONFUSION. NO H/O-FEVER/HEAD INJURY.
EXAMINATION WAS DONE. NEUROLOGY OPINION SOUGHT. CT BRAIN ADVISED.
TREATMENT GIVEN
INJ-LORAZEPAM 2 MG IV STAT
INJ-LEVETIRACETAM 1 GM IV BD
IVF-NS @50 ML/HR
SEIZURE PRECAUTIONS TAKEN
"""),
        
        ("SURGERY - Acute Appendicitis", """
PATIENT GOT ADMITTED WITH C/O-PAIN ABDOMEN SINCE 1 DAY, INITIALLY PERIUMBILICAL NOW SHIFTED TO RIGHT ILIAC FOSSA. ASSOCIATED WITH NAUSEA+.
SURGICAL OPINION WAS TAKEN. ULTRASOUND ABDOMEN WAS DONE. PATIENT KEPT NPO.
TREATMENT GIVEN
IVF-NS @80 ML/HR
INJ-PAN 40 MG IV OD
INJ-CEFTRIAXONE 1 GM IV BD
INJ-METROGYL 100 ML IV TID
INJ-PCT 1 GM IV SOS
"""),
        
        ("GYNECOLOGY - Menorrhagia", """
PATIENT PRESENTED WITH C/O-EXCESSIVE BLEEDING DURING MENSTRUATION SINCE LAST 2 CYCLES ASSOCIATED WITH FATIGUE AND DIZZINESS.
GYNECOLOGY OPINION WAS TAKEN. ULTRASOUND PELVIS ADVISED.
TREATMENT GIVEN
TAB-TRANEXAMIC ACID BD
TAB-IRON + FOLIC ACID OD
TAB-PANTOP 40 MG OD
"""),
        
        ("PEDIATRICS - Respiratory Distress", """
CHILD GOT ADMITTED WITH C/O-COUGH AND BREATHLESSNESS SINCE 2 DAYS ASSOCIATED WITH FEEDING DIFFICULTY. NO H/O-SEIZURES.
PEDIATRIC EXAMINATION WAS DONE. SUPPORTIVE MANAGEMENT STARTED.
TREATMENT GIVEN
OXYGEN VIA NASAL PRONGS
NEB-SALBUTAMOL QID
NEB-BUDESONIDE BD
IVF-MAINTENANCE FLUIDS
"""),
        
        ("OPHTHALMOLOGY - Conjunctivitis", """
PATIENT PRESENTED WITH C/O-REDNESS AND DISCHARGE FROM BOTH EYES SINCE 2 DAYS ASSOCIATED WITH ITCHING.
OPHTHALMIC EXAMINATION WAS DONE.
TREATMENT GIVEN
EYE DROP-MOXIFLOXACIN QID
EYE DROP-LUBRICANT QID
EYE HYGIENE ADVISED
"""),
        
        ("ENT - Pharyngitis", """
PATIENT PRESENTED WITH C/O-SORE THROAT AND FEVER SINCE 2 DAYS ASSOCIATED WITH DYSPHAGIA.
ENT EXAMINATION WAS DONE.
TREATMENT GIVEN
TAB-AMOXYCLAV 625 MG TID
TAB-PCT 650 MG SOS
WARM SALINE GARGLES ADVISED
""")
    ]
    
    print("\n" + "="*80)
    print("🧪 COMPREHENSIVE TESTING - ALL MEDICAL DEPARTMENTS")
    print("="*80)
    
    total_tests = len(test_cases)
    total_codes = 0
    
    for name, report in test_cases:
        codes = test_department(name, report)
        total_codes += codes
    
    print(f"\n{'='*80}")
    print(f"📊 OVERALL RESULTS")
    print(f"{'='*80}")
    print(f"Total Tests: {total_tests}")
    print(f"Total Codes Generated: {total_codes}")
    print(f"Average Codes per Test: {total_codes/total_tests:.1f}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
