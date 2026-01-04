"""
Evaluation Dataset Generator for RAGAS

Generates a deterministic evaluation dataset (50+ samples) by sampling
questions across all BNM PDFs. The dataset is designed for RAGAS metrics.

Output format:
{
  "question": "...",
  "contexts": [],  # Populated during evaluation  
  "ground_truth": "..."
}
"""

import json
import random
from pathlib import Path

# Fixed seed for reproducibility
SEED = 42
random.seed(SEED)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "bnm"
OUTPUT_FILE = PROJECT_ROOT / "tests" / "eval_dataset.json"

# Evaluation questions organized by source PDF
# Each entry: (question, ground_truth, source_pdf, category)
EVAL_QUESTIONS = [
    # === Credit Card Policy (bnm_credit_card_policy_2025.pdf) ===
    ("What is the minimum monthly repayment for credit cards?",
     "The minimum monthly repayment comprises at least 5% of the total amount outstanding, the total amount of contracted monthly instalments of any EPP and BTP, and the contracted monthly term loan instalment for any Automatic Balance Conversion.",
     "bnm_credit_card_policy_2025.pdf", "credit_card"),
    
    ("What is the maximum credit limit for a new credit card applicant with income below RM36,000?",
     "The credit limit shall not exceed two times the monthly income for applicants with annual income below RM36,000.",
     "bnm_credit_card_policy_2025.pdf", "credit_card"),
    
    ("What is the minimum age requirement for credit card principal holders?",
     "A principal cardholder must be at least 21 years of age.",
     "bnm_credit_card_policy_2025.pdf", "credit_card"),
    
    ("What are the requirements for credit card balance transfer plans?",
     "Issuers must clearly disclose the balance transfer amount, tenure, monthly instalment, finance charges, and any fees applicable to the balance transfer plan.",
     "bnm_credit_card_policy_2025.pdf", "credit_card"),
    
    ("What must issuers do before increasing a credit card limit?",
     "Issuers must obtain cardholder's consent before increasing the credit limit and must conduct affordability assessment.",
     "bnm_credit_card_policy_2025.pdf", "credit_card"),
    
    # === Debit Card Policy (bnm_debit_card_policy_2025.pdf) ===
    ("What are the requirements for debit card transaction limits?",
     "Issuers must set appropriate transaction limits and allow cardholders to set their own limits within the issuer's maximum limits.",
     "bnm_debit_card_policy_2025.pdf", "debit_card"),
    
    ("What security features are required for debit cards?",
     "Debit cards must have chip and PIN authentication, and issuers must implement fraud monitoring systems.",
     "bnm_debit_card_policy_2025.pdf", "debit_card"),
    
    ("What are the notification requirements for debit card transactions?",
     "Issuers must send real-time notifications to cardholders for all transactions.",
     "bnm_debit_card_policy_2025.pdf", "debit_card"),
    
    # === AML/CFT Policy (bnm_aml_cft_policy_2025.pdf) ===
    ("What is Customer Due Diligence (CDD) under AML/CFT requirements?",
     "Customer Due Diligence requires financial institutions to identify customers and verify their identity, identify beneficial owners, and understand the purpose and intended nature of the business relationship.",
     "bnm_aml_cft_policy_2025.pdf", "aml_cft"),
    
    ("When must a financial institution file a Suspicious Transaction Report?",
     "A financial institution must file a Suspicious Transaction Report when it has reasonable grounds to suspect that a transaction involves proceeds of unlawful activity or is related to terrorism financing.",
     "bnm_aml_cft_policy_2025.pdf", "aml_cft"),
    
    ("What is Enhanced Due Diligence (EDD)?",
     "Enhanced Due Diligence is additional scrutiny applied to higher-risk customers, including politically exposed persons, and requires more rigorous verification measures.",
     "bnm_aml_cft_policy_2025.pdf", "aml_cft"),
    
    ("What are the record-keeping requirements under AML/CFT?",
     "Financial institutions must maintain records of customer identification, transactions, and business correspondence for at least six years.",
     "bnm_aml_cft_policy_2025.pdf", "aml_cft"),
    
    ("What is the requirement for ongoing monitoring under AML/CFT?",
     "Financial institutions must conduct ongoing monitoring of business relationships to ensure transactions are consistent with the institution's knowledge of the customer.",
     "bnm_aml_cft_policy_2025.pdf", "aml_cft"),
    
    # === Claims Settlement (bnm_claims_settlement_practices_policy_2024.pdf) ===
    ("What is the timeframe for settling insurance claims?",
     "Insurers must settle claims within 14 days from the date of receipt of complete documentation for non-complex claims.",
     "bnm_claims_settlement_practices_policy_2024.pdf", "claims"),
    
    ("What are the requirements for claims documentation?",
     "Insurers must clearly communicate the required documentation to claimants and not request unnecessary documents.",
     "bnm_claims_settlement_practices_policy_2024.pdf", "claims"),
    
    ("What recourse do policyholders have for disputed claims?",
     "Policyholders may refer disputes to the Financial Ombudsman Scheme or the Ombudsman for Financial Services.",
     "bnm_claims_settlement_practices_policy_2024.pdf", "claims"),
    
    # === eKYC Policy (bnm_ekyc_policy_2024.pdf) ===
    ("What are the requirements for electronic Know Your Customer (eKYC)?",
     "Financial institutions must ensure eKYC processes include identity verification through reliable and independent sources, liveness detection, and secure storage of verification data.",
     "bnm_ekyc_policy_2024.pdf", "ekyc"),
    
    ("What is liveness detection in eKYC?",
     "Liveness detection ensures that the person being verified is physically present and not using a photograph, video, or mask.",
     "bnm_ekyc_policy_2024.pdf", "ekyc"),
    
    ("What are the data protection requirements for eKYC?",
     "Financial institutions must implement appropriate security measures to protect customer data collected through eKYC processes.",
     "bnm_ekyc_policy_2024.pdf", "ekyc"),
    
    # === Fair Treatment of Consumers (bnm_fair_treatment_consumers_policy_2024.pdf) ===
    ("What are the disclosure requirements for financial products?",
     "Financial service providers must provide clear, accurate and timely disclosure of product features, terms, fees and risks in a Product Disclosure Sheet before the customer enters into a contract.",
     "bnm_fair_treatment_consumers_policy_2024.pdf", "consumer_protection"),
    
    ("What are the requirements for handling customer complaints?",
     "Financial service providers must have effective complaint handling procedures and resolve complaints fairly and promptly.",
     "bnm_fair_treatment_consumers_policy_2024.pdf", "consumer_protection"),
    
    ("What is the cooling-off period for financial products?",
     "Customers have a cooling-off period during which they can cancel certain financial products without penalty.",
     "bnm_fair_treatment_consumers_policy_2024.pdf", "consumer_protection"),
    
    # === Technology Risk (bnm_rmit_technology_risk_policy_2025.pdf) ===
    ("What are the technology risk management requirements for financial institutions?",
     "Financial institutions must establish a technology risk management framework that includes governance, risk assessment, security controls, incident management, and business continuity.",
     "bnm_rmit_technology_risk_policy_2025.pdf", "technology"),
    
    ("What are the cybersecurity requirements for financial institutions?",
     "Financial institutions must implement security controls including access management, network security, encryption, and security monitoring.",
     "bnm_rmit_technology_risk_policy_2025.pdf", "technology"),
    
    ("What are the requirements for technology incident reporting?",
     "Financial institutions must report significant technology incidents to Bank Negara Malaysia within specified timeframes.",
     "bnm_rmit_technology_risk_policy_2025.pdf", "technology"),
    
    ("What is the requirement for business continuity planning?",
     "Financial institutions must establish business continuity plans to ensure critical operations can continue during disruptions.",
     "bnm_rmit_technology_risk_policy_2025.pdf", "technology"),
    
    # === Shariah Governance (bnm_shariah_governance_policy_2019.pdf) ===
    ("What is the role of the Shariah Committee in Islamic financial institutions?",
     "The Shariah Committee is responsible for advising the board of directors on Shariah matters, endorsing Shariah policies and procedures, and ensuring that products, services and operations comply with Shariah.",
     "bnm_shariah_governance_policy_2019.pdf", "islamic_finance"),
    
    ("What are the qualifications required for Shariah Committee members?",
     "Shariah Committee members must have appropriate qualifications in Shariah, particularly in fiqh muamalat, and relevant experience in Islamic finance.",
     "bnm_shariah_governance_policy_2019.pdf", "islamic_finance"),
    
    ("What is Shariah non-compliance risk?",
     "Shariah non-compliance risk is the risk of financial or non-financial impact arising from failure to comply with Shariah rulings.",
     "bnm_shariah_governance_policy_2019.pdf", "islamic_finance"),
    
    # === Takaful Operating Costs (bnm_operating_cost_takaful_operators_2023.pdf) ===
    ("What are the requirements for takaful operators regarding operating costs?",
     "Takaful operators must ensure that operating costs are managed efficiently and do not adversely affect the interests of participants.",
     "bnm_operating_cost_takaful_operators_2023.pdf", "islamic_finance"),
    
    ("What is the wakalah fee in takaful?",
     "The wakalah fee is a fee charged by the takaful operator for managing the takaful fund on behalf of participants.",
     "bnm_operating_cost_takaful_operators_2023.pdf", "islamic_finance"),
    
    # === Electronic Money (bnm_electronic_money_policy_2025.pdf) ===
    ("What are the requirements for electronic money issuers?",
     "Electronic money issuers must maintain safeguarding requirements for customer funds, implement robust security controls, and provide clear terms and conditions to users.",
     "bnm_electronic_money_policy_2025.pdf", "payments"),
    
    ("What are the safeguarding requirements for e-money funds?",
     "E-money issuers must safeguard customer funds by placing them in a trust account or obtaining a bank guarantee.",
     "bnm_electronic_money_policy_2025.pdf", "payments"),
    
    ("What are the transaction limits for e-money?",
     "E-money accounts are subject to maximum balance limits and transaction limits based on the verification level of the account holder.",
     "bnm_electronic_money_policy_2025.pdf", "payments"),
    
    # === Capital Adequacy (bnm_capital_adequacy_framework_2024.pdf) ===
    ("What is the minimum capital adequacy ratio for banks?",
     "Banks must maintain a minimum Common Equity Tier 1 capital ratio of 4.5%, Tier 1 capital ratio of 6%, and total capital ratio of 8%.",
     "bnm_capital_adequacy_framework_2024.pdf", "capital"),
    
    ("What is the capital conservation buffer?",
     "The capital conservation buffer requires banks to hold an additional 2.5% of Common Equity Tier 1 capital above the minimum requirements.",
     "bnm_capital_adequacy_framework_2024.pdf", "capital"),
    
    ("What are risk-weighted assets?",
     "Risk-weighted assets are calculated by assigning risk weights to different asset classes based on their credit risk.",
     "bnm_capital_adequacy_framework_2024.pdf", "capital"),
    
    # === Islamic Capital Adequacy (bnm_capital_adequacy_islamic_framework_2024.pdf) ===
    ("What are the capital requirements for Islamic banks?",
     "Islamic banks must maintain capital ratios in accordance with the Basel III framework, with specific adjustments for Islamic financial instruments.",
     "bnm_capital_adequacy_islamic_framework_2024.pdf", "islamic_finance"),
    
    ("How are profit-sharing investment accounts treated for capital purposes?",
     "Profit-sharing investment accounts may be excluded from the calculation of risk-weighted assets under certain conditions.",
     "bnm_capital_adequacy_islamic_framework_2024.pdf", "islamic_finance"),
    
    # === Corporate Governance DFI (bnm_corporate_governance_dfi_2024.pdf) ===
    ("What are the board composition requirements for development financial institutions?",
     "The board must comprise a majority of independent directors and have appropriate mix of skills, experience, and diversity to discharge its responsibilities effectively.",
     "bnm_corporate_governance_dfi_2024.pdf", "governance"),
    
    ("What are the responsibilities of the board of directors?",
     "The board is responsible for setting strategy, overseeing management, ensuring effective risk management, and maintaining integrity and ethical standards.",
     "bnm_corporate_governance_dfi_2024.pdf", "governance"),
    
    ("What is the role of the audit committee?",
     "The audit committee oversees financial reporting, internal controls, internal audit, and external audit functions.",
     "bnm_corporate_governance_dfi_2024.pdf", "governance"),
    
    # === Medical Insurance/Takaful (bnm_medical_health_insurance_takaful_policy_2024.pdf) ===
    ("What are the requirements for medical insurance claim processing?",
     "Insurers must process medical claims promptly, provide clear communication on claim status, and ensure fair assessment of claims based on policy terms.",
     "bnm_medical_health_insurance_takaful_policy_2024.pdf", "insurance"),
    
    ("What are the pre-authorization requirements for medical treatment?",
     "Insurers may require pre-authorization for certain treatments, but must respond to pre-authorization requests within specified timeframes.",
     "bnm_medical_health_insurance_takaful_policy_2024.pdf", "insurance"),
    
    ("What is guaranteed renewability for medical insurance?",
     "Medical insurance policies must offer guaranteed renewability, meaning the insurer cannot refuse to renew coverage based on the policyholder's claims history.",
     "bnm_medical_health_insurance_takaful_policy_2024.pdf", "insurance"),
    
    # === Payment System Operators (bnm_payment_system_operator_policy_2022.pdf) ===
    ("What are the licensing requirements for payment system operators?",
     "Payment system operators must obtain approval from Bank Negara Malaysia before commencing operations.",
     "bnm_payment_system_operator_policy_2022.pdf", "payments"),
    
    ("What are the operational requirements for payment systems?",
     "Payment system operators must ensure system reliability, security, and resilience, and have adequate business continuity arrangements.",
     "bnm_payment_system_operator_policy_2022.pdf", "payments"),
    
    # === Financial Advisers (bnm_financial_advisers_conduct_2022.pdf) ===
    ("What are the duties of financial advisers to clients?",
     "Financial advisers must act in the best interests of clients, provide suitable recommendations, and disclose all relevant information including conflicts of interest.",
     "bnm_financial_advisers_conduct_2022.pdf", "consumer_protection"),
    
    ("What are the qualification requirements for financial advisers?",
     "Financial advisers must meet minimum qualification requirements and undergo continuous professional development.",
     "bnm_financial_advisers_conduct_2022.pdf", "consumer_protection"),
    
    # === Financial Brokers (bnm_financial_brokers_conduct_2025.pdf) ===
    ("What are the conduct requirements for financial brokers?",
     "Financial brokers must act with integrity, treat customers fairly, and ensure suitability of products recommended.",
     "bnm_financial_brokers_conduct_2025.pdf", "consumer_protection"),
    
    # === Currency Processing (bnm_currency_processing_business_policy_2024.pdf) ===
    ("What are the requirements for currency processing businesses?",
     "Currency processing businesses must be registered with Bank Negara Malaysia and comply with security and operational standards.",
     "bnm_currency_processing_business_policy_2024.pdf", "payments"),
]


def generate_dataset():
    """Generate the evaluation dataset."""
    dataset = []
    
    for i, (question, ground_truth, source_pdf, category) in enumerate(EVAL_QUESTIONS, 1):
        dataset.append({
            "id": f"q{i:02d}",
            "question": question,
            "ground_truth": ground_truth,
            "contexts": [],  # To be populated during evaluation
            "source_pdf": source_pdf,
            "category": category
        })
    
    # Shuffle with fixed seed for consistent ordering
    random.shuffle(dataset)
    
    # Re-assign IDs after shuffle
    for i, item in enumerate(dataset, 1):
        item["id"] = f"q{i:02d}"
    
    return dataset


def main():
    """Main entry point."""
    print("Generating RAGAS evaluation dataset...")
    print(f"Seed: {SEED}")
    
    dataset = generate_dataset()
    
    # Write to file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {len(dataset)} evaluation samples")
    print(f"Output: {OUTPUT_FILE}")
    
    # Summary by category
    categories = {}
    for item in dataset:
        cat = item["category"]
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nSamples by category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
