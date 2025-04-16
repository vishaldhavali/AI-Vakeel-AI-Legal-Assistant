import random
from datetime import datetime, timedelta
import json
import os
import re

# List of all Indian states and Union Territories
INDIAN_STATES = [
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chhattisgarh",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",
]

# Union Territories
UNION_TERRITORIES = [
    "Andaman and Nicobar Islands",
    "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi",
    "Jammu and Kashmir",
    "Ladakh",
    "Lakshadweep",
    "Puducherry",
]

# State abbreviations mapping
STATE_ABBREVIATIONS = {
    "ap": "Andhra Pradesh",
    "ar": "Arunachal Pradesh",
    "as": "Assam",
    "br": "Bihar",
    "cg": "Chhattisgarh",
    "ga": "Goa",
    "gj": "Gujarat",
    "hr": "Haryana",
    "hp": "Himachal Pradesh",
    "jh": "Jharkhand",
    "ka": "Karnataka",
    "kl": "Kerala",
    "mp": "Madhya Pradesh",
    "mh": "Maharashtra",
    "mn": "Manipur",
    "ml": "Meghalaya",
    "mz": "Mizoram",
    "nl": "Nagaland",
    "od": "Odisha",
    "pb": "Punjab",
    "rj": "Rajasthan",
    "sk": "Sikkim",
    "tn": "Tamil Nadu",
    "tg": "Telangana",
    "tr": "Tripura",
    "up": "Uttar Pradesh",
    "uk": "Uttarakhand",
    "wb": "West Bengal",
    "dl": "Delhi",
    "jk": "Jammu and Kashmir",
}

# Chief Justices (real names)
CHIEF_JUSTICES = {
    "Supreme Court": "Justice D.Y. Chandrachud",
    "Andhra Pradesh": "Justice Dhiraj Singh Thakur",
    "Arunachal Pradesh": "Justice Nongmeikapam Kotiswar Singh",
    "Assam": "Justice Vijay Bishnoi",
    "Bihar": "Justice K. Vinod Chandran",
    "Chhattisgarh": "Justice Arup Kumar Goswami",
    "Delhi": "Justice Manmohan",
    "Goa": "Justice M.S. Sonak",
    "Gujarat": "Justice Sunita Agarwal",
    "Himachal Pradesh": "Justice Manoj Vyas",
    "Jharkhand": "Justice Sanjaya Kumar Mishra",
    "Karnataka": "Justice Nilay Vipinchandra Anjaria",
    "Kerala": "Justice A. Muhamed Mustaque",
    "Madhya Pradesh": "Justice Ravi Malimath",
    "Manipur": "Justice Siddharth Mridul",
    "Maharashtra": "Justice Devendra Kumar Upadhyaya",
    "Meghalaya": "Justice S. Vaidyanathan",
    "Nagaland": "Justice Nongmeikapam Kotiswar Singh",
    "Odisha": "Justice Chakradhari Sharan Singh",
    "Punjab": "Justice Gurmeet Singh Sandhawalia",
    "Rajasthan": "Justice Augustine George Masih",
    "Sikkim": "Justice Biswanath Somadder",
    "Tamil Nadu": "Justice R. Mahadevan",
    "Telangana": "Justice Alok Aradhe",
    "Tripura": "Justice Aparesh Kumar Singh",
    "Uttar Pradesh": "Justice Arun Bhansali",
    "Uttarakhand": "Justice Ritu Bahri",
    "West Bengal": "Justice T.S. Sivagnanam",
    "Jammu and Kashmir": "Justice N. Kotiswar Singh",
}

# Dummy data for courts and judges
COURTS_DATA = {
    "supreme_court": {
        "total_judges": 34,
        "current_judges": 27,
        "vacancies": 7,
        "chief_justice": "Justice John Doe",
        "last_updated": "2024-03-15",
    },
    "high_courts": [
        {
            "name": "Delhi High Court",
            "total_judges": 45,
            "current_judges": 38,
            "vacancies": 7,
            "chief_justice": "Justice Jane Smith",
            "last_updated": "2024-03-14",
        },
        {
            "name": "Mumbai High Court",
            "total_judges": 40,
            "current_judges": 35,
            "vacancies": 5,
            "chief_justice": "Justice Robert Brown",
            "last_updated": "2024-03-13",
        },
    ],
    "district_courts": [
        {
            "state": "Delhi",
            "total_judges": 150,
            "current_judges": 125,
            "vacancies": 25,
            "last_updated": "2024-03-12",
        },
        {
            "state": "Maharashtra",
            "total_judges": 200,
            "current_judges": 180,
            "vacancies": 20,
            "last_updated": "2024-03-11",
        },
    ],
}

# Dummy data for case status
CASE_STATUS = [
    {
        "case_number": "CRL-2024-001",
        "court": "Delhi High Court",
        "status": "Pending",
        "filing_date": "2024-01-15",
        "next_hearing": "2024-04-25",
        "judge": "Justice Rajesh Kumar",
        "category": "Criminal",
        "petitioner": "State",
        "respondent": "John Doe",
        "last_hearing": "2024-03-10",
        "last_order": "Arguments heard, next hearing for final arguments",
    },
    {
        "case_number": "CIV-2024-045",
        "court": "Bombay High Court",
        "status": "Active",
        "filing_date": "2024-02-10",
        "next_hearing": "2024-04-20",
        "judge": "Justice Priya Singh",
        "category": "Civil",
        "petitioner": "ABC Corporation",
        "respondent": "XYZ Ltd.",
        "last_hearing": "2024-03-15",
        "last_order": "Documentary evidence submitted, next for cross-examination",
    },
    {
        "case_number": "WP-2024-123",
        "court": "Supreme Court",
        "status": "Active",
        "filing_date": "2024-02-25",
        "next_hearing": "2024-04-15",
        "judge": "Justice D.Y. Chandrachud",
        "category": "Writ Petition",
        "petitioner": "Environmental Action Group",
        "respondent": "Union of India",
        "last_hearing": "2024-03-20",
        "last_order": "Counter affidavit to be filed by respondent",
    },
    {
        "case_number": "ARB-2024-015",
        "court": "Delhi High Court",
        "status": "Pending",
        "filing_date": "2024-01-30",
        "next_hearing": "2024-04-18",
        "judge": "Justice Amit Bansal",
        "category": "Arbitration",
        "petitioner": "Global Tech Solutions",
        "respondent": "Indian Infrastructure Ltd",
        "last_hearing": "2024-03-12",
        "last_order": "Arbitration proceedings to continue as scheduled",
    },
    {
        "case_number": "TAX-2024-078",
        "court": "Income Tax Appellate Tribunal",
        "status": "Active",
        "filing_date": "2024-02-15",
        "next_hearing": "2024-04-22",
        "judge": "Justice Sanjay Gupta",
        "category": "Tax",
        "petitioner": "Tech Corp India Pvt Ltd",
        "respondent": "Income Tax Department",
        "last_hearing": "2024-03-18",
        "last_order": "Assessment records to be produced by department",
    },
]

# Dummy data for traffic violations
TRAFFIC_VIOLATIONS = [
    {
        "violation_id": "TV-2024-001",
        "vehicle_number": "DL-01-AB-1234",
        "violation_type": "Red Light",
        "location": "CP, New Delhi",
        "date": "2024-03-10",
        "fine_amount": 1000,
        "status": "Unpaid",
    },
    {
        "violation_id": "TV-2024-002",
        "vehicle_number": "MH-02-CD-5678",
        "violation_type": "Speeding",
        "location": "Marine Drive, Mumbai",
        "date": "2024-03-12",
        "fine_amount": 2000,
        "status": "Paid",
    },
]

# Dummy data for fast track courts
FAST_TRACK_COURTS = [
    {
        "court_id": "FTC-001",
        "name": "POCSO Fast Track Court - Delhi",
        "type": "POCSO",
        "cases_handled": 150,
        "cases_disposed": 95,
        "pending_cases": 55,
        "judge": "Justice Sarah Wilson",
    },
    {
        "court_id": "FTC-002",
        "name": "Women's Safety Court - Mumbai",
        "type": "Women Safety",
        "cases_handled": 200,
        "cases_disposed": 140,
        "pending_cases": 60,
        "judge": "Justice Michael Brown",
    },
]

# Dummy data for live streaming
LIVE_STREAMING = [
    {
        "stream_id": "LS-001",
        "court": "Supreme Court",
        "case_number": "CRL-2024-001",
        "stream_url": "https://stream.courts.gov.in/supreme/001",
        "status": "Live",
        "viewers": 1500,
    },
    {
        "stream_id": "LS-002",
        "court": "Delhi High Court",
        "case_number": "CIV-2024-045",
        "stream_url": "https://stream.courts.gov.in/delhi/002",
        "status": "Scheduled",
        "schedule_time": "2024-04-15 10:30:00",
    },
]


# Generate dynamic court vacancy data for each state
def generate_court_vacancy_data():
    """Generate dynamic court vacancy data for all states"""
    today = datetime.now().strftime("%Y-%m-%d")

    # Supreme Court data
    sc_total = 34
    sc_current = random.randint(25, 32)

    courts_data = {
        "supreme_court": {
            "total_judges": sc_total,
            "current_judges": sc_current,
            "vacancies": sc_total - sc_current,
            "chief_justice": CHIEF_JUSTICES["Supreme Court"],
            "last_updated": today,
        },
        "high_courts": [],
        "district_courts": [],
    }

    # Generate data for all states
    for state in INDIAN_STATES + ["Delhi", "Jammu and Kashmir"]:
        # High Court
        total_hc = random.randint(25, 75)
        current_hc = random.randint(int(total_hc * 0.65), int(total_hc * 0.95))

        high_court = {
            "name": f"{state} High Court",
            "total_judges": total_hc,
            "current_judges": current_hc,
            "vacancies": total_hc - current_hc,
            "chief_justice": CHIEF_JUSTICES.get(
                state,
                f"Justice {random.choice(['S.K.', 'A.K.', 'P.K.', 'R.K'])} {random.choice(['Sharma', 'Singh', 'Verma', 'Gupta', 'Reddy', 'Kumar'])}",
            ),
            "last_updated": today,
        }
        courts_data["high_courts"].append(high_court)

        # District Courts
        population_factor = random.uniform(0.8, 2.2)  # Simulate different state sizes
        total_district = int(random.randint(150, 450) * population_factor)
        current_district = random.randint(
            int(total_district * 0.7), int(total_district * 0.9)
        )

        district_court = {
            "state": state,
            "total_judges": total_district,
            "current_judges": current_district,
            "vacancies": total_district - current_district,
            "last_updated": today,
        }
        courts_data["district_courts"].append(district_court)

    return courts_data


# Initialize court data once
COURTS_DATA = generate_court_vacancy_data()


# Expanded case status data
def generate_expanded_case_status():
    """Generate more comprehensive case status data"""
    case_types = ["CRL", "CIV", "WP", "CP", "ARB", "TAX", "FAM"]
    courts = ["Supreme Court"] + [f"{state} High Court" for state in INDIAN_STATES[:10]]
    judges = [
        f"Justice {name}"
        for name in [
            "Arun Mishra",
            "Rohinton Nariman",
            "D.Y. Chandrachud",
            "S.A. Bobde",
            "L. Nageswara Rao",
            "Hemant Gupta",
            "Ajay Rastogi",
            "Aniruddha Bose",
            "Indu Malhotra",
            "Indira Banerjee",
            "Sanjiv Khanna",
            "Krishna Murari",
        ]
    ]

    statuses = ["Pending", "Active", "Reserved for Judgment", "Disposed", "Adjourned"]
    categories = [
        "Criminal",
        "Civil",
        "Constitutional",
        "Tax",
        "Family",
        "Property",
        "Commercial",
    ]

    expanded_cases = []
    for i in range(50):  # Generate 50 cases
        year = random.randint(2020, 2024)
        case_number = f"{random.choice(case_types)}-{year}-{random.randint(1, 999):03d}"
        filing_date = datetime.strptime(
            f"{year}-{random.randint(1,12)}-{random.randint(1,28)}", "%Y-%m-%d"
        )
        next_hearing = filing_date + timedelta(days=random.randint(30, 365))

        case = {
            "case_number": case_number,
            "court": random.choice(courts),
            "status": random.choice(statuses),
            "filing_date": filing_date.strftime("%Y-%m-%d"),
            "next_hearing": next_hearing.strftime("%Y-%m-%d"),
            "judge": random.choice(judges),
            "category": random.choice(categories),
            "petitioner": f"{random.choice(['Mr.', 'Mrs.', 'Ms.'])} {random.choice(['Amit', 'Rahul', 'Priya', 'Neha', 'Vikram', 'Sanjay'])} {random.choice(['Singh', 'Sharma', 'Patel', 'Gupta', 'Verma', 'Kumar'])}",
            "respondent": f"{random.choice(['Mr.', 'Mrs.', 'Ms.', 'State of'])} {random.choice(['Raj', 'Sunita', 'Mohan', 'Deepak', 'Anita'] + INDIAN_STATES)} {random.choice(['Singh', 'Sharma', 'Patel', 'Gupta', 'Verma', 'Kumar', ''])}",
        }
        expanded_cases.append(case)

    return expanded_cases


# Generate expanded traffic violations
def generate_expanded_traffic_violations():
    """Generate expanded traffic violations data"""
    violation_types = [
        "Red Light",
        "Speeding",
        "Driving Without License",
        "Drunk Driving",
        "No Helmet",
        "No Seatbelt",
        "Wrong Side Driving",
        "Illegal Parking",
        "Dangerous Driving",
        "Using Mobile While Driving",
        "Lane Violation",
    ]

    state_codes = {
        "Delhi": "DL",
        "Maharashtra": "MH",
        "Karnataka": "KA",
        "Tamil Nadu": "TN",
        "Uttar Pradesh": "UP",
        "Rajasthan": "RJ",
        "Gujarat": "GJ",
        "Haryana": "HR",
        "Punjab": "PB",
        "Kerala": "KL",
        "Telangana": "TS",
        "Andhra Pradesh": "AP",
        "West Bengal": "WB",
        "Bihar": "BR",
        "Madhya Pradesh": "MP",
        "Odisha": "OD",
    }

    expanded_violations = []
    for i in range(100):  # Generate 100 violations
        state = random.choice(list(state_codes.keys()))
        vehicle_number = f"{state_codes[state]}-{random.randint(1,99):02d}-{random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}-{random.randint(1000,9999)}"

        violation_date = datetime.now() - timedelta(days=random.randint(1, 60))

        violation = {
            "violation_id": f"TV-{violation_date.year}-{random.randint(1, 9999):04d}",
            "vehicle_number": vehicle_number,
            "violation_type": random.choice(violation_types),
            "location": f"{random.choice(['Main Road', 'Highway', 'Market Area', 'Ring Road'])}, {state}",
            "date": violation_date.strftime("%Y-%m-%d"),
            "fine_amount": random.choice([500, 1000, 1500, 2000, 5000, 10000]),
            "status": random.choice(["Paid", "Unpaid", "Disputed", "Under Review"]),
            "officer": f"Officer {random.choice(['Kumar', 'Singh', 'Sharma', 'Verma', 'Patel', 'Gupta'])}",
            "evidence_available": random.choice([True, False]),
            "appeal_deadline": (violation_date + timedelta(days=30)).strftime(
                "%Y-%m-%d"
            ),
        }
        expanded_violations.append(violation)

    return expanded_violations


# Generate expanded fast track courts
def generate_expanded_fast_track_courts():
    """Generate expanded fast track courts data"""
    ftc_types = [
        "POCSO",
        "Women Safety",
        "Commercial Disputes",
        "Land Acquisition",
        "Senior Citizens",
    ]
    expanded_ftc = []

    for state in INDIAN_STATES + ["Delhi"]:
        for ftc_type in ftc_types:
            if random.random() < 0.7:  # 70% chance for each state to have each type
                cases_handled = random.randint(100, 500)
                cases_disposed = random.randint(
                    int(cases_handled * 0.5), int(cases_handled * 0.9)
                )

                ftc = {
                    "court_id": f"FTC-{state[:2].upper()}-{ftc_type[:3].upper()}-{random.randint(1, 99):02d}",
                    "name": f"{ftc_type} Fast Track Court - {state}",
                    "type": ftc_type,
                    "location": f"{random.choice(['East', 'West', 'North', 'South', 'Central'])} {state}",
                    "cases_handled": cases_handled,
                    "cases_disposed": cases_disposed,
                    "pending_cases": cases_handled - cases_disposed,
                    "judge": f"Justice {random.choice(['Anil', 'Sunita', 'Rajesh', 'Priya', 'Vikram', 'Deepika', 'Sanjay'])} {random.choice(['Kumar', 'Singh', 'Sharma', 'Verma', 'Patel', 'Gupta'])}",
                    "establishment_year": random.randint(2010, 2023),
                    "disposal_rate": f"{random.randint(50, 90)}%",
                }
                expanded_ftc.append(ftc)

    return expanded_ftc


# Generate expanded live streaming
def generate_expanded_live_streams():
    """Generate expanded live streaming data"""
    expanded_streams = []

    courts = ["Supreme Court"] + [f"{state} High Court" for state in INDIAN_STATES[:15]]
    case_numbers = [get_random_case_number() for _ in range(20)]

    # Live streams
    for i in range(5):
        court = random.choice(courts)
        stream = {
            "stream_id": f"LS-{datetime.now().year}-{random.randint(1, 999):03d}",
            "court": court,
            "case_number": random.choice(case_numbers),
            "stream_url": f"https://stream.courts.gov.in/{court.lower().replace(' ', '')}/{random.randint(1, 999):03d}",
            "status": "Live",
            "viewers": random.randint(500, 5000),
            "stream_quality": f"{random.choice(['720p', '1080p', '480p'])}",
            "started_at": (
                datetime.now() - timedelta(minutes=random.randint(5, 120))
            ).strftime("%Y-%m-%d %H:%M:%S"),
            "bench": f"{random.randint(2, 5)}-Judge Bench",
        }
        expanded_streams.append(stream)

    # Scheduled streams
    for i in range(15):
        court = random.choice(courts)
        scheduled_time = datetime.now() + timedelta(
            days=random.randint(1, 14), hours=random.randint(1, 8)
        )
        stream = {
            "stream_id": f"LS-{datetime.now().year}-{random.randint(1000, 9999):04d}",
            "court": court,
            "case_number": random.choice(case_numbers),
            "stream_url": f"https://stream.courts.gov.in/{court.lower().replace(' ', '')}/{random.randint(1, 999):03d}",
            "status": "Scheduled",
            "schedule_time": scheduled_time.strftime("%Y-%m-%d %H:%M:%S"),
            "expected_duration": f"{random.randint(1, 5)} hours",
            "description": f"Hearing on {random.choice(['PIL', 'Appeal', 'Review Petition', 'Constitutional Matter'])}",
            "notification_enabled": random.choice([True, False]),
        }
        expanded_streams.append(stream)

    return expanded_streams


# Generate legal aid centers data
def generate_legal_aid_centers():
    """Generate legal aid centers data for all states"""
    legal_aid_centers = []

    for state in INDIAN_STATES + UNION_TERRITORIES:
        centers_count = random.randint(5, 30)
        for i in range(centers_count):
            center = {
                "center_id": f"LAC-{state[:2].upper()}-{random.randint(1, 999):03d}",
                "name": f"{state} Legal Aid Center {i+1}",
                "type": random.choice(
                    [
                        "District Legal Services Authority",
                        "Taluka Legal Services Committee",
                        "NGO",
                        "Law School Clinic",
                    ]
                ),
                "address": f"{random.choice(['123', '456', '789'])} {random.choice(['Main Road', 'Gandhi Road', 'Court Complex'])}, {state}",
                "contact": f"+91 {random.randint(7000000000, 9999999999)}",
                "services": random.sample(
                    [
                        "Free Legal Advice",
                        "Legal Representation",
                        "Mediation",
                        "Lok Adalat",
                        "Legal Awareness",
                        "Para-Legal Training",
                    ],
                    k=random.randint(2, 6),
                ),
                "lawyers_count": random.randint(5, 50),
                "cases_handled_monthly": random.randint(20, 200),
                "established": random.randint(1990, 2020),
            }
            legal_aid_centers.append(center)

    return legal_aid_centers


# Generate prison statistics
def generate_prison_statistics():
    """Generate prison statistics data for all states"""
    prison_stats = {
        "national": {
            "total_prisons": 0,
            "total_capacity": 0,
            "current_population": 0,
            "occupancy_rate": 0,
            "undertrials_percentage": 0,
            "women_percentage": 0,
            "foreign_nationals": 0,
        },
        "state_wise": {},
    }

    total_prisons = 0
    total_capacity = 0
    total_population = 0
    total_undertrials = 0
    total_women = 0
    total_foreign = 0

    for state in INDIAN_STATES + UNION_TERRITORIES:
        prisons = random.randint(5, 50)
        capacity = prisons * random.randint(500, 2000)
        population = int(capacity * random.uniform(0.8, 1.5))  # 80% to 150% occupancy
        undertrials = int(population * random.uniform(0.6, 0.8))  # 60-80% undertrials
        women = int(population * random.uniform(0.03, 0.08))  # 3-8% women
        foreign = int(population * random.uniform(0.01, 0.05))  # 1-5% foreign nationals

        state_stats = {
            "prisons": prisons,
            "capacity": capacity,
            "population": population,
            "occupancy_rate": round(population / capacity * 100, 2),
            "undertrials": undertrials,
            "undertrials_percentage": round(undertrials / population * 100, 2),
            "women": women,
            "women_percentage": round(women / population * 100, 2),
            "foreign_nationals": foreign,
            "highest_crime_category": random.choice(
                [
                    "Property Offenses",
                    "Violent Crimes",
                    "Drug Offenses",
                    "Economic Offenses",
                ]
            ),
        }

        prison_stats["state_wise"][state] = state_stats

        # Update national totals
        total_prisons += prisons
        total_capacity += capacity
        total_population += population
        total_undertrials += undertrials
        total_women += women
        total_foreign += foreign

    # Calculate national statistics
    prison_stats["national"]["total_prisons"] = total_prisons
    prison_stats["national"]["total_capacity"] = total_capacity
    prison_stats["national"]["current_population"] = total_population
    prison_stats["national"]["occupancy_rate"] = round(
        total_population / total_capacity * 100, 2
    )
    prison_stats["national"]["undertrials_percentage"] = round(
        total_undertrials / total_population * 100, 2
    )
    prison_stats["national"]["women_percentage"] = round(
        total_women / total_population * 100, 2
    )
    prison_stats["national"]["foreign_nationals"] = total_foreign

    return prison_stats


def get_court_vacancy(state=None):
    """Get current court vacancy data, optionally filtered by state"""
    if not state:
        return COURTS_DATA

    # Check for state abbreviation
    state_lower = state.lower()
    if state_lower in STATE_ABBREVIATIONS:
        state = STATE_ABBREVIATIONS[state_lower]

    # Filter high courts for the requested state
    high_courts = [hc for hc in COURTS_DATA["high_courts"] if state.lower() in hc["name"].lower()]

    # Filter district courts for the requested state
    district_courts = [dc for dc in COURTS_DATA["district_courts"] if state.lower() == dc["state"].lower()]

    # Return filtered data
    return {
        "supreme_court": COURTS_DATA["supreme_court"],
        "high_courts": high_courts,
        "district_courts": district_courts,
    }


def match_state_from_query(query):
    """Extract state name from a query about courts"""
    query = query.lower()

    # Check for exact state names
    for state in INDIAN_STATES + UNION_TERRITORIES:
        if state.lower() in query:
            return state

    # Check for abbreviations
    for abbr, state in STATE_ABBREVIATIONS.items():
        if abbr in query.split() or f" {abbr} " in query or f" {abbr}." in query:
            return state

    return None


def get_case_status(case_number=None):
    """Get case status by case number"""
    if case_number:
        # Find the specific case
        case = next(
            (case for case in CASE_STATUS if case["case_number"] == case_number), None
        )
        if case:
            return case
    return CASE_STATUS


def get_traffic_violation(violation_id=None):
    """Get traffic violation details"""
    global TRAFFIC_VIOLATIONS
    if not isinstance(TRAFFIC_VIOLATIONS, list) or len(TRAFFIC_VIOLATIONS) < 10:
        TRAFFIC_VIOLATIONS = generate_expanded_traffic_violations()

    if violation_id:
        return next(
            (
                violation
                for violation in TRAFFIC_VIOLATIONS
                if violation["violation_id"] == violation_id
            ),
            None,
        )
    return TRAFFIC_VIOLATIONS


def get_fast_track_courts():
    """Get fast track courts data"""
    global FAST_TRACK_COURTS
    if not isinstance(FAST_TRACK_COURTS, list) or len(FAST_TRACK_COURTS) < 10:
        FAST_TRACK_COURTS = generate_expanded_fast_track_courts()
    return FAST_TRACK_COURTS


def get_live_streams():
    """Get live streaming data"""
    global LIVE_STREAMING
    if not isinstance(LIVE_STREAMING, list) or len(LIVE_STREAMING) < 10:
        LIVE_STREAMING = generate_expanded_live_streams()
    return LIVE_STREAMING


def get_legal_aid_centers():
    """Get legal aid centers data"""
    global LEGAL_AID_CENTERS
    if not "LEGAL_AID_CENTERS" in globals() or not LEGAL_AID_CENTERS:
        LEGAL_AID_CENTERS = generate_legal_aid_centers()
    return LEGAL_AID_CENTERS


def get_prison_statistics():
    """Get prison statistics data"""
    global PRISON_STATISTICS
    if not "PRISON_STATISTICS" in globals() or not PRISON_STATISTICS:
        PRISON_STATISTICS = generate_prison_statistics()
    return PRISON_STATISTICS


def get_random_case_number():
    """Generate a random case number"""
    case_types = ["CRL", "CIV", "WP", "CP", "ARB", "TAX", "FAM"]
    return f"{random.choice(case_types)}-{datetime.now().year}-{random.randint(1, 999):03d}"


def get_random_violation_id():
    """Generate a random violation ID"""
    return f"TV-{datetime.now().year}-{random.randint(1, 999):03d}"


def export_to_json(data_type):
    """Export a specific data type to a JSON file"""
    try:
        if data_type == "courts":
            data = get_court_vacancy()
            filename = "court_vacancy_data.json"
        elif data_type == "cases":
            data = get_case_status()
            filename = "case_status_data.json"
        elif data_type == "violations":
            data = get_traffic_violation()
            filename = "traffic_violations_data.json"
        elif data_type == "fast_track":
            data = get_fast_track_courts()
            filename = "fast_track_courts_data.json"
        elif data_type == "live_streams":
            data = get_live_streams()
            filename = "live_streams_data.json"
        elif data_type == "legal_aid":
            data = get_legal_aid_centers()
            filename = "legal_aid_centers_data.json"
        elif data_type == "prisons":
            data = get_prison_statistics()
            filename = "prison_statistics_data.json"
        else:
            print(f"Unknown data type: {data_type}")
            return False

        output_dir = "data/exports"
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(
            f"Successfully exported {data_type} data to {os.path.join(output_dir, filename)}"
        )
        return True
    except Exception as e:
        print(f"Error exporting data: {e}")
        return False


def export_all_data():
    """Export all data types to JSON files"""
    data_types = [
        "courts",
        "cases",
        "violations",
        "fast_track",
        "live_streams",
        "legal_aid",
        "prisons",
    ]
    results = {}

    for data_type in data_types:
        results[data_type] = export_to_json(data_type)

    return results


# If this file is run directly, demonstrate the functionality
if __name__ == "__main__":
    print("AI-Vakeel Dummy Data Generator")
    print("-" * 40)

    # Example usage
    action = input(
        "Choose an action:\n1. View data examples\n2. Export all data to JSON\n3. Export specific data type\nEnter choice (1-3): "
    )

    if action == "1":
        data_type = input(
            "\nChoose data type to view:\n1. Court Vacancy\n2. Case Status\n3. Traffic Violations\n4. Fast Track Courts\n5. Live Streams\n6. Legal Aid Centers\n7. Prison Statistics\nEnter choice (1-7): "
        )

        if data_type == "1":
            courts = get_court_vacancy()
            print(f"\nSupreme Court: {courts['supreme_court']['vacancies']} vacancies")
            print(
                f"Sample High Court: {courts['high_courts'][0]['name']} - {courts['high_courts'][0]['vacancies']} vacancies"
            )
        elif data_type == "2":
            cases = get_case_status()
            print(f"\nSample Case: {cases[0]['case_number']} in {cases[0]['court']}")
            print(
                f"Status: {cases[0]['status']}, Next Hearing: {cases[0]['next_hearing']}"
            )
        elif data_type == "3":
            violations = get_traffic_violation()
            print(f"\nSample Violation: {violations[0]['violation_id']}")
            print(
                f"Type: {violations[0]['violation_type']}, Fine: ₹{violations[0]['fine_amount']}"
            )
        elif data_type == "4":
            ftc = get_fast_track_courts()
            print(f"\nSample Fast Track Court: {ftc[0]['name']}")
            print(
                f"Cases Handled: {ftc[0]['cases_handled']}, Disposed: {ftc[0]['cases_disposed']}"
            )
        elif data_type == "5":
            streams = get_live_streams()
            print(f"\nSample Live Stream: {streams[0]['court']}")
            print(f"Status: {streams[0]['status']}")
        elif data_type == "6":
            centers = get_legal_aid_centers()
            print(f"\nSample Legal Aid Center: {centers[0]['name']}")
            print(f"Services: {', '.join(centers[0]['services'])}")
        elif data_type == "7":
            prison_stats = get_prison_statistics()
            print(f"\nNational Prison Statistics:")
            print(f"Total prisons: {prison_stats['national']['total_prisons']}")
            print(f"Occupancy rate: {prison_stats['national']['occupancy_rate']}%")
        else:
            print("Invalid choice")

    elif action == "2":
        results = export_all_data()
        success = all(results.values())
        if success:
            print("\nAll data exported successfully to data/exports/ directory!")
        else:
            print("\nSome exports failed. Check the error messages above.")

    elif action == "3":
        data_type = input(
            "\nChoose data type to export:\n1. Court Vacancy\n2. Case Status\n3. Traffic Violations\n4. Fast Track Courts\n5. Live Streams\n6. Legal Aid Centers\n7. Prison Statistics\nEnter choice (1-7): "
        )

        type_map = {
            "1": "courts",
            "2": "cases",
            "3": "violations",
            "4": "fast_track",
            "5": "live_streams",
            "6": "legal_aid",
            "7": "prisons",
        }

        if data_type in type_map:
            result = export_to_json(type_map[data_type])
            if result:
                print(f"\nData exported successfully to data/exports/ directory!")
            else:
                print("\nExport failed. Check the error messages above.")
        else:
            print("Invalid choice")

    else:
        print("Invalid choice")
