import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SKILLS = [
    "Python",
    "Java",
    "C++",
    "C",
    "JavaScript",
    "TypeScript",
    "HTML",
    "CSS",
    "React",
    "Node.js",
    "Express",
    "Django",
    "Flask",
    "FastAPI",
    "SQL",
    "MySQL",
    "PostgreSQL",
    "MongoDB",
    "SQLite",
    "Artificial Intelligence",
    "Machine Learning",
    "Deep Learning",
    "Generative AI",
    "Natural Language Processing",
    "NLP",
    "Computer Vision",
    "Neural Networks",
    "TensorFlow",
    "PyTorch",
    "Scikit-learn",
    "Pandas",
    "NumPy",
    "OpenCV",
    "Data Analysis",
    "Data Science",
    "Data Visualization",
    "Matplotlib",
    "Power BI",
    "Excel",
    "AWS",
    "Azure",
    "Google Cloud",
    "Docker",
    "Kubernetes",
    "Git",
    "GitHub",
    "Linux",
    "Jupyter",
    "Google Colab",
    "VS Code",
    "REST API",
    "API",
    "JSON",
    "ONNX",
    "CUDA",
]

SKILL_ALIASES = {
    "sql": ["sql", "mysql", "postgresql", "sqlite"],
    "python": ["python"],
    "javascript": ["javascript", "js"],
    "typescript": ["typescript", "ts"],
    "machine learning": ["machine learning", "ml"],
    "artificial intelligence": ["artificial intelligence", "ai"],
    "natural language processing": [
        "natural language processing",
        "nlp"
    ],
    "computer vision": ["computer vision", "opencv"],
    "deep learning": ["deep learning"],
    "fastapi": ["fastapi"],
    "django": ["django"],
    "flask": ["flask"],
    "react": ["react", "react.js"],
    "node.js": ["node.js", "nodejs"],
    "git": ["git", "github"],
    "github": ["github"],
    "docker": ["docker"],
    "pytorch": ["pytorch"],
    "tensorflow": ["tensorflow"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "rest api": ["rest api", "restful api", "rest"],
    "data analysis": ["data analysis", "data analytics"],
}

CANONICAL_SKILLS = {
    "mysql": "SQL",
    "postgresql": "SQL",
    "sqlite": "SQL",
    "nlp": "Natural Language Processing",
    "ml": "Machine Learning",
    "ai": "Artificial Intelligence",
    "opencv": "Computer Vision",
    "js": "JavaScript",
    "ts": "TypeScript",
    "nodejs": "Node.js",
    "sklearn": "Scikit-learn",
    "restful api": "REST API",
    "rest": "REST API",
    "data analytics": "Data Analysis",
    "github": "Git",
}

SKILL_IMPORTANCE = {
    "python": "core",
    "java": "core",
    "javascript": "core",
    "typescript": "core",
    "machine learning": "core",
    "artificial intelligence": "core",
    "deep learning": "core",
    "natural language processing": "core",
    "computer vision": "core",
    "fastapi": "core",
    "django": "core",
    "flask": "core",
    "react": "core",
    "sql": "core",
    "pytorch": "core",
    "tensorflow": "core",

    "git": "supporting",
    "github": "supporting",
    "docker": "supporting",
    "html": "supporting",
    "css": "supporting",
    "numpy": "supporting",
    "pandas": "supporting",
    "matplotlib": "supporting",
    "rest api": "supporting",
}


# =========================================================
# NORMALIZE SKILL
# =========================================================

def normalize_skill(skill):

    skill_key = skill.lower().strip()

    if skill_key in CANONICAL_SKILLS:
        return CANONICAL_SKILLS[skill_key]

    for known_skill in SKILLS:

        if known_skill.lower() == skill_key:
            return known_skill

    return skill


def extract_skills(text):

    found_skills = []

    text = text.lower()

    for skill in SKILLS:

        skill_key = skill.lower()

        aliases = SKILL_ALIASES.get(
            skill_key,
            [skill_key]
        )

        for alias in aliases:

            pattern = (
                r"(?<!\w)"
                + re.escape(alias)
                + r"(?!\w)"
            )

            if re.search(pattern, text):

                canonical_skill = normalize_skill(alias)

                if canonical_skill not in found_skills:
                    found_skills.append(canonical_skill)

                break

    return found_skills


# =========================================================
# TEXT SIMILARITY
# =========================================================

def calculate_text_similarity(
    resume_text,
    job_description
):

    documents = [
        resume_text,
        job_description
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(
        documents
    )

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0]

    return round(
        similarity * 100,
        2
    )


# =========================================================
# RECOMMENDATIONS
# =========================================================

def generate_recommendations(
    missing_skills,
    matched_skills
):

    recommendations = []

    core_missing = []
    supporting_missing = []

    for skill in missing_skills:

        importance = SKILL_IMPORTANCE.get(
            skill.lower(),
            "supporting"
        )

        if importance == "core":
            core_missing.append(skill)
        else:
            supporting_missing.append(skill)

    if core_missing:

        recommendations.append(
            "Focus on developing these core skills: "
            + ", ".join(core_missing)
            + "."
        )

    if supporting_missing:

        recommendations.append(
            "Consider adding these supporting skills: "
            + ", ".join(supporting_missing)
            + "."
        )

    if matched_skills:

        recommendations.append(
            "Your resume demonstrates relevant "
            "experience in: "
            + ", ".join(matched_skills)
            + "."
        )

    if not missing_skills:

        recommendations.append(
            "Your listed skills cover all the "
            "skills detected in this job description."
        )

    return recommendations


# =========================================================
# RESUME QUALITY ANALYSIS
# =========================================================

def analyze_resume_quality(resume_text):

    text = resume_text.strip()

    words = re.findall(
        r"\b[\w+#.-]+\b",
        text
    )

    word_count = len(words)

    text_lower = text.lower()


    # -----------------------------------------------------
    # CONTACT INFORMATION
    # -----------------------------------------------------

    email_found = bool(
        re.search(
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            text
        )
    )

    phone_found = bool(
        re.search(
            r"(?:\+91[-\s]?)?[6-9]\d{9}\b",
            text
        )
    )


    # -----------------------------------------------------
    # RESUME SECTIONS
    # -----------------------------------------------------

    section_patterns = {

        "education": [
            "education",
            "academic background",
            "qualification"
        ],

        "experience": [
            "experience",
            "work experience",
            "employment"
        ],

        "projects": [
            "projects",
            "academic projects",
            "personal projects"
        ],

        "skills": [
            "skills",
            "technical skills",
            "technologies"
        ],

        "certifications": [
            "certifications",
            "certificates"
        ],

        "achievements": [
            "achievements",
            "awards",
            "accomplishments"
        ]
    }


    detected_sections = []


    for section, keywords in section_patterns.items():

        for keyword in keywords:

            if keyword in text_lower:

                detected_sections.append(section)

                break


    detected_sections = list(
        dict.fromkeys(detected_sections)
    )


    # -----------------------------------------------------
    # QUALITY SCORE
    # -----------------------------------------------------

    score = 0

    if 300 <= word_count <= 1200:
        score += 25

    elif 150 <= word_count < 300:
        score += 15

    elif word_count > 1200:
        score += 10

    else:
        score += 5


    if email_found:
        score += 15


    if phone_found:
        score += 15


    section_score = min(
        len(detected_sections) * 5,
        30
    )

    score += section_score


    if "skills" in detected_sections:
        score += 5


    score = min(score, 100)


    # -----------------------------------------------------
    # QUALITY SUGGESTIONS
    # -----------------------------------------------------

    suggestions = []


    if not email_found:

        suggestions.append(
            "Add a professional email address."
        )


    if not phone_found:

        suggestions.append(
            "Add a contact phone number."
        )


    if word_count < 300:

        suggestions.append(
            "Your resume appears short. "
            "Consider adding relevant projects, "
            "experience, or achievements."
        )


    if word_count > 1200:

        suggestions.append(
            "Your resume may be too long. "
            "Consider removing less relevant information."
        )


    required_sections = [
        "education",
        "experience",
        "projects",
        "skills"
    ]


    for section in required_sections:

        if section not in detected_sections:

            suggestions.append(
                "Consider adding a "
                + section
                + " section."
            )


    if not suggestions:

        suggestions.append(
            "Your resume contains the main "
            "elements detected by the analyzer."
        )


    return {

        "resume_quality_score": score,

        "word_count": word_count,

        "email_detected": email_found,

        "phone_detected": phone_found,

        "detected_sections": detected_sections,

        "suggestions": suggestions
    }


# =========================================================
# MAIN MATCHING FUNCTION
# =========================================================

def match_job_description(
    resume_text,
    job_description
):

    resume_skills = extract_skills(
        resume_text
    )

    job_skills = extract_skills(
        job_description
    )


    # -----------------------------------------------------
    # MATCH SKILLS
    # -----------------------------------------------------

    matched_skills = []
    missing_skills = []


    for job_skill in job_skills:

        if job_skill in resume_skills:

            matched_skills.append(job_skill)

        else:

            missing_skills.append(job_skill)


    # -----------------------------------------------------
    # WEIGHTED SKILL SCORE
    # -----------------------------------------------------

    total_weight = 0
    matched_weight = 0


    for skill in job_skills:

        importance = SKILL_IMPORTANCE.get(
            skill.lower(),
            "supporting"
        )

        weight = (
            2
            if importance == "core"
            else 1
        )

        total_weight += weight

        if skill in matched_skills:

            matched_weight += weight


    if total_weight > 0:

        skill_match_score = (
            matched_weight /
            total_weight
        ) * 100

    else:

        skill_match_score = 0


    # -----------------------------------------------------
    # NLP SIMILARITY
    # -----------------------------------------------------

    text_similarity_score = (
        calculate_text_similarity(
            resume_text,
            job_description
        )
    )


    # -----------------------------------------------------
    # OVERALL SCORE
    # -----------------------------------------------------

    overall_score = (
        (skill_match_score * 0.6)
        +
        (text_similarity_score * 0.4)
    )


    # -----------------------------------------------------
    # SKILL BREAKDOWN
    # -----------------------------------------------------

    skill_breakdown = []


    for skill in job_skills:

        importance = SKILL_IMPORTANCE.get(
            skill.lower(),
            "supporting"
        )

        status = (
            "matched"
            if skill in matched_skills
            else "missing"
        )

        skill_breakdown.append({

            "skill": skill,

            "importance": importance,

            "status": status
        })


    # -----------------------------------------------------
    # RECOMMENDATIONS
    # -----------------------------------------------------

    recommendations = generate_recommendations(
        missing_skills,
        matched_skills
    )


    # -----------------------------------------------------
    # RESUME QUALITY
    # -----------------------------------------------------

    resume_quality = analyze_resume_quality(
        resume_text
    )


    # -----------------------------------------------------
    # FINAL RESULT
    # -----------------------------------------------------

    return {

        "resume_skills": resume_skills,

        "job_skills": job_skills,

        "matched_skills": matched_skills,

        "missing_skills": missing_skills,

        "skill_breakdown": skill_breakdown,

        "skill_match_score": round(
            skill_match_score,
            2
        ),

        "text_similarity_score":
            text_similarity_score,

        "overall_match_score": round(
            overall_score,
            2
        ),

        "recommendations": recommendations,

        "resume_quality": resume_quality
    }