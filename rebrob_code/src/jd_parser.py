"""
jd_parser.py — Job Description parser for the Redrob ranking system.

Parses the Senior AI Engineer JD into a structured ``JDProfile`` dataclass.
Since the JD is a .docx file and we want the ranking pipeline to run without
python-docx at runtime, this module returns a fully hardcoded profile derived
from the JD document.

Usage:
    from jd_parser import get_jd_profile
    jd = get_jd_profile()
    print(jd.title, jd.must_have_skills[:5])
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ANTI_SKILLS, MUST_HAVE_SKILLS, NICE_TO_HAVE_SKILLS


@dataclass(frozen=True, slots=True)
class JDProfile:
    """Structured representation of a Job Description.

    All fields are populated from the JD document at construction time.
    The ``frozen=True`` flag ensures immutability — the JD never changes
    during a ranking run.

    Attributes:
        title: Job title as stated in the JD.
        company: Hiring company name.
        location: Office location(s).
        experience_range: (min, max) years of experience required.
        ideal_experience: Sweet-spot experience in years.
        must_have_skills: Skills the JD explicitly requires.
        nice_to_have_skills: Skills the JD considers a plus.
        anti_skills: Skills that are negative signals (CV/speech/robotics).
        anti_patterns: Descriptions of candidate profiles to avoid.
        key_responsibilities: Core duties of the role.
        preferred_locations: Cities where the role is based.
        work_mode: Remote / hybrid / onsite.
        summary_text: Rich prose summary of the JD for embedding.
        requirements_text: Concatenated requirements for embedding.
        anti_pattern_text: Concatenated anti-patterns for anti-embedding.
    """

    title: str
    company: str
    location: str
    experience_range: tuple[float, float]
    ideal_experience: float
    must_have_skills: list[str] = field(default_factory=list)
    nice_to_have_skills: list[str] = field(default_factory=list)
    anti_skills: list[str] = field(default_factory=list)
    anti_patterns: list[str] = field(default_factory=list)
    key_responsibilities: list[str] = field(default_factory=list)
    preferred_locations: list[str] = field(default_factory=list)
    work_mode: str = "hybrid"
    summary_text: str = ""
    requirements_text: str = ""
    anti_pattern_text: str = ""


def get_jd_profile() -> JDProfile:
    """Return the hardcoded JDProfile for the Senior AI Engineer role.

    This function encapsulates every detail from the Redrob AI job
    description so that downstream scoring modules never need to parse
    the original .docx file.

    Returns:
        JDProfile: Fully populated job description profile.
    """
    return JDProfile(
        title="Senior AI Engineer — Founding Team",
        company="Redrob AI",
        location="Pune/Noida, India",
        experience_range=(5.0, 9.0),
        ideal_experience=7.0,
        must_have_skills=sorted(MUST_HAVE_SKILLS),
        nice_to_have_skills=sorted(NICE_TO_HAVE_SKILLS),
        anti_skills=sorted(ANTI_SKILLS),
        anti_patterns=[
            "Title chasers who hop jobs every 12-18 months chasing a better "
            "designation without deepening expertise.",
            "Framework-of-the-month enthusiasts who chase every new tool but "
            "have never shipped a production system end-to-end.",
            "Consulting-only careers entirely at IT services companies such as "
            "TCS, Wipro, Infosys, Cognizant, Accenture, HCL, Capgemini, or "
            "Tech Mahindra — with no product company or startup experience.",
            "Computer vision, speech recognition, or robotics specialists "
            "whose entire career is in CV/speech/robotics with no NLP, "
            "search, or retrieval experience.",
            "Pure researchers or academics who have published papers but never "
            "deployed a model to production or written production-grade code.",
            "Candidates with no recent hands-on coding — current role is pure "
            "management, architecture, or strategy with no code contributions "
            "in the last 2 years.",
            "Candidates whose skills are entirely non-technical (project "
            "management, marketing, sales) with no ML/AI background.",
        ],
        key_responsibilities=[
            "Own the intelligence layer: ranking, retrieval, and candidate-job "
            "matching across the Redrob platform.",
            "Ship the v2 ranking system — redesign and deploy a new "
            "embedding-based ranking pipeline that replaces the existing "
            "keyword-matching system.",
            "Set up evaluation infrastructure: build offline evaluation "
            "pipelines using NDCG, MRR, MAP, and design online A/B testing "
            "frameworks to measure real-world ranking quality.",
            "Build and maintain production embedding pipelines using "
            "sentence-transformers, OpenAI embeddings, or fine-tuned models.",
            "Integrate and operate vector databases (Pinecone, Weaviate, "
            "Qdrant, FAISS) for low-latency approximate nearest neighbor "
            "search at scale.",
            "Design hybrid retrieval strategies combining dense vector search "
            "with sparse keyword-based retrieval (BM25) for optimal recall.",
            "Develop learning-to-rank models using gradient-boosted trees "
            "(XGBoost, LightGBM) or neural rankers (cross-encoders).",
            "Collaborate with product and engineering to define ranking "
            "quality metrics and drive continuous improvement.",
            "Contribute to the LLM layer: prompt engineering, RAG pipelines, "
            "and potential fine-tuning (LoRA/QLoRA/PEFT) for HR-specific "
            "language understanding.",
            "Operate in an async-first, high-autonomy environment — write "
            "design docs, ship code, and iterate fast as part of the "
            "founding engineering team.",
        ],
        preferred_locations=[
            "Pune",
            "Noida",
        ],
        work_mode="hybrid",
        summary_text=(
            "Redrob AI, a Series A HR-tech startup, is hiring a Senior AI "
            "Engineer for its founding engineering team based in Pune or "
            "Noida, India (hybrid). The role owns the intelligence layer — "
            "ranking, retrieval, and candidate-job matching — and requires "
            "5-9 years of experience (sweet spot 6-8 years) building "
            "production ML systems. The ideal candidate has deep hands-on "
            "expertise in embeddings and semantic search using "
            "sentence-transformers or similar, has operated vector databases "
            "like Pinecone, Weaviate, Qdrant, or FAISS at scale, and is a "
            "strong Python engineer who can ship end-to-end. They should be "
            "comfortable setting up evaluation infrastructure (NDCG, MRR, "
            "MAP, A/B testing) and have experience with NLP, information "
            "retrieval, or recommendation systems. Nice-to-haves include "
            "LLM fine-tuning (LoRA/QLoRA/PEFT), learning-to-rank, HR-tech "
            "domain experience, distributed systems, and open-source "
            "contributions. The culture values shippers over researchers, "
            "async-first communication, and engineers who write code daily. "
            "This is a founding-team role with outsized impact at a company "
            "reimagining how hiring works with AI."
        ),
        requirements_text=(
            "The candidate must have 5 to 9 years of hands-on experience "
            "building production machine learning systems. "
            "They must have deep expertise in production embeddings and "
            "retrieval systems, including sentence-transformers, dense "
            "retrieval, and semantic search. "
            "Experience with vector databases such as Pinecone, Weaviate, "
            "Qdrant, Milvus, or FAISS is required for approximate nearest "
            "neighbor search at scale. "
            "Strong Python programming skills are mandatory. "
            "The candidate must have experience building evaluation "
            "frameworks using metrics like NDCG, MRR, MAP, and conducting "
            "A/B tests to measure ranking quality. "
            "Experience with NLP, natural language processing, text mining, "
            "and information retrieval is required. "
            "Familiarity with production ML tooling including PyTorch, "
            "TensorFlow, HuggingFace Transformers, MLOps, and model serving "
            "is expected. "
            "The candidate should have experience building recommendation "
            "systems, search ranking, or learning-to-rank models. "
            "Nice-to-have skills include LLM fine-tuning with LoRA, QLoRA, "
            "or PEFT, gradient-boosted ranking models (XGBoost, LightGBM), "
            "cross-encoder and bi-encoder architectures, RAG pipelines, "
            "distributed systems (Kubernetes, Docker, Kafka), HR-tech or "
            "recruiting technology domain experience, and open-source "
            "contributions."
        ),
        anti_pattern_text=(
            "Candidates who are title chasers — hopping jobs every 12 to 18 "
            "months chasing designations without deepening expertise — are "
            "not a fit. "
            "Framework-of-the-month enthusiasts who experiment with every new "
            "tool but have never shipped a production system are a red flag. "
            "Candidates whose entire career has been at IT services or "
            "consulting firms such as TCS, Wipro, Infosys, Cognizant, "
            "Accenture, HCL, Capgemini, or Tech Mahindra — with no product "
            "company or startup experience — do not match this role. "
            "Computer vision specialists, speech recognition engineers, and "
            "robotics-focused candidates whose entire background is in "
            "CV, speech, or robotics with no NLP, search, or retrieval "
            "experience are not relevant. "
            "Pure researchers and academics who have published papers but "
            "never deployed a model to production or written "
            "production-grade code are not suitable. "
            "Candidates with no recent hands-on coding — whose current role "
            "is pure management, architecture, or strategy with no code "
            "contributions in the last two years — do not fit. "
            "Candidates whose skills are entirely non-technical such as "
            "project management, marketing, or sales with no ML or AI "
            "background are not relevant to this role."
        ),
    )


if __name__ == "__main__":
    # Quick validation: print the JD profile
    jd = get_jd_profile()
    print(f"Title:       {jd.title}")
    print(f"Company:     {jd.company}")
    print(f"Location:    {jd.location}")
    print(f"Experience:  {jd.experience_range[0]}-{jd.experience_range[1]} yrs "
          f"(ideal: {jd.ideal_experience})")
    print(f"Work mode:   {jd.work_mode}")
    print(f"Must-haves:  {len(jd.must_have_skills)} skills")
    print(f"Nice-to-have:{len(jd.nice_to_have_skills)} skills")
    print(f"Anti-skills: {len(jd.anti_skills)} skills")
    print(f"Anti-patterns: {len(jd.anti_patterns)} patterns")
    print(f"Responsibilities: {len(jd.key_responsibilities)} items")
    print(f"Summary len: {len(jd.summary_text)} chars")
    print(f"Requirements len: {len(jd.requirements_text)} chars")
    print(f"Anti-pattern text len: {len(jd.anti_pattern_text)} chars")
    print("✓ JDProfile created successfully.")
