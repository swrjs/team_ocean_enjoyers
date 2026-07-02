import json
import csv
import time
from datetime import datetime
from typing import List, Dict
from pathlib import Path

import langchain
print("LANGCHAIN IS BEING LOADED FROM:", langchain.__file__)

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class CustomEnsembleRetriever:
    # merge sparse + dense results using rrf sorting math
    def __init__(self, retrievers: list, weights: List[float]):
        self.retrievers = retrievers
        self.weights = weights

    def invoke(self, query: str) -> List[Document]:
        bm25_docs = self.retrievers[0].invoke(query)
        chroma_docs = self.retrievers[1].invoke(query)
        
        doc_scores = {}
        doc_map = {}
        constant = 60 # standard rrf rank smoothing param
        
        for rank, doc in enumerate(bm25_docs):
            cid = doc.metadata.get("candidate_id")
            if cid:
                doc_map[cid] = doc
                doc_scores[cid] = doc_scores.get(cid, 0.0) + self.weights[0] * (1.0 / (rank + constant))
                
        for rank, doc in enumerate(chroma_docs):
            cid = doc.metadata.get("candidate_id")
            if cid:
                doc_map[cid] = doc
                doc_scores[cid] = doc_scores.get(cid, 0.0) + self.weights[1] * (1.0 / (rank + constant))
        
        sorted_cids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        return [doc_map[cid] for cid in sorted_cids]


def parse_date(date_str: str) -> datetime:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return datetime.min


def load_and_filter_candidates(filepath: str) -> List[Dict]:
    # line by line reading to keep the colab ram footprint tiny
    survivors = []
    current_date = datetime(2026, 6, 29) 
    
    print(f"Opening full dataset file: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            signals = cand.get("redrob_signals", {})
            
            # check if they have gone completely dark for half a year
            last_active = parse_date(signals.get("last_active_date", "2000-01-01"))
            days_inactive = (current_date - last_active).days
            if days_inactive > 180:
                continue 
                
            # skip profile if they never respond to mesages
            response_rate = signals.get("recruiter_response_rate", 0)
            if response_rate < 0.10:
                continue 
                
            survivors.append(cand)
            
    return survivors


def get_embeddings_model():
    # checking local folders first so cloud execution dont break paths
    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path(".").resolve()

    model_path = base_dir / "local_model_weights"

    if model_path.exists() and any(model_path.iterdir()):
        print(f"-> Loading embeddings from local path: {model_path}")
        model_target = str(model_path)
    else:
        print("-> Local weights not found. Streaming model from hub...")
        model_target = "sentence-transformers/all-MiniLM-L6-v2"

    return HuggingFaceEmbeddings(
        model_name=model_target,
        model_kwargs={'device': 'cpu'}
    )


def build_hybrid_retriever(candidates: List[Dict], query: str) -> List[Document]:
    # format dict payload into a clean string chunk for langchain
    all_docs = []
    for cand in candidates:
        profile = cand.get("profile", {})
        skills = ", ".join([s.get("name", "") for s in cand.get("skills", [])])
        
        history_text = ""
        for job in cand.get("career_history", []):
            history_text += f"{job.get('title')} at {job.get('company')} ({job.get('duration_months')} months). "
        
        content = f"Title: {profile.get('current_title')}\n" \
                  f"Summary: {profile.get('summary')}\n" \
                  f"Skills: {skills}\n" \
                  f"History: {history_text}"
                
        all_docs.append(Document(page_content=content, metadata={"candidate_id": cand.get("candidate_id")}))
        
    # capping at 1000 candidates otherwise cpu dense embedding takes ages
    print(f"   Indexing {len(all_docs)} active profiles into primary BM25 pre-filter...")
    initial_bm25 = BM25Retriever.from_documents(all_docs)
    initial_bm25.k = min(1000, len(all_docs)) 
    narrowed_docs = initial_bm25.invoke(query)
    
    # building the mini vector store on the downsampled subset
    print("   Computing ChromaDB dense vector embeddings for top 1000 context matches...")
    embeddings = get_embeddings_model()

    vectorstore = Chroma.from_documents(narrowed_docs, embeddings)
    
    # keeping search windows small to smash through the 5 min limit
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 200})
    precise_bm25 = BM25Retriever.from_documents(narrowed_docs)
    precise_bm25.k = 200
    
    ensemble_retriever = CustomEnsembleRetriever(
        retrievers=[precise_bm25, chroma_retriever], 
        weights=[0.4, 0.6] 
    )
    
    return ensemble_retriever.invoke(query)


def apply_signal_penalties(retrieved_docs: List[Document], original_candidates: List[Dict]) -> List[Dict]:
    # adjust the hybrid score using behavioral metrics multiplier
    cand_lookup = {c["candidate_id"]: c for c in original_candidates}
    scored_candidates = []
    max_base_score = len(retrieved_docs)
    
    for rank, doc in enumerate(retrieved_docs):
        cid = doc.metadata["candidate_id"]
        cand_data = cand_lookup[cid]
        signals = cand_data.get("redrob_signals", {})
        
        base_score = (max_base_score - rank) / max_base_score 
        
        interview_rate = signals.get("interview_completion_rate", 1.0) 
        response_rate = signals.get("recruiter_response_rate", 1.0)
        
        final_hybrid_score = base_score * interview_rate * response_rate
        
        scored_candidates.append({
            "candidate_id": cid,
            "data": cand_data,
            "hybrid_score": final_hybrid_score
        })
        
    scored_candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return scored_candidates[:200] 


def final_local_judge(top_candidates: List[Dict]) -> List[Dict]:
    # filter out non tech keyword stuffers from the final list
    final_results = []
    
    TRAP_TITLES = [
        "marketing", "sales", "hr", "recruiter", "talent", "writer", 
        "g  raphic", "designer", "mechanical", "support", "accountant", 
        "executive", "business analyst"
    ]
    
    print(f"   Running local algorithmic checks on top {len(top_candidates)} candidates...")
    for cand in top_candidates:
        profile = cand["data"].get("profile", {})
        title = profile.get("current_title", "").lower()
        experience = profile.get("years_of_experience", 0)
        
        base_score = cand["hybrid_score"]
        
        # if title is a trap string drop score to zero unless they are actually an engineer
        if any(trap in title for trap in TRAP_TITLES) and "engineer" not in title and "developer" not in title:
            final_score = 0.0
            reasoning = f"Title trap detected: {profile.get('current_title')} is not an engineer."
        else:
            final_score = base_score
            reasoning = f"{profile.get('current_title')} with {experience} yrs; signal match."

        final_results.append({
            "candidate_id": cand["candidate_id"],
            "score": final_score,
            "reasoning": reasoning
        })
        
    # sequential stable sorting so tie breaks always match submission spec
    final_results.sort(key=lambda x: x["candidate_id"])
    final_results.sort(key=lambda x: x["score"], reverse=True)
    
    return final_results[:100] 


def main():
    start_time = time.time()
    dataset_file = "candidates.jsonl"
    output_csv = "ocean_enjoyers.csv"
    
    query = "Senior AI Engineer machine learning embeddings RAG fine-tuning python LLM pipelines"
    
    print("1. Purging ghosts and inactive candidates from full 100k pool...")
    survivors = load_and_filter_candidates(dataset_file)
    print(f"   {len(survivors)} candidates survived the initial purge.")
    
    print("2. Launching hybrid filtering and search index passes...")
    retrieved_docs = build_hybrid_retriever(survivors, query)
    
    print("3. Applying Behavioral Signal Math Penalties...")
    top_200 = apply_signal_penalties(retrieved_docs, survivors)
    
    print("4. Running candidates pool through pure-Python local judge verification...")
    final_100 = final_local_judge(top_200)
    
    print("5. Formatting and writing submission CSV...")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        
        for rank, cand in enumerate(final_100, 1):
            writer.writerow([
                cand['candidate_id'], 
                rank, 
                f"{cand['score']:.4f}", 
                cand['reasoning']
            ])
            
    print(f"Done! Created submission file: {output_csv}")
    print(f"Run validation test: python validate_submission.py {output_csv}")
    
    end_time = time.time()  
    total_time = end_time - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")


if __name__ == "__main__":
    main()