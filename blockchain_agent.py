# blockchain_agent.py
import os
import re
import json
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
from enum import Enum
from threading import Lock

# === Désactiver la télémétrie Chroma (critique) ===
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# LangChain
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import Tool

# Vectorstore
try:
    from langchain_chroma import Chroma
except ImportError:
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        from langchain.vectorstores import Chroma

# Web Search
try:
    from ddgs import DDGS
except ImportError:
    DDGS = None

# Autres
import concurrent.futures

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =====================================================================
# GLOBAL CACHE (simule Redis)
# =====================================================================
class SimpleCache:
    def __init__(self, ttl_seconds: int = 7200):  # 2 heures
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._ttl = ttl_seconds
        self._lock = Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                return None
            value, timestamp = self._cache[key]
            if time.time() - timestamp > self._ttl:
                del self._cache[key]
                return None
            return value

    def set(self, key: str, value: Any):
        with self._lock:
            self._cache[key] = (value, time.time())

    def invalidate(self, prefix: str = ""):
        with self._lock:
            if not prefix:
                self._cache.clear()
            else:
                keys = [k for k in self._cache if k.startswith(prefix)]
                for k in keys:
                    del self._cache[k]


# Instancier le cache global
global_cache = SimpleCache(ttl_seconds=7200)


# =====================================================================
# DATA MODELS
# =====================================================================
@dataclass
class DynamicCriteria:
    explicit_requirements: Dict[str, Any] = field(default_factory=dict)
    implicit_concerns: List[str] = field(default_factory=list)
    domain_context: str = ""
    discovered_keywords: set = field(default_factory=set)
    constraint_thresholds: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0


class UseCaseType(Enum):
    SUPPLY_CHAIN = "supply_chain"
    NFT = "nft"
    DEFI = "defi"
    GAMING = "gaming"
    IOT = "iot"
    HEALTHCARE = "healthcare"
    IDENTITY = "identity"
    ENTERPRISE = "enterprise"
    GENERAL = "general"


@dataclass
class ProjectRequirements:
    use_case: UseCaseType
    tps_required: int
    max_cost_per_tx: float
    max_latency_ms: int
    security_level: str
    interoperability_needs: List[str] = field(default_factory=list)
    compliance_needs: List[str] = field(default_factory=list)


# =====================================================================
# DYNAMIC DISCOVERY ENGINE
# =====================================================================
class DynamicDiscoveryEngine:
    def __init__(self):
        self._setup_nlp()
        self.patterns = self._init_patterns()

    def _setup_nlp(self):
        try:
            import spacy
            try:
                self.nlp = spacy.load("models/blockchain_ner_en")
                logging.info("✅ Custom NER model loaded")
            except OSError:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    logging.info("✅ Fallback to en_core_web_sm")
                except:
                    self.nlp = spacy.blank("en")
                    logging.warning("⚠️ Using blank English model")
        except ImportError:
            logging.warning("⚠️ spacy not installed. Text analysis limited to regex.")
            self.nlp = None

    def _init_patterns(self) -> Dict:
        return {
            "domains": {
                "supply_chain": r"supply.*chain|provenance|traceability|logistics",
                "nft": r"nft|digital art|gallery|collectible",
                "iot": r"iot|sensor|device|connectivity|offline",
                "defi": r"defi|finance|lending|yield|staking",
                "gaming": r"game|gaming|play.*to.*earn",
                "healthcare": r"health|medical|patient|hospital|clinical",
                "identity": r"identity|did|verifiable credential|ssi|gdpr|digital id",
                "enterprise": r"enterprise|business|corporate"
            },
            "metrics": {
                "tps": r"(\d+(?:,\d+)*)\s*(?:k|000)?\s*(?:tps|transactions? per second)",
                "cost": r"(\d*\.?\d+)\s*(?:usd|dollar|€)\s*per\s*(?:tx|transaction)",
                "latency": r"(\d+)\s*(?:ms|milliseconds?|seconds?).*finality"
            },
            "concerns": {
                "offline": r"(?:offline|low connectivity|disconnected|sync|synchronization)",
                "gdpr": r"(?:gdpr|compliance|privacy|rgpd)"
            }
        }

    def discover_requirements_dynamically(self, query: str) -> Tuple[DynamicCriteria, Optional[ProjectRequirements]]:
        cache_key = f"discovery:{hash(query.lower())}"
        cached = global_cache.get(cache_key)
        if cached:
            logger.info("🔁 Dynamic analysis cache hit")
            return cached

        start = time.time()
        criteria = DynamicCriteria()
        query_lower = query.lower()

        # Domaine
        for domain, pattern in self.patterns["domains"].items():
            if re.search(pattern, query_lower):
                criteria.domain_context = domain
                break
        if not criteria.domain_context:
            criteria.domain_context = "general"

        # TPS
        if tps_match := re.search(self.patterns["metrics"]["tps"], query_lower):
            criteria.explicit_requirements["tps"] = int(tps_match.group(1).replace(",", ""))

        # Coût
        if cost_match := re.search(self.patterns["metrics"]["cost"], query_lower):
            try:
                criteria.explicit_requirements["max_cost_per_tx"] = float(cost_match.group(1))
            except ValueError:
                pass

        # Latence
        if latency_match := re.search(self.patterns["metrics"]["latency"], query_lower):
            criteria.explicit_requirements["max_latency_ms"] = int(latency_match.group(1))

        # Préoccupations implicites
        if re.search(self.patterns["concerns"]["offline"], query_lower):
            criteria.implicit_concerns.append("offline")
        if re.search(self.patterns["concerns"]["gdpr"], query_lower):
            criteria.implicit_concerns.append("gdpr")

        # Mots-clés
        keywords = {"blockchain", "ethereum", "solana", "polygon", "hyperledger", "iota",
                    "did", "verifiable", "credential", "ssi", "gdpr", "offline"}
        criteria.discovered_keywords = {
            w for w in re.findall(r'\b\w{4,}\b', query_lower) if w.lower() in keywords
        }

        # Score de confiance
        criteria.confidence_score = 0.3 + 0.2 * bool(criteria.explicit_requirements) + 0.3 * bool(criteria.discovered_keywords)

        # Seuil de contraintes
        criteria.constraint_thresholds = {
            "min_tps": criteria.explicit_requirements.get("tps", 50),
            "max_cost_per_tx": criteria.explicit_requirements.get("max_cost_per_tx", 0.10),
            "max_latency_ms": criteria.explicit_requirements.get("max_latency_ms", 30000)
        }

        # Générer ProjectRequirements si confiance élevée
        project_req = None
        if criteria.confidence_score > 0.5:
            try:
                use_case = UseCaseType(criteria.domain_context)
            except ValueError:
                use_case = UseCaseType.GENERAL
            project_req = ProjectRequirements(
                use_case=use_case,
                tps_required=criteria.constraint_thresholds["min_tps"],
                max_cost_per_tx=criteria.constraint_thresholds["max_cost_per_tx"],
                max_latency_ms=criteria.constraint_thresholds["max_latency_ms"],
                security_level="high" if "gdpr" in criteria.implicit_concerns else "medium",
                compliance_needs=["GDPR"] if "gdpr" in criteria.implicit_concerns else []
            )

        result = (criteria, project_req)
        global_cache.set(cache_key, result)
        logger.info(f"⏱️ Dynamic Analysis: {time.time() - start:.2f}s")
        return result


# =====================================================================
# SIMPLE AGENT
# =====================================================================
class SimpleAgent:
    def __init__(self, llm, tools, agent_ref=None):
        self.llm = llm
        self.tools: Dict[str, Any] = {tool.name: tool for tool in tools}
        self.agent = agent_ref  # Référence à BlockchainAgent

    def _detect_query_type(self, query: str) -> str:
        query_lower = query.lower().strip()
        if any(word in query_lower for word in ["best", "recommend", "which blockchain", "what blockchain", "good for", "suited for", "optimal for", "choose", "select"]):
            return "recommendation"
        if "compare" in query_lower or "vs" in query_lower or "versus" in query_lower:
            return "comparison"
        if any(word in query_lower for word in ["explain", "what is", "define", "how does", "history of", "difference between", "transition from"]):
            return "explanation"
        return "general"

    def invoke(self, inputs: Dict[str, Any], analysis_result: Tuple = None) -> Dict[str, Any]:
        start_total = time.time()
        try:
            input_text = inputs.get("input", "")
            chat_history = inputs.get("chat_history", "")
            tool_descriptions = "\n".join([f"- {name}: {tool.description}" for name, tool in self.tools.items()])

            query_type = self._detect_query_type(input_text)
            criteria, project_req = analysis_result or self.agent.discovery_engine.discover_requirements_dynamically(input_text)
            context = f"Context: {project_req}" if project_req else ""

            # === Étape 2 : Web Research (avec cache) ===
            web_result = ""
            web_time = 0.0
            if query_type == "recommendation":
                cache_key = f"web:{hash(input_text)}"
                web_result = global_cache.get(cache_key)
                if web_result is None:
                    start_web = time.time()
                    web_tool = self.tools.get("Web_Research")
                    if web_tool:
                        try:
                            web_result = web_tool.func(input_text)
                            global_cache.set(cache_key, web_result)
                        except Exception as e:
                            logging.warning(f"Web search failed: {e}")
                            web_result = f"❌ Web search failed: {e}"
                    web_time = time.time() - start_web
                    logger.info(f"⏱️ Web Research: {web_time:.2f}s")

            # === Étape 3 : RAG Search (avec cache) ===
            rag_result = ""
            rag_time = 0.0
            if self.tools.get("Blockchain_Knowledge"):
                cache_key = f"rag:{hash(input_text)}"
                rag_result = global_cache.get(cache_key)
                if rag_result is None:
                    start_rag = time.time()
                    rag_result = self._search_blockchain_knowledge_cached(input_text)
                    global_cache.set(cache_key, rag_result)
                    rag_time = time.time() - start_rag
                    logger.info(f"⏱️ RAG Search: {rag_time:.2f}s")

            # === Étape 4 : LLM Generation (streaming) ===
            start_llm = time.time()
            prompt = self._build_prompt(input_text, criteria, web_result, rag_result, tool_descriptions, query_type)
            return {"output": self._stream_llm_response(prompt)}

        except Exception as e:
            logging.error(f"Agent error: {e}")
            return {"output": [f"Error: {str(e)}"]}

    def _build_prompt(self, input_text, criteria, web_result, rag_result, tool_descriptions, query_type):
        if query_type == "recommendation":
            return f"""You are a senior blockchain architect. Your task is to recommend the most suitable blockchain for the user's use case, following the project's specifications.
User request: {input_text}
MANDATORY ANALYSIS (follow these steps):
1. Use TOOL[Web_Research] to get up-to-date information (cost, TPS, finality, recent upgrades).
2. Use TOOL[Dynamic_Analysis] to extract key requirements (TPS, cost, latency, domain, decentralization level).
3. Use TOOL[Blockchain_Knowledge] for context if relevant.
OUTPUT FORMAT (STRICT - MUST FOLLOW):
## 1. PROJECT OBJECTIVE
[Clear summary of the use case]
## 2. NEEDS ANALYSIS
**Detected domain**: [domain]
**Technical requirements**:
- TPS required: [number]
- Max cost per tx: [amount]
- Latency and finality: [time]
- Security level: [high/medium/low]
- Interoperability needs: [list]
## 3. BLOCKCHAIN MATCHING
**Recommended type**: [public/private/hybrid/consortium]
**Consensus protocol**: [PoW/PoS/PBFT/DPOS/etc.]
**Decentralization level**: [high/medium/low]
## 4. ARGUED RECOMMENDATION
**Chosen blockchain**: [Exact name]
**Justification**:
- Scalability: [real TPS data from web search]
- Cost: [real cost per tx, from web]
- Latency & Finality: [real times, from web]
- Governance: [governance model]
**Pros**:
- [3-5 specific advantages based on use case]
**Cons**:
- [2-3 honest disadvantages]
## 5. TOOLS & ECOSYSTEM
**Compatible frameworks**: [list]
**Available oracles**: [services like Chainlink, Pyth]
**Interoperability solutions**: [bridges, IBC, CCIP, etc.]
**Development tools**: [SDKs, APIs, wallets]
## 6. UPDATE & SOURCES
**Latest info** (from web search):
{web_result if web_result else 'No web data'}
**Sources consulted**:
[List official URLs]
Answer ONLY in this format. No extra text. Answer in English only."""
        elif query_type == "comparison":
            return f"""You are a blockchain expert. Compare the requested blockchains objectively and naturally, like a senior architect explaining to a client.
Available tools:
{tool_descriptions}
Request: {input_text}
Steps:
1. Use TOOL[Web_Research] for updated info
2. Use TOOL[Dynamic_Analysis] to understand the context
3. Provide a balanced, honest comparison — not a forced table.
Guidelines:
- Be honest: no blockchain is perfect.
- Focus on: performance, cost, decentralization, ecosystem, use case fit.
- Use bullet points or short paragraphs.
- Never invent data.
- Cite sources at the end.
Answer in clear, professional English."""
        else:
            return f"""You are a blockchain educator. Explain the concept clearly and simply.
Available tools:
{tool_descriptions}
User request: {input_text}
Steps:
1. Use TOOL[Web_Research] for updated info
2. Explain in a structured, educational way: context, evolution, real examples.
3. Be concise but complete.
Answer in clear, natural English. Never use forced formats."""

    def _search_blockchain_knowledge_cached(self, query: str) -> str:
        results = self.agent._search_blockchain_knowledge(query, k=3)
        return results

    def _stream_llm_response(self, prompt: str):
        try:
            for chunk in self.llm.stream(prompt):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
        except Exception as e:
            yield f"\n❌ Erreur de génération : {e}"


# =====================================================================
# MAIN AGENT
# =====================================================================
class BlockchainAgent:
    def __init__(self, model_name: str = "phi3:mini"):
        logging.info("🚀 Initializing Blockchain Agent...")
        self.model_name = model_name
        self.discovery_engine = DynamicDiscoveryEngine()
        self.llm: Optional[ChatOllama] = None
        self.vectorstore: Optional[Any] = None
        self.retriever = None
        self.agent: Optional[SimpleAgent] = None
        self.blockchain_data: List[Dict] = []
        self.conversation_history: List[Dict] = []
        self.history_file = "conversation_history.json"
        self.max_history_entries = 100
        self.history_save_interval = 5
        self.history_count_since_save = 0
        self._load_conversation_history()
        self._load_data()

    def _load_conversation_history(self):
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, "r", encoding="utf-8") as f:
                    self.conversation_history = json.load(f)[-self.max_history_entries:]
                logging.info(f"Loaded {len(self.conversation_history)} history entries")
            else:
                self.conversation_history = []
        except Exception as e:
            logging.error(f"Failed to load history: {e}")
            self.conversation_history = []

    def _load_data(self):
        try:
            if os.path.exists("blockchains.json"):
                with open("blockchains.json", "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                seen = set()
                unique_data = []
                for bc in raw_data:  # ✅ CORRIGÉ ici : raw_data
                    name = bc.get("name", "Unknown")
                    if name not in seen:
                        unique_data.append(bc)
                        seen.add(name)
                self.blockchain_data = unique_data
                logging.info(f"✅ {len(self.blockchain_data)} unique blockchains loaded.")
            else:
                logging.warning("⚠️ blockchains.json not found.")
                self.blockchain_data = []
        except Exception as e:
            logging.error(f"❌ Error loading blockchains.json: {e}")
            self.blockchain_data = []

    def _initialize_components(self):
        """Lazy loading du LLM et du VectorStore"""
        if self.llm is None:
            self.llm = ChatOllama(
                model=self.model_name,
                temperature=0.3,
                top_p=0.9,
                keep_alive="5m",
                streaming=True  # 🔥 Streaming activé
            )
        if self.vectorstore is None and self.blockchain_data:
            self._setup_rag()
        if self.agent is None:
            self._setup_simple_agent()

    def _setup_rag(self):
        persist_dir = "./chroma_db"
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        if os.path.exists(persist_dir):
            self.vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding)
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            logging.info("✅ RAG: Loaded from persisted DB")
        else:
            if not self.blockchain_data:
                logging.warning("⚠️ No blockchain data to build RAG")
                return
            documents = self._create_documents()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

            def embed_and_store(doc):
                return text_splitter.split_documents([doc])

            all_splits = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = executor.map(embed_and_store, documents)
                for split in results:
                    all_splits.extend(split)

            self.vectorstore = Chroma.from_documents(
                all_splits,
                embedding,
                persist_directory=persist_dir
            )
            self.vectorstore.persist()
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            logging.info("✅ RAG: Created, embedded in parallel, and persisted")

    def _create_documents(self) -> List[Document]:
        docs = []
        for bc in self.blockchain_data:  # ✅ CORRIGÉ ici : blockchain_data
            content = f"Name: {bc['name']}\n"  # ✅ \n au lieu de 
            for key in ["type", "consensus", "TPS", "tx_cost", "use_cases", "pros", "cons"]:
                if bc.get(key):
                    value = ", ".join(bc[key]) if isinstance(bc[key], list) else str(bc[key])
                    content += f"{key.capitalize()}: {value}\n"
            docs.append(Document(page_content=content, metadata={"name": bc["name"]}))
        return docs

    def _setup_simple_agent(self):
        tools = [
            Tool(name="Web_Research", func=self._search_web, description="Recherche web pour infos récentes"),
            Tool(name="Dynamic_Analysis", func=self._analyze_query_dynamically, description="Analyse dynamique des besoins"),
        ]
        if self.vectorstore:
            tools.append(Tool(
                name="Blockchain_Knowledge",
                func=self._search_blockchain_knowledge,
                description="Base locale (RAG)"
            ))
        self.agent = SimpleAgent(self.llm, tools, agent_ref=self)
        logging.info(f"✅ SimpleAgent configured with {len(tools)} tools")

    def _analyze_query_dynamically(self, query: str) -> str:
        criteria, req = self.discovery_engine.discover_requirements_dynamically(query)
        result = [f"🔍 Domain: {criteria.domain_context}", f"Confidence: {criteria.confidence_score:.2f}"]
        if req:
            result.extend([
                f"Use case: {req.use_case.value}",
                f"TPS: {req.tps_required}",
                f"Cost: ${req.max_cost_per_tx}",
                f"Latency: {req.max_latency_ms}ms"
            ])
        return "\n".join(result)

    def _search_blockchain_knowledge(self, query: str, k: int = 3) -> str:
        if not self.vectorstore:
            return "❌ Base locale non disponible"
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            parts = []
            for doc in docs:
                score = doc.metadata.get("score", "N/A")
                name = doc.metadata.get("name", "Inconnu")
                preview = doc.page_content[:300].replace("\n", " ").strip()
                parts.append(f"📌 {name} (score: {score}): {preview}...")
            return "\n".join(parts)
        except Exception as e:
            return f"Erreur RAG: {str(e)}"

    def _search_web(self, query: str) -> str:
        if DDGS is None:
            return "❌ Recherche web non disponible"
        cache_key = f"web:{hash(query)}"
        result = global_cache.get(cache_key)
        if result:
            return result

        def fetch():
            try:
                enhanced_query = f"{query} blockchain 2025 site:ethereum.org OR solana.com OR polygon.technology"
                with DDGS() as ddgs:
                    return list(ddgs.text(enhanced_query, max_results=3))
            except Exception as e:
                return str(e)

        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(fetch)
                results = future.result(timeout=8)
            if isinstance(results, str):
                return f"❌ Web search failed: {results}"
            if not results:
                return "❌ Aucun résultat trouvé."
            result = "\n".join([
                f"📌 {r['title'][:100]}\n🔗 {r['href']}\n📝 {r['body'][:150]}..." 
                for r in results[:3]
            ])
            global_cache.set(cache_key, result)
            return result
        except concurrent.futures.TimeoutError:
            return "⏳ Recherche web expirée (8s)"
        except Exception as e:
            return f"❌ Erreur web: {str(e)}"

    def analyser_question_utilisateur(self, question: str) -> dict:
        question_nettoyee = re.sub(r'[^\w\s\.\?\!\,]', ' ', question.lower().strip())
        keywords = []
        patterns = {
            "actions": ["recommandez", "quelle solution", "proposez", "choisir", "optimal", "best", "recommend"],
            "concepts": r"\b(nft|blockchain|smart contract|ipfs|arweave|royalties|gas fees|token|wallet|oracle|traçabilité|provenance|crypto|fiat|rewards)\b",
            "contraintes": r"\b(gas|coût|tps|latence|frais|multi-devise|fiat|crypto|durable)\b"
        }
        for key, pattern in patterns.items():
            if isinstance(pattern, str):
                found = re.findall(pattern, question_nettoyee)
                keywords.extend(found)
            else:
                keywords.extend([w for w in patterns["actions"] if w in question_nettoyee])
        type_question = "general"
        if any(w in question_nettoyee for w in ["recommande", "quelle solution", "choisir", "optimal"]):
            type_question = "recommendation"
        elif any(w in question_nettoyee for w in ["explique", "c'est quoi", "définis"]):
            type_question = "explanation"
        elif "compare" in question_nettoyee or "vs" in question_nettoyee:
            type_question = "comparison"
        prompt_web = f"Rechercher des solutions blockchain pour : {question.strip()} Focus sur : {', '.join(set(keywords)) if keywords else 'blockchain'}"
        return {
            "type": type_question,
            "mots_cles": list(set(keywords)),
            "prompt_optimise": prompt_web.strip()
        }

    def get_response(self, query: str) -> str:
        try:
            self._initialize_components()
            start = time.time()
            analysis = self.analyser_question_utilisateur(query)
            criteria, project_req = self.discovery_engine.discover_requirements_dynamically(query)
            print(f"\n🔍 Type: {analysis['type'].upper()}")
            if analysis['mots_cles']:
                print(f"🔑 Mots-clés: {', '.join(analysis['mots_cles'])}")
            print("\n🌐 Recherche web en cours...")
            print("💡 Base locale chargée.") if self.vectorstore else print("⚠️ Base locale non disponible.")
            context = f"Context: {project_req}" if project_req else ""
            agent_input = {"input": f"{query}\n{context}", "chat_history": self._get_recent_history_text()}
            result = self.agent.invoke(agent_input, analysis_result=(criteria, project_req))
            response_gen = result.get("output")
            print("\n📝 Réponse en temps réel :\n")
            full_response = ""
            for chunk in response_gen:
                print(chunk, end="", flush=True)
                full_response += chunk
            print()
            self._add_to_history(query, full_response, {"mode": "streamed"})
            total_time = time.time() - start
            logger.info(f"⏱️ Full get_response took {total_time:.2f}s")
            return full_response
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return f"❌ Error: {str(e)}"

    def get_response_stream(self, query: str):
        """
        Générateur pour le streaming progressif de la réponse.
        Utile pour Streamlit ou les interfaces web.
        """
        try:
            self._initialize_components()
            analysis = self.analyser_question_utilisateur(query)
            criteria, project_req = self.discovery_engine.discover_requirements_dynamically(query)
            context = f"Context: {project_req}" if project_req else ""
            agent_input = {"input": f"{query}\n{context}", "chat_history": self._get_recent_history_text()}
            result = self.agent.invoke(agent_input, analysis_result=(criteria, project_req))
            response_gen = result.get("output")

            full_response = ""
            for chunk in response_gen:
                if chunk:
                    full_response += chunk
                    yield chunk
            self._add_to_history(query, full_response, {"mode": "streamed"})

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            yield error_msg
            logging.error(f"Stream error: {e}")

    def _get_recent_history_text(self) -> str:
        return "\n".join([f"Q: {e['query'][:80]}...\nR: {e['response'][:120]}..." for e in self.conversation_history[-2:]])

    def _add_to_history(self, query: str, response: str, sources: Dict):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response[:1000],
            "sources": sources
        }
        self.conversation_history.append(entry)
        if len(self.conversation_history) > self.max_history_entries:
            self.conversation_history = self.conversation_history[-self.max_history_entries:]
        self.history_count_since_save += 1
        if self.history_count_since_save >= self.history_save_interval:
            self._save_history()
            self.history_count_since_save = 0

    def _save_history(self):
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"History save error: {e}")

    def _show_recent_history(self):
        if not self.conversation_history:
            print("No history available.")
            return
        print("\n📜 Recent History:")
        for entry in self.conversation_history[-5:]:
            print(f"Q: {entry['query'][:60]}...")
            print(f"R: {entry['response'][:100]}...\n")

    def start_interactive_chat(self):
        print("\n🚀 Blockchain Intelligence Agent")
        print("=" * 55)
        print("🎯 FEATURES:")
        print("  ✅ Blockchain recommendations")
        print("  ✅ Technical comparisons")
        print("  ✅ Explanations")
        print("  ✅ Up-to-date via web search")
        print("  ✅ Streaming LLM (real-time response)")
        print("  ✅ Smart query analysis")
        print("=" * 55)
        print("COMMANDS:")
        print("  /quit — Exit")
        print("  /help — Show help")
        print("  /history — Show history")
        print("=" * 55)
        while True:
            try:
                user_input = input("\n🤔 Your question: ").strip()
                if not user_input: continue
                if user_input.lower() in ['/quit', 'exit']: break
                if user_input.lower() == '/help': 
                    print("\n💡 TIPS: Be specific (TPS, cost, use case)")
                    continue
                if user_input.lower() == '/history': 
                    self._show_recent_history()
                    continue
                response = self.get_response(user_input)
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


# =====================================================================
# MAIN
# =====================================================================
def main():
    try:
        print("🚀 Starting Blockchain Agent...")
        agent = BlockchainAgent(model_name="phi3:mini")
        agent.start_interactive_chat()
    except Exception as e:
        logging.error(f"Critical error: {e}")
        print(f"❌ Startup failed: {e}")


if __name__ == "__main__":
    main()