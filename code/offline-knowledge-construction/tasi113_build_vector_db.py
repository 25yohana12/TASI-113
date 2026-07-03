# ==============================================================================
# SINGLE-VECTOR KNOWLEDGE BASE with Quality Filter
# ==============================================================================
import os
import json
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
import chromadb
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich import box
console = Console()
# ==============================================================================
# CONFIG
# ==============================================================================
@dataclass
class KBConfig:
    """Configuration untuk single-vector KB dengan quality filter"""
    input_json: str = ""
    output_dir: str = ""
    model_path: str = ""
    collection_name: str = "text2sql_questions"
    distance_metric: str = "cosine"
    batch_size: int = 512
    normalize_embeddings: bool = True
    quality_filter: str = "valid_quality"  # ← Filter untuk quality status
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
# ==============================================================================
# SINGLE-VECTOR BUILDER with QUALITY FILTER
# ==============================================================================
class SingleVectorBuilder:
    """Build simple 1:1 question-sql index dengan quality filtering"""
    
    def __init__(self, config: KBConfig):
        self.config = config
        self.stats = {
            'total_entries': 0,
            'quality_filter_applied': False,
            'passed_quality_check': 0,
            'failed_quality_check': 0,
            'missing_quality_field': 0,
            'missing_question_sql': 0,
            'final_indexed': 0,
        }
    
    def build_documents(self, dataset: List[Dict]) -> Tuple[List[str], List[str], List[Dict]]:
        """
        Build documents dengan quality filtering.
        
        Hanya entries dengan status="valid_quality" yang akan diindex.
        
        Returns:
            ids: question IDs
            docs: pertanyaan (untuk embed)
            metadatas: question + sql untuk retrieval
        """
        ids = []
        docs = []
        metadatas = []
        
        console.print(f"\n[cyan]Building Single-Vector KB with Quality Filter...[/cyan]")
        console.print(f"[cyan]Filter applied: status = '{self.config.quality_filter}'[/cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Processing entries...", total=len(dataset))
            
            for idx, entry in enumerate(dataset):
                self.stats['total_entries'] += 1
                
                # ===== QUALITY CHECK =====
                status = entry.get("status", None)
                
                if status is None:
                    self.stats['missing_quality_field'] += 1
                    progress.update(task, advance=1)
                    continue
                
                if status != self.config.quality_filter:
                    self.stats['failed_quality_check'] += 1
                    progress.update(task, advance=1)
                    continue
                
                self.stats['quality_filter_applied'] = True
                
                # ===== CONTENT CHECK =====
                try:
                    question = entry.get("question", "").strip()
                    sql = entry.get("sql", "").strip()
                    db_id = entry.get("db_id", "unknown")
                    source_id = str(entry.get("question_id", f"auto_{idx}"))
                    
                    if not question or not sql:
                        self.stats['missing_question_sql'] += 1
                        progress.update(task, advance=1)
                        continue
                    
                    # ===== ADD TO INDEX =====
                    ids.append(source_id)
                    docs.append(question)
                    metadatas.append({
                        "question": question,
                        "sql": sql,
                        "db_id": db_id,
                        "source_id": source_id,
                        "status": status,  # ← Track status di metadata
                    })
                    
                    self.stats['passed_quality_check'] += 1
                    self.stats['final_indexed'] += 1
                
                except Exception as e:
                    console.print(f"[yellow]  Warning entry {idx}: {e}[/yellow]")
                
                progress.update(task, advance=1)
        
        return ids, docs, metadatas
# ==============================================================================
# KNOWLEDGE BASE BUILDER
# ==============================================================================
class KnowledgeBaseBuilder:
    """Main builder untuk single-vector KB dengan quality filter"""
    
    def __init__(self, config: KBConfig):
        self.config = config
        self.model = None
        self.collection = None
        self.builder_stats = {}
        self.build_stats = {
            'start_time': None,
            'end_time': None,
            'total_entries': 0,
            'total_vectors': 0,
        }
    
    def build(self):
        self.build_stats['start_time'] = datetime.now()
        
        self._print_header()
        self._load_dataset()
        self._load_model()
        self._init_chromadb()
        self._load_and_index_data()
        self._verify()
        self._sample_test()
        
        self.build_stats['end_time'] = datetime.now()
        
        self._save_metadata()
        self._print_summary()
    
    def _print_header(self):
        """Print header"""
        console.print()
        console.print(Panel.fit(
            "[bold blue]SINGLE-VECTOR KNOWLEDGE BASE BUILDER[/bold blue]\n"
            "[dim]Optimized untuk: Query → Retrieve Question-SQL Pairs[/dim]\n"
            f"[yellow]Quality Filter: status = 'valid_quality'[/yellow]\n"
            f"ChromaDB + Cosine Similarity | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            border_style="blue",
            box=box.DOUBLE
        ))
    
    def _load_dataset(self):
        """Load and validate dataset"""
        console.print(f"\n[cyan]Loading dataset:[/cyan] {self.config.input_json}")
        
        if not os.path.exists(self.config.input_json):
            raise FileNotFoundError(f"File not found: {self.config.input_json}")
        
        with open(self.config.input_json, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
        
        console.print(f"[green]✓[/green] Loaded {len(self.dataset)} total entries")
        
        # ===== PREVIEW QUALITY STATUS DISTRIBUTION =====
        self._preview_quality_distribution()
    
    def _preview_quality_distribution(self):
        """Preview data quality distribution sebelum indexing"""
        console.print(f"\n[cyan]Quality Status Distribution:[/cyan]")
        
        status_counts = {}
        for entry in self.dataset:
            status = entry.get("status", "missing")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        total = len(self.dataset)
        for status, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total * 100) if total > 0 else 0
            
            if status == self.config.quality_filter:
                console.print(f"  [green]✓ {status:20s}: {count:6d} ({pct:5.1f}%)[/green]")
            else:
                console.print(f"  [red]✗ {status:20s}: {count:6d} ({pct:5.1f}%)[/red]")
    
    def _load_model(self):
        """Load embedding model"""
        console.print(f"\n[cyan]Loading embedding model:[/cyan] {self.config.model_path}")
        
        start = time.time()
        self.model = SentenceTransformer(
            self.config.model_path,
            trust_remote_code=True
        )
        elapsed = time.time() - start
        
        dim = self.model.get_sentence_embedding_dimension()
        console.print(f"[green]✓[/green] Loaded in {elapsed:.2f}s (dim: {dim})")
    
    def _init_chromadb(self):
        """Initialize ChromaDB"""
        console.print(f"\n[cyan]Initializing ChromaDB:[/cyan] {self.config.output_dir}")
        
        client = chromadb.PersistentClient(path=self.config.output_dir)
        
        try:
            client.delete_collection(name=self.config.collection_name)
            console.print(f"[yellow]Deleted existing collection[/yellow]")
        except Exception:
            pass
        
        self.collection = client.create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": self.config.distance_metric}
        )
        
        console.print(f"[green]✓[/green] Created collection: {self.config.collection_name}")
    
    def _load_and_index_data(self):
        """Process and index documents"""
        console.print(
            f"\n[cyan]Processing and indexing documents "
            f"(batch size: {self.config.batch_size})...[/cyan]"
        )
        
        # Build documents dengan quality filter
        builder = SingleVectorBuilder(self.config)
        ids, docs, metadatas = builder.build_documents(self.dataset)
        
        self.builder_stats = builder.stats
        
        console.print(f"\n[green]✓[/green] Quality filter results:")
        console.print(f"  Total entries: {builder.stats['total_entries']}")
        console.print(f"  [green]Passed quality check: {builder.stats['passed_quality_check']}[/green]")
        console.print(f"  [red]Failed quality check: {builder.stats['failed_quality_check']}[/red]")
        console.print(f"  [yellow]Missing quality field: {builder.stats['missing_quality_field']}[/yellow]")
        console.print(f"  [yellow]Missing question/sql: {builder.stats['missing_question_sql']}[/yellow]")
        console.print(f"  [cyan]Final indexed: {len(docs)}[/cyan]")
        
        # Index in batches
        console.print(f"\n[cyan]Encoding and indexing {len(docs)} documents...[/cyan]")
        
        if len(docs) == 0:
            console.print(f"[red]ERROR: No valid documents to index![/red]")
            return
        
        total_indexed = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Indexing batches...", total=len(docs))
            
            for i in range(0, len(docs), self.config.batch_size):
                batch_ids = ids[i:i + self.config.batch_size]
                batch_docs = docs[i:i + self.config.batch_size]
                batch_metas = metadatas[i:i + self.config.batch_size]
                
                # Encode
                embeddings = self.model.encode(
                    batch_docs,
                    show_progress_bar=False,
                    normalize_embeddings=self.config.normalize_embeddings,
                    batch_size=64,
                ).tolist()
                
                # Add to collection
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    embeddings=embeddings,
                    metadatas=batch_metas,
                )
                
                total_indexed += len(batch_ids)
                progress.update(task, advance=len(batch_ids))
        
        console.print(f"\n[green]✓[/green] Indexed {total_indexed} documents")
        self.build_stats['total_vectors'] = total_indexed
    
    def _verify(self):
        """Verify collection"""
        console.print("\n[cyan]Verifying indexed data...[/cyan]")
        
        final_count = self.collection.count()
        expected = self.builder_stats['final_indexed']
        
        console.print(f"  Collection count: {final_count}")
        console.print(f"  Expected count: {expected}")
        
        if final_count == expected:
            console.print(f"[green]✓[/green] Verification passed")
        else:
            console.print(f"[red]✗[/red] Mismatch: expected {expected}, got {final_count}")
            raise ValueError(f"Count mismatch")
    
    def _sample_test(self):
        """Run sample retrieval tests"""
        console.print("\n[cyan]Running sample retrieval tests...[/cyan]")
        
        test_queries = [
            "Find singers with average age by country",
            "List top 10 albums by release year",
            "Count genres where popularity is above threshold",
        ]
        
        for query in test_queries:
            console.print(f"\n  [yellow]Query:[/yellow] {query}")
            
            query_embedding = self.model.encode(
                [query],
                normalize_embeddings=self.config.normalize_embeddings
            ).tolist()
            
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=3,
            )
            
            if results["metadatas"] and results["metadatas"][0]:
                console.print("  Top 3 matches:")
                for i, (meta, dist) in enumerate(zip(results["metadatas"][0], results["distances"][0]), 1):
                    similarity = 1.0 - dist
                    q = meta['question'][:60] + "..." if len(meta['question']) > 60 else meta['question']
                    s = meta['sql'][:50] + "..." if len(meta['sql']) > 50 else meta['sql']
                    status = meta.get('status', 'unknown')
                    console.print(
                        f"    [{i}] Sim: {similarity:.4f} | Status: {status}\n"
                        f"        Q: {q}\n"
                        f"        S: {s}"
                    )
            else:
                console.print("  [yellow]No results found[/yellow]")
    
    def _save_metadata(self):
        metadata_file = os.path.join(self.config.output_dir, "kb_metadata.json")
    
        db_ids = set()
        for entry in self.dataset:
            if (entry.get("status") == self.config.quality_filter and 
                entry.get("question") and entry.get("sql")):
                db_ids.add(entry.get("db_id", "unknown"))
        
        duration = (self.build_stats['end_time'] - self.build_stats['start_time']).total_seconds()
        
        metadata = {
            "build_time": self.build_stats['start_time'].isoformat(),
            "build_duration_seconds": duration,  # ← Sekarang tidak None
            "architecture": "Single-Vector (Question-only)",
            "quality_filter": {
                "field": "status",
                "value": self.config.quality_filter
            },
            "statistics": {
                "total_entries_in_dataset": self.builder_stats['total_entries'],
                "passed_quality_check": self.builder_stats['passed_quality_check'],
                "failed_quality_check": self.builder_stats['failed_quality_check'],
                "missing_quality_field": self.builder_stats['missing_quality_field'],
                "missing_question_sql": self.builder_stats['missing_question_sql'],
            },
            "total_indexed_vectors": self.build_stats['total_vectors'],
            "vectors_per_entry": 1,
            "embedding_model": self.config.model_path,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "distance_metric": self.config.distance_metric,
            "indexing_algorithm": "HNSW",
            "unique_databases_indexed": len(db_ids),
            "use_case": "User Query → Retrieve matching Question-SQL Pairs",
            "quality_assurance": "Only entries with status='valid_quality' are indexed"
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        console.print(f"\n[green]✓[/green] Metadata saved: {metadata_file}")
    
    def _print_summary(self):
        """Print summary"""
        # ===== NOW BOTH TIMESTAMPS ARE SET =====
        duration = (self.build_stats['end_time'] - self.build_stats['start_time']).total_seconds()
        
        table = Table(
            title="\n📊 Single-Vector Knowledge Base Summary (with Quality Filter)",
            show_header=True,
            box=box.ROUNDED
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Build Duration", f"{duration:.2f}s")
        table.add_row("Total Dataset Entries", str(self.builder_stats['total_entries']))
        table.add_row("Passed Quality Filter", str(self.builder_stats['passed_quality_check']))
        table.add_row("Failed Quality Filter", str(self.builder_stats['failed_quality_check']))
        table.add_row("Missing Quality Field", str(self.builder_stats['missing_quality_field']))
        table.add_row("Missing Question/SQL", str(self.builder_stats['missing_question_sql']))
        table.add_row("Final Indexed Vectors", str(self.build_stats['total_vectors']))
        table.add_row("Vectors per Entry", "1 (question only)")
        table.add_row("Quality Filter", f"status = '{self.config.quality_filter}'")
        table.add_row("Embedding Model", os.path.basename(self.config.model_path))
        table.add_row("Distance Metric", self.config.distance_metric)
        table.add_row("Indexing Algorithm", "HNSW")
        table.add_row("Output Path", self.config.output_dir)
        
        console.print(table)
        console.print()
        console.print(Panel.fit(
            "[bold green]✓ KNOWLEDGE BASE BUILT SUCCESSFULLY[/bold green]\n"
            f"[dim]Indexed {self.build_stats['total_vectors']} valid entries with quality filter in {duration:.2f}s[/dim]",
            border_style="green",
            box=box.DOUBLE
        ))
        console.print()
# ==============================================================================
# CLI
# ==============================================================================
def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build Single-Vector Knowledge Base dengan Quality Filter"
    )
    parser.add_argument("--input", "-i", default="../../../outputs/offline-knowledge-construction/after-semhas/data-synthetic/quality-iter-01/quality-iter-01.json", help="Input JSON")
    parser.add_argument("--output", "-o", default="../../tasi113-evaluation/pipeline/kb/tasi113", help="Output dir")
    parser.add_argument("--model", "-m", default="../../../models/Qwen3-Embedding-0.6B", help="Model path")
    parser.add_argument("--batch-size", "-b", type=int, default=512, help="Batch size")
    parser.add_argument(
        "--quality-filter", "-q", 
        default="valid_quality", 
        help="Quality status to filter (default: 'valid_quality')"
    )
    
    args = parser.parse_args()
    
    config = KBConfig(
        input_json=args.input,
        output_dir=args.output,
        model_path=args.model,
        batch_size=args.batch_size,
        quality_filter=args.quality_filter,
    )
    
    builder = KnowledgeBaseBuilder(config)
    builder.build()
if __name__ == "__main__":
    main()