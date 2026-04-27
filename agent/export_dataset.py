"""Export Perpetual Context Data for Training Dataset Generation.

Converts conversation history into structured training formats:
- DPO (Direct Preference Optimization) pairs
- Supervised Fine-Tuning (SFT) datasets  
- Conversation summaries with topic tags
- User preference extraction

Usage:
    python export_dataset.py --db-path ~/.hermes/perpetual_context.db \
                             --output-dir ./training_data \
                             --format dpo,sft,summary
"""

import json
import logging
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DatasetExporter:
    """Export perpetual context data for model training."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        
    def export_dpo_pairs(
        self, 
        session_id: Optional[str] = None,
        min_turns: int = 2,
        max_samples: int = 10000
    ) -> List[Dict[str, Any]]:
        """Export DPO (Direct Preference Optimization) training pairs.
        
        Creates preference pairs where the assistant's response is the 
        'chosen' example and a modified/alternative version could be 
        the 'rejected' example.
        
        Args:
            session_id: Optional filter to specific conversation
            min_turns: Minimum number of turns in a conversation segment
            max_samples: Maximum number of pairs to export
            
        Returns:
            List of DPO training samples with chosen/rejected responses
        """
        logger.info(f"Exporting DPO pairs (max {max_samples} samples)")
        
        # Get conversation segments
        cursor = self.conn.execute("""
            SELECT session_id, role, content, timestamp, topic_tags
            FROM messages
            WHERE session_id IS NOT NULL
            ORDER BY session_id, timestamp ASC
        """)
        
        rows = cursor.fetchall()
        
        # Group by session and create segments
        sessions: Dict[str, List[Dict]] = {}
        for row in rows:
            sid = row['session_id']
            if sid not in sessions:
                sessions[sid] = []
            sessions[sid].append(dict(row))
        
        # Create DPO pairs from conversation segments
        dpo_pairs = []
        for session_id, messages in sessions.items():
            # Find assistant responses (potential 'chosen' examples)
            assistant_msgs = [m for m in messages if m['role'] == 'assistant']
            
            for i, chosen_msg in enumerate(assistant_msgs):
                if len(dpo_pairs) >= max_samples:
                    break
                
                # Get user query that prompted this response
                user_query = ""
                if i > 0 and assistant_msgs[i-1]['role'] == 'user':
                    user_query = assistant_msgs[i-1]['content']
                
                # Create a simple 'rejected' variant (shorter/less detailed)
                rejected_response = self._create_rejected_variant(chosen_msg['content'])
                
                pair = {
                    "prompt": user_query,
                    "chosen": chosen_msg['content'],
                    "rejected": rejected_response,
                    "metadata": {
                        "session_id": session_id,
                        "timestamp": chosen_msg['timestamp'],
                        "topics": json.loads(chosen_msg.get('topic_tags', '[]')) if chosen_msg.get('topic_tags') else [],
                        "turn_index": i
                    }
                }
                
                dpo_pairs.append(pair)
        
        logger.info(f"Exported {len(dpo_pairs)} DPO pairs")
        return dpo_pairs
    
    def _create_rejected_variant(self, response: str) -> str:
        """Create a simpler/less detailed variant of a response for 'rejected' example.
        
        In practice, you'd want to use actual alternative responses from:
        - Different model generations
        - Human annotations with lower quality
        - Modified versions (shorter, less accurate, etc.)
        
        For now, we create a simple truncated version as a placeholder.
        """
        # Simple truncation for demonstration
        sentences = response.split('. ')
        if len(sentences) > 2:
            return '. '.join(sentences[:2]) + '.'
        return response[:100] + "..."
    
    def export_sft_dataset(
        self,
        session_id: Optional[str] = None,
        max_samples: int = 10000
    ) -> List[Dict[str, Any]]:
        """Export Supervised Fine-Tuning (SFT) dataset.
        
        Creates instruction-response pairs for supervised training.
        
        Args:
            session_id: Optional filter to specific conversation
            max_samples: Maximum number of samples to export
            
        Returns:
            List of SFT training samples with messages array
        """
        logger.info(f"Exporting SFT dataset (max {max_samples} samples)")
        
        # Get all messages grouped by session
        cursor = self.conn.execute("""
            SELECT session_id, role, content, timestamp, topic_tags
            FROM messages
            WHERE session_id IS NOT NULL
            ORDER BY session_id, timestamp ASC
        """)
        
        rows = cursor.fetchall()
        
        # Group by session
        sessions: Dict[str, List[Dict]] = {}
        for row in rows:
            sid = row['session_id']
            if sid not in sessions:
                sessions[sid] = []
            sessions[sid].append(dict(row))
        
        # Create SFT samples (conversation segments)
        sft_samples = []
        for session_id, messages in sessions.items():
            if len(sft_samples) >= max_samples:
                break
            
            # Group into conversation turns (user + assistant pairs)
            turns = []
            current_turn = None
            
            for msg in messages:
                if msg['role'] == 'user':
                    if current_turn:
                        turns.append(current_turn)
                    current_turn = {'role': 'user', 'content': msg['content']}
                elif msg['role'] == 'assistant' and current_turn:
                    current_turn['content'] += '\n\n' + msg['content'] if current_turn.get('assistant_content') else msg['content']
                    current_turn['assistant_content'] = msg['content']
            
            if current_turn:
                turns.append(current_turn)
            
            # Create SFT samples from conversation segments
            for i in range(0, len(turns), 2):  # Take pairs of turns
                if len(sft_samples) >= max_samples:
                    break
                
                # Get context (previous turns) and current turn
                context_turns = turns[max(0, i-2):i]
                current_turn = turns[i] if i < len(turns) else None
                
                if not current_turn or current_turn['role'] != 'user':
                    continue
                
                # Build messages array for SFT format
                messages_array = []
                
                # Add system prompt if available
                messages_array.append({
                    "role": "system",
                    "content": "You are a helpful assistant with perfect memory."
                })
                
                # Add conversation history (context)
                for turn in context_turns:
                    messages_array.append(turn)
                
                # Add current user query and expected response
                messages_array.append({
                    "role": "user",
                    "content": current_turn['content']
                })
                
                # Get assistant response if available
                if i + 1 < len(turns):
                    next_turn = turns[i + 1]
                    if next_turn.get('assistant_content'):
                        messages_array.append({
                            "role": "assistant",
                            "content": next_turn['assistant_content']
                        })
                
                sample = {
                    "messages": messages_array,
                    "metadata": {
                        "session_id": session_id,
                        "turn_index": i,
                        "topics": [t.get('topic_tags', '') for t in context_turns if t.get('topic_tags')]
                    }
                }
                
                sft_samples.append(sample)
        
        logger.info(f"Exported {len(sft_samples)} SFT samples")
        return sft_samples
    
    def export_conversation_summaries(
        self,
        session_id: Optional[str] = None,
        max_sessions: int = 100
    ) -> List[Dict[str, Any]]:
        """Export conversation summaries with topic analysis.
        
        Creates high-level summaries of conversations for:
        - User preference learning
        - Topic modeling
        - Conversation quality assessment
        
        Args:
            session_id: Optional filter to specific conversation
            max_sessions: Maximum number of sessions to summarize
            
        Returns:
            List of conversation summaries with metadata
        """
        logger.info(f"Exporting conversation summaries (max {max_sessions} sessions)")
        
        # Get unique sessions
        cursor = self.conn.execute("""
            SELECT DISTINCT session_id, COUNT(*) as message_count, 
                   MIN(timestamp) as first_message,
                   MAX(timestamp) as last_message
            FROM messages
            WHERE session_id IS NOT NULL
            GROUP BY session_id
            ORDER BY last_message DESC
            LIMIT ?
        """, (max_sessions,))
        
        sessions = cursor.fetchall()
        
        summaries = []
        for session in sessions:
            sid = session['session_id']
            
            # Get all messages for this session
            cursor = self.conn.execute("""
                SELECT role, content, timestamp, topic_tags
                FROM messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """, (sid,))
            
            messages = cursor.fetchall()
            
            # Extract topics across the conversation
            all_topics = set()
            for msg in messages:
                if msg['topic_tags']:
                    try:
                        topics = json.loads(msg['topic_tags'])
                        all_topics.update(topics)
                    except:
                        pass
            
            # Create summary
            summary = {
                "session_id": sid,
                "message_count": session['message_count'],
                "duration_seconds": session['last_message'] - session['first_message'],
                "topics": list(all_topics),
                "topic_density": len(all_topics) / max(session['message_count'], 1),
                "roles_distribution": {
                    role: sum(1 for m in messages if m['role'] == role)
                    for role in ['user', 'assistant', 'system']
                },
                "first_message": messages[0]['content'][:200] if messages else "",
                "last_message": messages[-1]['content'][:200] if messages else ""
            }
            
            summaries.append(summary)
        
        logger.info(f"Exported {len(summaries)} conversation summaries")
        return summaries
    
    def export_user_preferences(
        self,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract user preferences from conversation history.
        
        Analyzes conversations to identify:
        - Repeated topics of interest
        - Communication style preferences
        - Knowledge level indicators
        - Explicit preference statements
        
        Args:
            session_id: Optional filter to specific conversation
            
        Returns:
            Dict with extracted user preferences and insights
        """
        logger.info("Exporting user preferences")
        
        # Get all user messages
        cursor = self.conn.execute("""
            SELECT role, content, timestamp, topic_tags, session_id
            FROM messages
            WHERE role = 'user'
            ORDER BY timestamp ASC
        """)
        
        user_messages = cursor.fetchall()
        
        # Analyze topics and frequency
        topic_counts: Dict[str, int] = {}
        for msg in user_messages:
            if msg['topic_tags']:
                try:
                    topics = json.loads(msg['topic_tags'])
                    for topic in topics:
                        topic_counts[topic] = topic_counts.get(topic, 0) + 1
                except:
                    pass
        
        # Identify top interests
        top_interests = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Analyze communication style (simple heuristics)
        avg_message_length = sum(len(msg['content']) for msg in user_messages) / max(len(user_messages), 1)
        question_count = sum(1 for msg in user_messages if '?' in msg['content'])
        
        preferences = {
            "top_interests": [{"topic": topic, "frequency": count} for topic, count in top_interests],
            "communication_style": {
                "avg_message_length": avg_message_length,
                "question_ratio": question_count / max(len(user_messages), 1),
                "detail_oriented": avg_message_length > 100
            },
            "total_conversations": len(set(msg['session_id'] for msg in user_messages)),
            "active_topics": list(topic_counts.keys())[:50]
        }
        
        logger.info(f"Extracted preferences: {len(preferences['top_interests'])} interests, "
                   f"{preferences['communication_style']['question_ratio']:.2f} question ratio")
        
        return preferences
    
    def export_all(
        self,
        output_dir: str = "./training_data",
        formats: List[str] = None
    ) -> Dict[str, int]:
        """Export all data formats to specified directory.
        
        Args:
            output_dir: Directory to save exported files
            formats: List of formats to export ('dpo', 'sft', 'summaries', 'preferences')
            
        Returns:
            Dict with counts of exported items per format
        """
        if formats is None:
            formats = ['dpo', 'sft', 'summaries', 'preferences']
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Export DPO pairs
        if 'dpo' in formats:
            dpo_pairs = self.export_dpo_pairs()
            dpo_file = output_path / "dpo_pairs.jsonl"
            with open(dpo_file, 'w') as f:
                for pair in dpo_pairs:
                    f.write(json.dumps(pair) + '\n')
            results['dpo'] = len(dpo_pairs)
        
        # Export SFT dataset
        if 'sft' in formats:
            sft_samples = self.export_sft_dataset()
            sft_file = output_path / "sft_dataset.jsonl"
            with open(sft_file, 'w') as f:
                for sample in sft_samples:
                    f.write(json.dumps(sample) + '\n')
            results['sft'] = len(sft_samples)
        
        # Export conversation summaries
        if 'summaries' in formats:
            summaries = self.export_conversation_summaries()
            summaries_file = output_path / "conversation_summaries.json"
            with open(summaries_file, 'w') as f:
                json.dump(summaries, f, indent=2)
            results['summaries'] = len(summaries)
        
        # Export user preferences
        if 'preferences' in formats:
            preferences = self.export_user_preferences()
            prefs_file = output_path / "user_preferences.json"
            with open(prefs_file, 'w') as f:
                json.dump(preferences, f, indent=2)
            results['preferences'] = 1
        
        logger.info(f"Export completed. Results: {results}")
        return results


def main():
    """CLI entry point for dataset export."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Export Perpetual Context data for training')
    parser.add_argument('--db-path', default='~/.hermes/perpetual_context.db',
                       help='Path to perpetual context database')
    parser.add_argument('--output-dir', default='./training_data',
                       help='Directory to save exported files')
    parser.add_argument('--format', nargs='+', 
                       choices=['dpo', 'sft', 'summaries', 'preferences'],
                       default=['dpo', 'sft', 'summaries', 'preferences'],
                       help='Export formats (default: all)')
    
    args = parser.parse_args()
    
    # Expand ~ in paths
    db_path = Path(args.db_path).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        sys.exit(1)
    
    # Export data
    exporter = DatasetExporter(str(db_path))
    results = exporter.export_all(str(output_dir), args.format)
    
    print("\nExport Results:")
    print("=" * 50)
    for format_name, count in results.items():
        print(f"  {format_name:15s}: {count:>6} items")
    print("=" * 50)
    print(f"\nFiles saved to: {output_dir}")


def export_dpo_data(db, session_id: str, output_path: str) -> Dict[str, Any]:
    """Export DPO pairs from a PerpetualContextDB instance.
    
    Wrapper around DatasetExporter for use by the perpetual context plugin.
    
    Args:
        db: PerpetualContextDB instance (has .conn and .db_path attributes)
        session_id: Session to export
        output_path: Output JSONL file path
        
    Returns:
        Dict with success status and metadata
    """
    try:
        exporter = DatasetExporter(str(db.db_path))
        pairs = exporter.export_dpo_pairs(session_id=session_id, max_samples=10000)
        
        with open(output_path, 'w') as f:
            for pair in pairs:
                f.write(json.dumps(pair) + '\n')
        
        return {"success": True, "pairs_exported": len(pairs), "output_file": output_path}
    except Exception as e:
        logger.error(f"DPO export failed: {e}")
        return {"error": str(e)}


def export_sft_data(db, session_id: str, output_path: str) -> Dict[str, Any]:
    """Export SFT dataset from a PerpetualContextDB instance.
    
    Wrapper around DatasetExporter for use by the perpetual context plugin.
    
    Args:
        db: PerpetualContextDB instance (has .conn and .db_path attributes)
        session_id: Session to export
        output_path: Output JSONL file path
        
    Returns:
        Dict with success status and metadata
    """
    try:
        exporter = DatasetExporter(str(db.db_path))
        samples = exporter.export_sft_dataset(session_id=session_id, max_samples=10000)
        
        with open(output_path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        
        return {"success": True, "samples_exported": len(samples), "output_file": output_path}
    except Exception as e:
        logger.error(f"SFT export failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    main()
