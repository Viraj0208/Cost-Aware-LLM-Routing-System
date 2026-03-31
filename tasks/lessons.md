# Lessons Learned

## Routing Accuracy
- Rule-based routing achieves ~60-70% accuracy on mixed prompts
- The ML classifier (DistilBERT) is needed to push accuracy above 85%
- Key signals: code patterns, math patterns, prompt length, keyword matching
- "implement" keyword is a strong indicator of code complexity

## Cost Savings
- With balanced simple/complex workload: ~18% savings with rule-based routing
- Savings increase dramatically with real-world workloads (more simple queries)
- FrugalGPT reports 98% savings because production traffic is heavily skewed toward simple queries

## Testing Strategy
- Mock backends with quality-differentiated responses make testing realistic
- Async tests need `asyncio.get_event_loop().run_until_complete()` or `pytest-asyncio`
- FastAPI TestClient handles async transparently
