# Hospital Bed Optimization Using a Heuristic Algorithm

## Author: Gustavo Ferreira Tavares  
## Advisor: Gabriela Nunes Lopes  
### Undergraduate Thesis Project in Systems Engineering (UFMG)

---

## About

Hospital bed allocation is a critical and complex challenge for healthcare systems worldwide. In countries like Brazil, where public hospitals face overcrowding and limited resources, improving bed management efficiency directly impacts patient safety, waiting times, and the overall quality of healthcare services.

This project models the Patient-Bed Allocation Problem (PBA) as a multi-objective integer programming problem and proposes a heuristic approach capable of providing near-optimal solutions in real time. Unlike exact methods, which are often computationally infeasible in large hospital settings, the proposed heuristic can generate efficient and practical solutions within seconds.

---

## How to run

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run
python src/main.py
```
