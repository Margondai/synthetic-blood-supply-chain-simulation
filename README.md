# synthetic-blood-supply-chain-simulation
Simulation framework for synthetic blood deployment in emergency settings - MODSIM World 2025
# Synthetic Blood Supply Chain Simulation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MODSIM World 2025](https://img.shields.io/badge/Conference-MODSIM%20World%202025-blue)](https://www.modsimworld.org/)

Overview

This repository contains a comprehensive discrete-event simulation framework for optimizing the production and deployment of synthetic blood in emergency and resource-limited clinical settings. The simulation models the entire supply chain, from production to patient delivery, incorporating realistic constraints, degradation kinetics, and multimodal delivery systems.

Paper: "Simulation-Based Optimization of Synthetic Blood Production and Deployment in Emergency and Resource-Limited Clinical Settings"  
Conference: MODSIM World 2025 (Accepted)  
Lead Researcher: Soraya Hani  
Code Developer: Ancuta Margondai  
Institution: University of Central Florida  

# What is Synthetic Blood?

Synthetic blood products offer transformative potential for emergency medicine:
- HBOC (Hemoglobin-Based Oxygen Carriers): Room temperature stable, no cross-matching required
- PFC (Perfluorocarbon Emulsions): High oxygen solubility, extended shelf life
- Critical for: Rural areas, conflict zones, disaster response, military medicine

## Research Team

- Soraya Hani (Lead Researcher) - SsorayaHani@gmail.com
- Ancuta Margondai (Simulation Developer) - Ancuta.Margondai@ucf.edu
- Cindy Von Ahlefeldt (Co-Researcher)
- Valentina Ezcurra (Co-Researcher)  
- Anamaria Acevedo Diaz (Co-Researcher)
- Sara Willox (Co-Researcher)
- Dr. Mustapha Mouloua (Principal Investigator) - Mustapha.Mouloua@ucf.edu

## Key Features

### Advanced Simulation Capabilities
- Multi-phase experimental design with validation, scaling, and policy testing
- 32,400+ Monte Carlo runs for comprehensive parameter optimization
- Literature-grounded parameters from peer-reviewed medical research
- **Real-time degradation modeling** for both HBOC and PFC products

### Realistic System Modeling
- Markov chain system states (Operational, Delayed, Emergency, Failure)
- Multi-modal delivery optimization (drone vs. land transport)
- Geographic delay factors (urban, rural, remote, conflict zones)
- Clinical risk integration (MI and CARPA adverse events)

### Crisis Response Features
- Disaster mode simulation for surge demand scenarios
- Multi-facility resource sharing with bottleneck analysis
- Policy intervention testing for conflict zone deployment
- Real-time decision support for emergency planners

## Key Results

- Stockout Prevention: <5% stockout rates across all tested scenarios
- Delivery Optimization: 1.75-6.46 hours delivery times depending on configuration
- Optimal Configuration: HBOC threshold=8 units, PFC threshold=16 liters, 75% drone allocation
- Policy Impact: 90% drone allocation reduced delivery delays by 26% in conflict zones
- Computational Efficiency: 1.48-1.79 seconds per simulation iteration

## Installation & Usage

### Prerequisites
```bash
pip install simpy numpy matplotlib pandas
```

### Quick Start
```bash
# Clone the repository
git clone https://github.com/Margondai/synthetic-blood-supply-chain-simulation.git
cd synthetic-blood-supply-chain-simulation

# Install dependencies
pip install -r requirements.txt

# Run the simulation
python Soraya.py
```

### System Requirements
- Python: 3.8 or higher
- Hardware: Tested on Apple M1 MacBook (any modern computer sufficient)
- Memory: 8GB+ recommended for large Monte Carlo runs
- Storage: ~100MB for output files

## Simulation Framework

The simulation encompasses four progressive experimental phases:

### Phase 1: Model Development (32,400 runs)
- Parameter optimization through extensive Monte Carlo analysis
- Design space exploration across multiple variables
- Baseline performance establishment

### Phase 2: Validation (200 runs)
- Robustness testing across varying demand scenarios
- Sensitivity analysis for conflict conditions
- Bottleneck identification and analysis

### Phase 3: Multi-Facility Scaling (100 runs)
- Urban hospital + conflict clinic simulation
- Shared resource allocation modeling
- Disaster mode surge testing (12 patients/day)

### Phase 4: Policy Implementation (100 runs)
- Real-world case study (500-bed conflict hospital)
- Policy intervention testing
- Cost-effectiveness analysis

## Key Parameters

| Category | Parameter | Value | Source |
|----------|-----------|-------|---------|
| **Production** | HBOC Production Time | 5-7 days | Jahr, 2022 |
| | PFC Production Time | 3-5 days | Kim et al., 2024 |
| | Production Failure Rate | 7.5% | Khan et al., 2020 |
| **Delivery** | Drone Delivery Time | 0.5 hours | Roberts et al., 2018 |
| | Land Transport Time | 2-6 hours | Roberts et al., 2018 |
| | Drone Failure Rate | 1.8% | Glick et al., 2020 |
| **Clinical** | Trauma Demand | 4-10 units | Holcomb et al., 2005 |
| | HBOC MI Risk | 2.01% | Estep, 2025 |
| | PFC CARPA Risk | 1% | Kim et al., 2024 |
| **Economic** | HBOC Unit Cost | $10,000 | Estrada et al., 2025 |
| | PFC Cost | $2,000/L | Vichare & Janjic, 2025 |
| | Drone Cost | $75,000 | Glick et al., 2020 |

## Output Files

The simulation generates comprehensive analysis outputs:
- Performance Metrics: CSV files with stockout rates, delivery delays, costs
- Delay Analysis: Box plots by system state and geographic location
- Bottleneck Identification: Detailed logs of system constraints
- Policy Evaluation: Comparative analysis of intervention strategies

## Applications

### Emergency Medicine
- Rural hospital supply chain optimization
- Disaster response surge capacity planning
- Emergency department inventory management

### Military Medicine
- Combat zone synthetic blood deployment
- Field hospital logistics optimization
- Casualty evacuation supply coordination

### Humanitarian Aid
- Conflict zone medical support
- Refugee camp healthcare logistics
- Natural disaster emergency response

## Citation

If you use this simulation in your research, please cite our work:

```bibtex
@inproceedings{hani2025synthetic,
  title={Simulation-Based Optimization of Synthetic Blood Production and Deployment in Emergency and Resource-Limited Clinical Settings},
  author={Hani, Soraya and Margondai, Ancuta and Von Ahlefeldt, Cindy and Ezcurra, Valentina and Acevedo Diaz, Anamaria and Mouloua, Mustapha},
  booktitle={MODSIM World 2025},
  year={2025},
  organization={University of Central Florida}
}
```

For the code specifically:
```bibtex
@software{margondai2025simulation,
  title={Synthetic Blood Supply Chain Simulation},
  author={Margondai, Ancuta and Hani, Soraya},
  year={2025},
  url={https://github.com/Margondai/synthetic-blood-supply-chain-simulation},
  institution={University of Central Florida}
}
```

## Contributing

We welcome contributions from the research community! Please feel free to:
- Report bugs or suggest improvements
- Extend the simulation for new scenarios
- Adapt the framework for other medical supplies
- Collaborate on future research

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank the University of Central Florida and the MODSIM World 2025 conference for their support of this research. Special recognition to the medical logistics and emergency medicine research communities whose work informed our parameter selection.

## Contact

For questions about this research or potential collaborations:
- Technical Questions: Ancuta.Margondai@ucf.edu
- Research Inquiries: SsorayaHani@gmail.com
- Principal Investigator: Mustapha.Mouloua@ucf.edu

---

"Advancing synthetic blood deployment through simulation-based optimization for emergency and resource-limited clinical settings."
