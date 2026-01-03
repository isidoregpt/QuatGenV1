# Quat Generator Pro: Instruction Manual for Research Chemists

**AI-Powered Quaternary Ammonium Compound Discovery**

*Version 1.0 | January 2026*

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [The Science Behind the Tool](#2-the-science-behind-the-tool)
3. [Installation Guide](#3-installation-guide)
4. [Using the Application](#4-using-the-application)
5. [Interpreting Results](#5-interpreting-results)
6. [Worked Examples](#6-worked-examples)
7. [Quick Reference](#7-quick-reference)
8. [Appendices](#8-appendices)

> *In a hurry? Skip to [Section 4.2: Generating New Molecules](#42-generating-new-molecules) to start immediately, or [Section 7: Quick Reference](#7-quick-reference) for a one-page summary.*

---

## 1. Executive Summary

### What is Quat Generator Pro?

Quat Generator Pro is an AI-powered molecular discovery platform specifically designed for quaternary ammonium compound (quat) research. It combines state-of-the-art generative chemistry models with multi-objective optimization to suggest novel antimicrobial candidates tailored to your research priorities.

Unlike traditional approaches that rely on combinatorial enumeration or manual scaffold hopping, Quat Generator Pro uses a deep learning model trained on millions of drug-like molecules to propose chemically valid, synthesizable structures that balance efficacy, safety, environmental impact, and synthetic accessibility.

### Why AI-Assisted Molecular Design Matters

Quaternary ammonium antimicrobials remain critical for infection control, yet the field faces mounting challenges: growing antimicrobial resistance, increasing environmental scrutiny, and stringent safety requirements. Traditional lead discovery methods—testing compound libraries or iterating on known scaffolds—are time-consuming and often fail to explore novel chemical space.

AI-assisted design addresses these limitations by:

- **Accelerating time to lead**: Generate hundreds of scored candidates in minutes rather than months
- **Exploring vast chemical space**: Access structural diversity beyond what intuition or enumeration can reach
- **Optimizing multiple objectives simultaneously**: Balance efficacy against safety and environmental impact from the start
- **Reducing synthesis waste**: Prioritize candidates with the highest probability of success before committing resources

### What Problems Does It Solve?

| Challenge | How Quat Generator Pro Helps |
|-----------|------------------------------|
| Limited structural diversity in existing quat libraries | Generates novel scaffolds beyond traditional benzalkonium/pyridinium frameworks |
| Trade-offs between potency and safety | Multi-objective scoring identifies compounds with optimal balance |
| Environmental regulatory pressure | Integrated biodegradability and aquatic toxicity predictions |
| Synthesis bottlenecks | SA Score prioritizes readily synthesizable candidates |
| Data overload from high-throughput screening | Intelligent ranking focuses attention on top prospects |

### Key Capabilities at a Glance

- **Generate** novel quaternary ammonium structures with customizable optimization targets
- **Score** candidates for antimicrobial efficacy (predicted MIC), human safety (ADMET), environmental impact, and synthetic accessibility
- **Search** generated libraries by substructure patterns or molecular similarity
- **Benchmark** candidates against eight established reference quats (BAC, CPC, DDAC, and more)
- **Export** results in CSV, SDF, or PDF formats for lab notebooks and collaborators

---

## 2. The Science Behind the Tool

### 2.1 How AI Generates Novel Quaternary Ammonium Structures

#### The REINVENT Generative Model

At the heart of Quat Generator Pro is REINVENT, a 171-million parameter deep learning model that has "learned the grammar of chemistry." Think of it as a language model, but instead of predicting the next word in a sentence, it predicts the next atom or bond in a molecular structure.

*Imagine teaching a chemist to draw molecules by showing them millions of examples. Eventually, they learn the "rules"—carbons make four bonds, aromatic rings are stable, certain groups appear together. REINVENT learned these rules from 750 million molecules, so when it draws new structures, they're chemically sensible.*

**Training Foundation**: REINVENT was trained on millions of molecules from public databases (ZINC, ChEMBL) before being fine-tuned specifically for quaternary ammonium generation. Through this training, the model learned:

- Valid chemical bonding rules (no five-valent carbons or impossible ring systems)
- Drug-like property distributions (molecular weight, lipophilicity, polar surface area)
- Common medicinal chemistry motifs and synthetic building blocks

**Why Generated Structures Are Chemically Valid**: Unlike random enumeration, which produces many invalid or nonsensical structures, REINVENT generates molecules token-by-token using learned probabilities. Each generation step considers which atoms and bonds are chemically permissible given what has already been constructed. Invalid SMILES strings are automatically rejected.

#### Reinforcement Learning for Objective Optimization

The true power of the system comes from reinforcement learning (RL) fine-tuning. During generation, the model receives "rewards" based on how well each generated molecule scores against your specified objectives.

- Molecules with high efficacy, safety, and SA scores receive positive reinforcement
- Molecules with poor scores or invalid structures receive negative reinforcement
- Over thousands of iterations, the model learns to preferentially generate structures that meet your criteria

This differs fundamentally from random screening: instead of generating molecules blindly and hoping some are good, the system learns what "good" means for your project and actively seeks it.

> **Why This Matters**: Traditional combinatorial approaches generate molecules randomly and filter afterward, wasting computational resources on poor candidates. RL-guided generation is like having a chemist who learns from every failed compound and adjusts their design strategy in real-time.

### 2.2 Scoring and Prediction Models

Every generated molecule is evaluated by four independent scoring systems, each producing a normalized score from 0-100 where higher is better.

#### Efficacy Scoring: Predicting Antimicrobial Activity

The efficacy score predicts how potently a compound will inhibit microbial growth, expressed as MIC (Minimum Inhibitory Concentration) values.

**Prediction Approach**:

1. **Molecular Encoding**: Each SMILES structure is converted to a 768-dimensional vector using ChemBERTa, a chemistry-specific language model from DeepChem
2. **MIC Prediction**: A neural network trained on known antimicrobial data predicts MIC values for multiple organisms:
   - *Staphylococcus aureus* (Gram-positive)
   - *Escherichia coli* (Gram-negative)
   - *Pseudomonas aeruginosa* (resistant Gram-negative)
   - *Candida albicans* (fungal)
3. **Score Conversion**: Predicted MIC values are converted to a 0-100 scale using a logarithmic transformation

**Structure-Activity Relationships Captured**:
- Optimal alkyl chain length (C12-C16 for single-chain quats)
- Headgroup effects (benzyl, pyridinium, imidazolium preferences)
- Lipophilicity balance (LogP typically 2-5 for membrane activity)
- Critical micelle concentration considerations

**Data Sources**: The MIC predictor was trained using experimental data from ChEMBL screening assays and the reference compound database.

#### Safety Scoring: ADMET Property Predictions

The safety score evaluates human toxicity risk across multiple endpoints using pre-trained ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) models.

| Property | What It Predicts | Clinical Relevance |
|----------|-----------------|-------------------|
| **hERG Inhibition** | Cardiac ion channel blockade | Risk of QT prolongation and arrhythmias |
| **AMES Mutagenicity** | Bacterial reverse mutation | Regulatory flag for genotoxicity |
| **DILI** | Drug-induced liver injury | Hepatotoxicity risk |
| **LD50** | Acute oral toxicity | Accidental exposure hazard |
| **BBB Penetration** | Blood-brain barrier crossing | CNS side effect potential |

For each property, lower toxicity probability translates to a higher safety score.

> **Clinical Context**: hERG inhibition is a critical concern for any compound that might be absorbed systemically. The AMES test is a regulatory requirement for consumer product ingredients. LD50 values inform occupational exposure limits and packaging requirements.

#### Environmental Scoring: Ecological Impact Assessment

Environmental scores predict a compound's fate and effects in aquatic and terrestrial ecosystems.

**Components Evaluated**:

| Parameter | What It Measures | Why It Matters |
|-----------|-----------------|-----------------|
| **Biodegradability** | Rate of microbial breakdown | Persistence in wastewater treatment |
| **Bioconcentration Factor (BCF)** | Accumulation in aquatic organisms | Food chain magnification risk |
| **Aquatic Toxicity** | Fish LC50 estimates | Environmental hazard classification |
| **Water Solubility** | Environmental mobility | Spread potential in ecosystems |

These predictions help identify candidates compatible with green chemistry initiatives and upcoming environmental regulations on antimicrobial compounds.

#### Synthetic Accessibility: Can You Actually Make It?

The SA Score (Synthetic Accessibility Score) estimates how difficult a molecule would be to synthesize, based on the algorithm developed by Ertl and Schuffenhauer.

**Factors Considered**:

- **Fragment Complexity**: How unusual are the structural fragments compared to common building blocks?
- **Ring System Penalties**: Macrocycles, spiro centers, and bridged systems increase difficulty
- **Stereochemistry Burden**: Each chiral center adds synthesis steps
- **Problematic Functional Groups**: Azides, peroxides, and other sensitive groups flagged

**Quat-Specific Extensions**: The standard SA Score has been extended to account for quaternary nitrogen synthesis:

- Simple alkylation of tertiary amines (+15 points)
- Pre-formed pyridinium rings (+15 points)
- Benzyl quat formation (+10 points)
- Multiple quat centers (-10 points each)

**Estimated Synthesis Steps**: The tool also provides a rough estimate of synthetic steps (2-20), helping prioritize candidates for rapid prototyping.

> *Note: The raw SA Score algorithm outputs 1-10 (lower = easier). Quat Generator Pro converts this to a 0-100 scale (higher = easier to synthesize) for consistency with other scores.*

### 2.3 Reference Compound Database

Quat Generator Pro includes a curated database of eight established quaternary ammonium antimicrobials for benchmarking. These represent the diversity of commercial quats and serve as performance baselines.

| Compound | Abbreviation | Primary Application | Key Characteristics |
|----------|-------------|---------------------|---------------------|
| Benzalkonium Chloride (C12) | BAC | Surface disinfectant, antiseptic | Most widely used; moderate safety concerns |
| Cetylpyridinium Chloride | CPC | Mouthwash, throat lozenges | Excellent oral safety; pyridinium headgroup |
| Didecyldimethylammonium Chloride | DDAC | Hard surface disinfectant, algaecide | Twin-chain; best biodegradability |
| Cetrimonium Bromide | CTAB | Hair conditioner, antiseptic | Single long chain; widely studied |
| Benzethonium Chloride | — | Antiseptic, cosmetic preservative | Complex structure; highest LD50 (safest) |
| Domiphen Bromide | — | Throat lozenges, surface disinfectant | Phenoxyethyl linker |
| Octenidine Dihydrochloride | — | Wound antiseptic | Best MIC values; excellent safety profile |
| Dequalinium Chloride | — | Throat lozenges, vaginal antiseptic | Bis-quat structure; antifungal activity |

**Why These Were Chosen**: This set spans the major structural classes (benzyl, pyridinium, aliphatic, bis-quat) and application domains (clinical, consumer, industrial) while including compounds with exceptional properties as aspirational targets (octenidine for potency, DDAC for biodegradability).

### 2.4 Limitations and Appropriate Use

#### These Are Predictions, Not Experimental Data

Every score produced by Quat Generator Pro is a computational prediction based on machine learning models. These models have inherent limitations:

- **Training Data Bias**: Predictions are most reliable for molecules similar to the training set
- **Model Uncertainty**: Confidence varies by compound; novel scaffolds may have less reliable predictions
- **Biological Complexity**: No model perfectly captures the nuances of microbial resistance, metabolism, or ecological fate

> **Important**: Quat Generator Pro is a prioritization tool, not a replacement for wet lab validation. All candidates require experimental confirmation before advancing.

#### When to Trust Predictions

| Situation | Confidence Level | Recommendation |
|-----------|-----------------|----------------|
| Structure similar to reference quats | High | Predictions likely reliable |
| Novel scaffold, familiar functional groups | Moderate | Treat as hypothesis; synthesize and test |
| Unusual heterocycles or complex polycyclic systems | Lower | Higher uncertainty; validate early |
| Predictions near decision thresholds | Variable | Consider multiple candidates spanning the threshold |

#### What the Tool Does NOT Do

- Replace safety testing (cytotoxicity assays, animal studies, clinical trials)
- Guarantee synthetic routes (SA Score indicates feasibility, not a protocol)
- Account for formulation effects (counterion, excipient interactions, pH)
- Predict resistance development or biofilm penetration
- Provide regulatory guidance or approval pathways

The tool suggests candidates for synthesis and provides decision-support data. Your expertise as a research chemist remains essential for final compound selection and experimental design.

---

## 3. Installation Guide

### 3.1 System Requirements

#### Minimum Requirements

| Component | Specification |
|-----------|---------------|
| **RAM** | 8 GB minimum |
| **Storage** | 20 GB free space (for application, models, and database) |
| **Processor** | 4+ CPU cores (Intel i5/AMD Ryzen 5 or better) |
| **Operating System** | Windows 10/11, macOS 11+, or Linux (Ubuntu 20.04+) |
| **Internet** | Required for initial model downloads (~2 GB) |

#### Recommended Configuration

| Component | Specification |
|-----------|---------------|
| **RAM** | 16 GB or more |
| **Storage** | 50 GB SSD |
| **Processor** | 8+ CPU cores |
| **GPU (Optional)** | NVIDIA GPU with 8+ GB VRAM and CUDA 11.8+ |

> **About GPU Acceleration**: A compatible NVIDIA GPU significantly speeds up molecule generation (5-10x faster). However, the application works fully on CPU-only systems—generation simply takes longer.

### 3.2 Step-by-Step Installation

#### For Windows Users

**Step 1: Install Python**

1. Visit https://www.python.org/downloads/
2. Download Python 3.11 (the large yellow "Download Python 3.11.x" button)
3. Run the installer
4. **Critical**: Check the box "Add Python to PATH" at the bottom of the installer window
5. Click "Install Now"
6. When complete, click "Close"

**Step 2: Install Node.js**

1. Visit https://nodejs.org/
2. Download the "LTS" (Long Term Support) version
3. Run the installer, accepting all defaults
4. Click through to completion

**Step 3: Download Quat Generator Pro**

1. Download the application ZIP file from your provided distribution link
2. Right-click the ZIP file and select "Extract All..."
3. Choose a location such as `C:\QuatGenerator` (avoid paths with spaces)
4. Click "Extract"

**Step 4: Run the Setup Script**

1. Open File Explorer and navigate to your extracted folder
2. Double-click `setup_windows.bat`
3. A command window will appear showing installation progress
4. Wait for the message "Setup complete!" (typically 5-10 minutes)

**Step 5: Verify Installation**

1. Double-click `start_windows.bat`
2. Two command windows will open (one for backend, one for frontend)
3. Open your web browser and go to http://localhost:5173
4. You should see the Quat Generator Pro interface

#### For Mac Users

**Step 1: Install Homebrew (if not already installed)**

1. Open Terminal (Applications → Utilities → Terminal)
2. Paste this command and press Enter:
   ```
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
3. Follow the on-screen instructions

**Step 2: Install Python and Node.js**

1. In Terminal, run:
   ```
   brew install python@3.11 node
   ```
2. Wait for installation to complete

**Step 3: Download and Extract Quat Generator Pro**

1. Download the application ZIP file
2. Double-click to extract it to your desired location (e.g., `~/QuatGenerator`)

**Step 4: Run the Setup Script**

1. In Terminal, navigate to the extracted folder:
   ```
   cd ~/QuatGenerator
   ```
2. Make the setup script executable and run it:
   ```
   chmod +x setup_mac.sh
   ./setup_mac.sh
   ```
3. Wait for "Setup complete!" message

**Step 5: Verify Installation**

1. Run the start script:
   ```
   ./start_mac.sh
   ```
2. Open Safari or Chrome and go to http://localhost:5173

### 3.3 First Launch

#### What to Expect on First Run

The first time you launch Quat Generator Pro, the system will download required AI models from the HuggingFace model repository. This is a one-time process.

| Model | Size | Purpose |
|-------|------|---------|
| REINVENT 171M | ~700 MB | Molecule generation |
| ChemBERTa-77M | ~300 MB | Molecular encoding |
| ADMET models | ~500 MB | Safety/environmental predictions |

**Total download: ~1.5-2 GB | Time: 10-15 minutes on typical broadband**

A progress indicator will appear in the terminal window. Do not close the application during this process.

#### Troubleshooting Common Issues

| Problem | Solution |
|---------|----------|
| "Python not found" error | Reinstall Python, ensuring "Add to PATH" is checked |
| "npm not found" error | Reinstall Node.js |
| Download stalls or times out | Check internet connection; retry by restarting the application |
| "Port already in use" error | Another application is using port 8000 or 5173; close it or restart your computer |
| Page won't load in browser | Wait for terminal to show "Application ready" before opening browser |

#### How to Verify Everything Is Working

1. The main interface loads without errors
2. Click the "Generate" tab
3. Set "Number of Molecules" to 5
4. Click "Start Generation"
5. If molecules appear with scores after 1-2 minutes, the installation is successful

### 3.4 Getting Help

**Technical Support**: Report issues at https://github.com/isidoregpt/QuatGenV1/issues

**Documentation**: Additional guides are available in the `docs/` folder of your installation.

---

## 4. Using the Application

### 4.1 The Main Interface

When you open Quat Generator Pro, you see three main areas:

**Left Sidebar (Control Panel)**: Contains all input controls for the currently selected mode. This is where you set generation parameters, enter search queries, or configure benchmarks.

**Central Results Area**: Displays generated molecules, search results, or benchmark reports as cards in a grid layout. Each card shows the molecular structure, scores, and key properties.

**Right Detail Panel**: When you click on a molecule, detailed information appears here including full property breakdowns, predicted MIC values, and ADMET predictions.

#### The Three Main Tabs

| Tab | Purpose | When to Use |
|-----|---------|-------------|
| **Generate** | Create new molecules optimized for your objectives | Starting a discovery campaign or expanding your candidate pool |
| **Search** | Find molecules matching structural patterns or similarity criteria | Looking for specific scaffold types or analogs of a hit compound |
| **Benchmark** | Compare candidates against reference quats | Evaluating whether candidates meet or exceed current standards |

#### Understanding Score Cards

Each molecule card displays four colored scores:

| Color | Score Range | Meaning |
|-------|-------------|---------|
| **Green** | 80-100 | Excellent - high priority |
| **Blue** | 65-79 | Good - comparable to established quats |
| **Yellow** | 50-64 | Moderate - may need optimization |
| **Orange** | 35-49 | Below average - deprioritize |
| **Red** | 0-34 | Poor - do not pursue |

[DIAGRAM: Screenshot of main interface with labeled components]

### 4.2 Generating New Molecules

#### Setting Your Objectives

The objective weights determine how the AI prioritizes different properties during generation. Weights must sum to 100%.

**Efficacy Weight**
- *Default: 40%*
- Increase when: You need the most potent antimicrobials possible; early discovery phase; targeting resistant organisms
- Decrease when: Potency is already adequate; other concerns dominate

**Safety Weight**
- *Default: 30%*
- Increase when: Developing clinical candidates; consumer products with skin contact; regulatory scrutiny is high
- Decrease when: Industrial use with limited human exposure; contained applications

**Environmental Weight**
- *Default: 15%*
- Increase when: Green chemistry initiatives; wastewater discharge concerns; anticipating regulations
- Decrease when: Contained industrial processes; degradation is addressed by formulation

**Synthesis Weight**
- *Default: 15%*
- Increase when: Rapid prototyping needed; limited synthetic resources; early screening of many compounds
- Decrease when: Willing to invest in complex synthesis for optimal candidates

> **Tip**: For a first run, use the defaults. Adjust weights based on what you learn about the generated candidates.

#### Generation Parameters

**Number of Molecules**: How many candidates to generate in this run.

| Batch Size | Use Case |
|------------|----------|
| 10-50 | Quick exploration or testing settings |
| 100-500 | Standard discovery campaign |
| 500-1000 | Comprehensive coverage of chemical space |
| 1000+ | Exhaustive generation (allow extra time) |

**Molecular Weight Constraints**: Restrict generated molecules to a specific MW range.

- Minimum MW: 200 Da (default) - ensures adequate size for activity
- Maximum MW: 600 Da (default) - avoids overly complex structures

**Chain Length Constraints**: For quaternary ammoniums, alkyl chain length is critical for activity.

- Minimum: 8 carbons (shorter chains typically have weak activity)
- Maximum: 18 carbons (longer chains may have solubility issues)

**Diversity Mode**: When enabled, the generator actively avoids producing similar molecules, ensuring structural variety in your output.

#### Running Generation

1. Set your objective weights
2. Adjust constraints if needed
3. Set the number of molecules
4. Click **Start Generation**

**During Generation**:
- A progress bar shows molecules generated vs. target
- The best current scores are displayed
- You can click **Stop Generation** at any time to keep what has been generated

**Typical Generation Times**:

| Molecules | CPU-Only | With GPU |
|-----------|----------|----------|
| 100 | 5-10 min | 1-2 min |
| 500 | 25-45 min | 5-10 min |
| 1000 | 1-2 hours | 15-30 min |

[DIAGRAM: Flowchart showing Generation → Filtering → Benchmarking → Synthesis Queue]

#### Reviewing Results

Once generation completes, results appear in the central grid.

**Sorting Options**:
- Combined Score (default): Overall ranking
- Efficacy: Most potent first
- Safety: Safest first
- SA Score: Easiest to synthesize first

**Filtering**:
- Set minimum scores to hide poor candidates
- Filter by scaffold type (benzyl, pyridinium, etc.)
- Filter by molecular weight range

**Selecting Candidates**:
- Click any molecule to see full details in the right panel
- Star molecules to mark them as favorites
- Add notes for your lab notebook

### 4.3 Searching Your Results

#### Substructure Search

Substructure search finds all molecules containing a specific structural pattern. Patterns are specified using SMARTS notation, but you don't need to write SMARTS yourself—use the template selector.

**Using the Pattern Templates**:

1. Click the pattern dropdown menu
2. Select a predefined pattern:

| Template | What It Finds |
|----------|---------------|
| Any Quaternary | All molecules with [N+] |
| Benzyl Quat | BAC-like benzylammonium structures |
| Pyridinium | CPC-like pyridinium structures |
| Long Chain (C12+) | Structures with long alkyl chains |
| Twin Chain | DDAC-like di-alkyl structures |
| Aromatic Ring | Any aromatic system |

3. Click **Search**
4. Matching molecules appear with highlighted match atoms

**Common Search Scenarios**:

- *Finding all pyridiniums*: Select "Pyridinium" template
- *Finding benzyl quats*: Select "Benzyl Quat" template
- *Finding twin-chain structures*: Select "Twin Chain" template

#### Similarity Search

Similarity search finds molecules structurally related to a query compound.

1. Enter a SMILES string for your query molecule
2. Set the similarity threshold:
   - 0.7 = Moderately similar (same scaffold family)
   - 0.8 = Quite similar (close analogs)
   - 0.85+ = Very similar (nearly identical cores)
3. Click **Search**

**When to Use Similarity Search**:
- Finding analogs of a hit compound from screening
- Exploring variations around a known active
- Identifying structurally related candidates for SAR analysis

### 4.4 Benchmarking Against Known Quats

#### Single Molecule Benchmark

To see how a specific candidate compares to established quats:

1. Click the **Benchmark** tab
2. Enter the SMILES of your candidate (or select from generated molecules)
3. Click **Benchmark**

The report shows:
- Overall score (0-100)
- The three closest reference compounds by similarity
- Property-by-property comparison
- Predicted advantages and disadvantages

#### Batch Benchmarking

To benchmark all your generated molecules at once:

1. Set a minimum score threshold (e.g., 70) to focus on top candidates
2. Set how many top results to display
3. Click **Run Batch Benchmark**

Results show a ranked table with:
- Each candidate's scores
- Best matching reference compound
- Number of properties where the candidate exceeds the reference

#### Reading the Benchmark Report

**Summary Statistics**:
- Average scores for your candidate set
- Distribution of scaffold types
- How many candidates exceed each reference compound

**Scaffold Distribution**:
- What percentage are benzyl quats vs. pyridiniums vs. other types
- Helps understand the chemical diversity of your candidates

**Recommendations**:
- "Excellent candidate" - Outperforms most references
- "Good candidate" - Comparable to established quats
- "Promising but needs optimization" - Some weaknesses to address
- "Not recommended" - Multiple significant liabilities

---

## 5. Interpreting Results

### 5.1 Understanding Scores

#### Score Ranges and What They Mean

| Score Range | Interpretation | Recommended Action |
|-------------|----------------|-------------------|
| **80-100** | Excellent - Top tier candidate | Fast-track for synthesis; prioritize in screening queue |
| **65-79** | Good - Comparable to reference quats | Include in synthesis queue; good prospects |
| **50-64** | Moderate - Has merit but also weaknesses | Consider structural modifications; synthesize if resources allow |
| **35-49** | Below average - Significant concerns | Deprioritize unless unique scaffold of interest |
| **0-34** | Poor - Multiple liabilities | Do not pursue; move to different chemical space |

[DIAGRAM: Example molecule card with score annotations]

#### Efficacy Score Deep Dive

The efficacy score (0-100) integrates multiple antimicrobial activity predictions.

**MIC-Based Scoring**:
- Predicted MIC ≤1 µg/mL → 100 points (exceptional)
- Predicted MIC ~4 µg/mL → ~75 points (good)
- Predicted MIC ~32 µg/mL → ~50 points (moderate)
- Predicted MIC ≥128 µg/mL → 0 points (inactive)

**How Chain Length Affects Predictions**:
- C10-C12: Typically good Gram-positive activity
- C12-C14: Optimal for broad-spectrum (Gram+ and Gram-)
- C14-C16: Strong activity but may have safety trade-offs
- C18+: Decreasing activity due to solubility limitations

**How Headgroup Affects Predictions**:
- Benzyl: Strong activity, well-characterized SAR
- Pyridinium: Excellent membrane interaction, oral safety
- Imidazolium: Good activity, often improved stability
- Simple tetraalkylammonium: Activity depends heavily on chain configuration

> **Why This Matters**: A candidate with an efficacy score of 78 and C14 chain length has different implications than one with score 78 and a novel heterocyclic headgroup. The latter may represent a more innovative lead despite the same score.

#### Safety Score Deep Dive

The safety score (0-100) combines multiple toxicity endpoints.

**hERG Liability (Cardiac)**:
- hERG inhibition probability <20%: Low cardiac concern
- 20-50%: Moderate concern—consider alternatives
- >50%: High concern—deprioritize for systemic applications

**AMES Mutagenicity**:
- Positive prediction: Regulatory red flag for consumer products
- Essential to confirm experimentally before advancement

**Hepatotoxicity (DILI)**:
- Higher probability correlates with liver enzyme elevation risk
- Critical for any compound with potential systemic exposure

**LD50 Predictions**:
- >1000 mg/kg: Low acute toxicity
- 300-1000 mg/kg: Moderate acute toxicity (most quats)
- <300 mg/kg: Higher acute toxicity—handle with care

> **Clinical Context**: For a surface disinfectant with minimal human contact, an LD50 of 300 mg/kg is acceptable. For a mouthwash, you would want LD50 >500 mg/kg and low oral irritation.

#### Environmental Score Deep Dive

The environmental score (0-100) addresses ecological fate and effects.

**Ready Biodegradability**:
- "Readily biodegradable": >60% breakdown in 28-day test
- "Inherently biodegradable": Slower degradation but eventual breakdown
- "Persistent": Limited biodegradation—environmental accumulation risk

**Bioconcentration Factor (BCF)**:
- BCF <100: Low bioaccumulation risk
- BCF 100-1000: Moderate bioaccumulation
- BCF >1000: High bioaccumulation—avoid

**Aquatic Toxicity**:
- Fish LC50 <1 mg/L: Very toxic to aquatic life
- 1-10 mg/L: Toxic
- >10 mg/L: Lower aquatic toxicity

Most quaternary ammoniums are inherently toxic to aquatic organisms due to their surfactant properties. The score identifies candidates with relatively lower impact.

### 5.2 MIC Predictions Explained

| Predicted MIC (µg/mL) | Activity Class | Context |
|-----------------------|----------------|---------|
| **≤1** | Excellent | Comparable to octenidine; top-tier antimicrobial |
| **1-4** | Very Good | Comparable to best commercial quats |
| **4-8** | Good | Suitable for most disinfection applications |
| **8-32** | Moderate | May work for surface applications; limited clinical utility |
| **32-128** | Weak | Not recommended for primary antimicrobial use |
| **>128** | Inactive | No meaningful antimicrobial activity |

**Confidence Scores**:

Each MIC prediction includes a confidence value (0-1):
- >0.8: High confidence—prediction based on similar known compounds
- 0.5-0.8: Moderate confidence—reasonable prediction
- <0.5: Low confidence—treat as hypothesis only

**Applicability Domain**:

Predictions are most reliable when:
- The molecule is structurally similar to training data (Tanimoto >0.6 to reference quats)
- Molecular weight is 200-600 Da
- LogP is 1-7

Predictions may be unreliable for:
- Highly unusual scaffolds with no similar training examples
- Very large or very small molecules
- Compounds with unusual charge distributions

### 5.3 Structural Features and Activity

#### Positive Indicators

| Feature | Why It Helps | Score Contribution |
|---------|--------------|-------------------|
| Single C12-C16 alkyl chain | Optimal membrane insertion | +15-25 efficacy |
| Benzyl substituent | Enhanced membrane disruption | +10-15 efficacy |
| Pyridinium headgroup | Favorable orientation | +10-15 efficacy |
| LogP 3-5 | Balanced hydrophilicity | +10 efficacy, +10 safety |
| MW 300-450 | Drug-like properties | +5-10 overall |
| Single quaternary center | Simpler synthesis, clearance | +15 SA, +10 safety |

#### Warning Signs

| Feature | Concern | Score Impact |
|---------|---------|--------------|
| Chain >C18 | Solubility issues, aggregation | -10 efficacy, -5 SA |
| Multiple quat centers | Synthesis complexity, charge density | -10 SA per center |
| Highly lipophilic (LogP >6) | Bioaccumulation, toxicity | -15 environmental, -10 safety |
| Large ring systems | Synthesis difficulty | -10 to -20 SA |
| Nitro groups | Mutagenicity risk | -20 safety |
| Halogenated aromatics | Persistence | -15 environmental |

### 5.4 From Prediction to Synthesis

#### Prioritization Strategy

1. **Rank by combined score**: Start with the highest overall performers
2. **Consider scaffold diversity**: Don't synthesize five variations of the same scaffold—pick representatives from different structural classes
3. **Balance novelty vs. confidence**: Mix some novel scaffolds (more uncertainty but more innovation) with traditional scaffolds (higher prediction confidence)

#### What to Synthesize First

| Priority Level | Criteria | Action |
|---------------|----------|--------|
| **Tier 1** | Combined score >80, SA score >70, novel scaffold | Synthesize immediately |
| **Tier 2** | Combined score >75, benchmark beats 2+ references | Add to synthesis queue |
| **Tier 3** | Combined score >70, unique structural feature | Synthesize if resources allow |
| **Tier 4** | Combined score 60-70, represents underexplored space | Hold for future expansion |

> **Practical Advice**: Start by synthesizing 5-10 diverse candidates from Tier 1-2. Test them experimentally. Use the results to calibrate your trust in the predictions and refine subsequent candidate selection.

---

## 6. Worked Examples

### Example 1: Finding a Safer Alternative to Benzalkonium Chloride

**Scenario**: Your research group wants to develop BAC alternatives with improved safety profiles for dermal applications while maintaining antimicrobial efficacy.

**Step 1: Set Your Objectives**

In the Generate tab, adjust the weights:
- Efficacy: 35% (maintain potency)
- Safety: 45% (prioritize human safety)
- Environmental: 10%
- Synthesis: 10%

**Step 2: Configure Constraints**

- Minimum MW: 250 Da
- Maximum MW: 500 Da
- Chain length: 10-16 carbons
- Enable diversity mode

**Step 3: Generate Candidates**

Set number of molecules to 200 and click Start Generation.

**Step 4: Filter Results**

After generation completes:
- Set minimum safety score to 70
- Set minimum efficacy score to 65

**Step 5: Benchmark Against BAC**

Select the top 10 filtered candidates. Run batch benchmark with BAC as the primary comparator.

**Step 6: Interpret Results**

Look for candidates showing:
- "Advantages: Improved safety profile" in the benchmark report
- LD50 predictions >300 mg/kg (BAC is ~240 mg/kg)
- Lower hERG inhibition probability
- Similar or better efficacy predictions

**Expected Outcome**: You should identify 3-5 candidates that match BAC's efficacy (score ~75-85) while showing improved safety metrics (safety score 75+ vs. BAC's ~60).

### Example 2: Exploring Pyridinium Scaffolds

**Scenario**: You hypothesize that novel pyridinium structures might offer better environmental profiles than traditional CPC.

**Step 1: Generate a Diverse Set**

Use default weights but generate 300 molecules with high diversity enabled.

**Step 2: Search for Pyridiniums**

After generation, go to the Search tab:
- Select "Pyridinium" from the pattern templates
- Click Search

**Step 3: Analyze Environmental Scores**

Sort the results by environmental score (highest first).

**Step 4: Compare to CPC**

Select the top 5 environmentally-scored pyridiniums. Run benchmark against CPC specifically.

**Step 5: Identify Candidates**

Look for structures that:
- Have environmental scores >65 (CPC scores ~55)
- Maintain efficacy scores >70
- Show predicted advantages in biodegradability

**Step 6: Examine Structural Features**

Click on promising candidates to examine what structural modifications improve environmental score:
- Shorter chains?
- Hydroxyl groups?
- Ether linkages?

**Expected Outcome**: Identification of pyridinium modifications (e.g., polyethylene glycol linkers, hydroxyl-terminated chains) that enhance biodegradability while maintaining antimicrobial activity.

### Example 3: Rapid Lead Generation for a New Project

**Scenario**: You're starting a new project and need 20 diverse quaternary ammonium leads for initial screening, prioritizing ease of synthesis.

**Step 1: Set Synthesis-Focused Objectives**

Adjust weights:
- Efficacy: 30%
- Safety: 25%
- Environmental: 15%
- Synthesis: 30% (elevated for rapid prototyping)

**Step 2: Generate with High Diversity**

- Number of molecules: 500
- Enable diversity mode
- Chain length: 8-16 (broader range for diversity)

**Step 3: Filter for Synthesizability**

After generation:
- Set minimum SA score to 75
- Set minimum combined score to 60

**Step 4: Run Batch Benchmark**

Benchmark all filtered candidates against the full reference set.

**Step 5: Select Diverse Representatives**

From the benchmark report, select:
- 5 benzyl quats (best scores in each)
- 5 pyridiniums
- 5 aliphatic quats
- 5 novel/other scaffolds

**Step 6: Create Synthesis Priority List**

Export your 20 selections to CSV. Sort by SA score descending. This is your synthesis queue, with easiest-to-make compounds first.

**Expected Outcome**: A prioritized list of 20 structurally diverse quaternary ammonium leads, each with combined scores >60 and SA scores >75, representing multiple scaffold classes and ready for rapid synthesis and initial screening.

---

## 7. Quick Reference

### 7.1 SMARTS Pattern Cheat Sheet

| Pattern Name | What It Finds | Example Match | When to Use |
|--------------|---------------|---------------|-------------|
| Any Quaternary | All quaternary nitrogens | All quats in database | General survey of candidates |
| Aliphatic Quat | Tetraalkylammonium only | DDAC, CTAB | Traditional surfactant types |
| Benzyl Quat | Benzylammonium structures | BAC, benzethonium | BAC-like candidates |
| Pyridinium | Pyridinium ring salts | CPC, dequalinium | CPC-like candidates |
| Imidazolium | Imidazolium ring salts | Novel imidazolium quats | Exploring imidazolium class |
| Long Chain (C12+) | 12+ carbon alkyl chain | Most antimicrobial quats | Amphiphilic structures |
| Twin Chain | Di-alkyl substitution | DDAC | DDAC-like structures |
| Hydroxyl | Hydroxy groups present | Hydroxyl-modified quats | Green chemistry modifications |
| Aromatic | Any aromatic ring | Benzyl, phenyl derivatives | Aromatic-containing candidates |

### 7.2 Reference Compounds Summary

| Compound | Key Strengths | Key Weaknesses | Typical Application |
|----------|---------------|----------------|---------------------|
| **BAC** | Broad-spectrum efficacy | Moderate safety concerns | Surface disinfection |
| **CPC** | Good oral safety | Lower Gram-negative activity | Mouthwash |
| **DDAC** | Best biodegradability | Lower LD50 | Industrial disinfection |
| **CTAB** | Well-studied, accessible | Moderate safety | Laboratory/research |
| **Benzethonium** | Highest LD50 (safest) | Complex structure | Consumer antiseptics |
| **Domiphen** | Good balance | Moderate all scores | Throat lozenges |
| **Octenidine** | Best efficacy + safety | Complex bis-structure | Wound care |
| **Dequalinium** | Excellent antifungal | Bis-quat synthesis | Oral/vaginal antiseptic |

### 7.3 Troubleshooting Common Issues

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| "Model not loaded" error | First run, models downloading | Wait 10-15 minutes for download to complete |
| Generation very slow | CPU-only mode | Normal for CPU; consider GPU for faster runs |
| No results from substructure search | Pattern too specific | Broaden pattern or use similarity search instead |
| All scores unexpectedly low | Unusual chemistry | Compounds may be outside model training domain |
| Application won't start | Port conflict | Close other applications; restart computer |
| Results look similar | Diversity mode off | Enable diversity mode in generation settings |
| "Invalid SMILES" error | Typo in SMILES input | Check SMILES syntax; use structure drawing tool to verify |
| Export fails | Permission issue | Run as administrator or choose different save location |

### 7.4 Glossary

| Term | Definition |
|------|------------|
| **SMILES** | Simplified Molecular Input Line Entry System; a text representation of molecular structure (e.g., CCO for ethanol) |
| **SMARTS** | SMILES Arbitrary Target Specification; a pattern language for substructure searching |
| **Tanimoto Similarity** | A measure of structural similarity between two molecules (0 = no similarity, 1 = identical) |
| **SA Score** | Synthetic Accessibility Score; in Quat Generator Pro displayed as 0-100 (higher = easier to synthesize) |
| **ADMET** | Absorption, Distribution, Metabolism, Excretion, Toxicity; pharmacokinetic and safety properties |
| **MIC** | Minimum Inhibitory Concentration; lowest concentration that prevents visible microbial growth |
| **hERG** | Human ether-à-go-go related gene; a cardiac ion channel; inhibition causes arrhythmia risk |
| **AMES** | Bacterial reverse mutation test for mutagenicity; a regulatory requirement |
| **LogP** | Partition coefficient (octanol/water); measures lipophilicity |
| **TPSA** | Topological Polar Surface Area; correlates with membrane permeability |
| **BCF** | Bioconcentration Factor; measures accumulation in aquatic organisms |
| **LD50** | Lethal Dose 50%; dose killing 50% of test animals; higher is safer |
| **DILI** | Drug-Induced Liver Injury; hepatotoxicity risk |
| **BBB** | Blood-Brain Barrier; penetration indicates potential CNS effects |
| **Pareto Frontier** | Set of candidates where no single property can be improved without worsening another |

---

## 8. Appendices

### Appendix A: The Reference Compound Database

#### Benzalkonium Chloride (C12) - BAC

- **CAS Number**: 8001-54-5
- **ChEMBL ID**: CHEMBL578
- **Structure**: Benzyldimethyldodecylammonium chloride
- **SMILES**: `CCCCCCCCCCCC[N+](C)(C)Cc1ccccc1.[Cl-]`
- **Molecular Weight**: 340.0 Da

**Typical MIC Values (µg/mL)**:
- *S. aureus*: 1-8
- *E. coli*: 8-64
- *P. aeruginosa*: 32-256
- *C. albicans*: 4-16

**Safety Profile**:
- LD50 (oral, rat): 240 mg/kg
- Skin irritation: Moderate
- Eye irritation: Severe

**Environmental Properties**:
- Aquatic LC50 (fish): 0.5 mg/L
- Biodegradability: Inherently biodegradable

**Applications**: Surface disinfectant, antiseptic, preservative in pharmaceuticals and cosmetics

---

#### Cetylpyridinium Chloride - CPC

- **CAS Number**: 123-03-5
- **ChEMBL ID**: CHEMBL1354
- **Structure**: 1-Hexadecylpyridinium chloride
- **SMILES**: `CCCCCCCCCCCCCCCC[n+]1ccccc1.[Cl-]`
- **Molecular Weight**: 340.0 Da

**Typical MIC Values (µg/mL)**:
- *S. aureus*: 0.5-4
- *E. coli*: 4-32
- *P. aeruginosa*: 16-128
- *C. albicans*: 2-8

**Safety Profile**:
- LD50 (oral, rat): 200 mg/kg
- Skin irritation: Mild
- Eye irritation: Moderate

**Environmental Properties**:
- Aquatic LC50 (fish): 0.3 mg/L
- Biodegradability: Inherently biodegradable

**Applications**: Mouthwash active ingredient, throat lozenges, surface disinfectant

---

#### Didecyldimethylammonium Chloride - DDAC

- **CAS Number**: 7173-51-5
- **ChEMBL ID**: CHEMBL1201135
- **Structure**: Didecyldimethylammonium chloride
- **SMILES**: `CCCCCCCCCC[N+](C)(C)CCCCCCCCCC.[Cl-]`
- **Molecular Weight**: 362.1 Da

**Typical MIC Values (µg/mL)**:
- *S. aureus*: 0.5-4
- *E. coli*: 2-16
- *P. aeruginosa*: 8-64
- *C. albicans*: 1-8

**Safety Profile**:
- LD50 (oral, rat): 84 mg/kg (lower than BAC—handle with care)
- Skin irritation: Moderate
- Eye irritation: Severe

**Environmental Properties**:
- Aquatic LC50 (fish): 0.2 mg/L
- Biodegradability: Readily biodegradable (best in class)

**Applications**: Hard surface disinfectant, algaecide, wood preservative

---

#### Cetrimonium Bromide - CTAB

- **CAS Number**: 57-09-0
- **ChEMBL ID**: CHEMBL447964
- **Structure**: Cetyltrimethylammonium bromide
- **SMILES**: `CCCCCCCCCCCCCCCC[N+](C)(C)C.[Br-]`
- **Molecular Weight**: 364.5 Da

**Typical MIC Values (µg/mL)**:
- *S. aureus*: 1-4
- *E. coli*: 4-16
- *P. aeruginosa*: 16-64
- *C. albicans*: 2-8

**Safety Profile**:
- LD50 (oral, rat): 410 mg/kg
- Skin irritation: Moderate
- Eye irritation: Severe

**Environmental Properties**:
- Aquatic LC50 (fish): 0.4 mg/L
- Biodegradability: Inherently biodegradable

**Applications**: Hair conditioner, antiseptic, DNA extraction in molecular biology

---

#### Benzethonium Chloride

- **CAS Number**: 121-54-0
- **ChEMBL ID**: CHEMBL1236088
- **Structure**: Benzyldimethyl[2-[2-[4-(1,1,3,3-tetramethylbutyl)phenoxy]ethoxy]ethyl]ammonium chloride
- **SMILES**: `CC(C)(C)CC(C)(C)c1ccc(OCCOCC[N+](C)(C)Cc2ccccc2)cc1.[Cl-]`
- **Molecular Weight**: 448.1 Da

**Typical MIC Values (µg/mL)**:
- *S. aureus*: 1-8
- *E. coli*: 8-32
- *P. aeruginosa*: 32-128
- *C. albicans*: 4-16

**Safety Profile**:
- LD50 (oral, rat): 368 mg/kg (highest in database—safest)
- Skin irritation: Mild
- Eye irritation: Moderate

**Environmental Properties**:
- Aquatic LC50 (fish): 1.0 mg/L
- Biodegradability: Inherently biodegradable

**Applications**: Antiseptic wipes, cosmetic preservative, first aid products

---

#### Domiphen Bromide

- **CAS Number**: 538-71-6
- **ChEMBL ID**: CHEMBL1201247
- **Structure**: Dodecyldimethyl(2-phenoxyethyl)ammonium bromide
- **SMILES**: `CCCCCCCCCCCC[N+](C)(C)CCOc1ccccc1.[Br-]`
- **Molecular Weight**: 414.5 Da

**Typical MIC Values (µg/mL)**:
- *S. aureus*: 1-8
- *E. coli*: 8-32
- *P. aeruginosa*: 32-128
- *C. albicans*: 4-16

**Safety Profile**:
- LD50 (oral, rat): 320 mg/kg
- Skin irritation: Mild
- Eye irritation: Moderate

**Environmental Properties**:
- Aquatic LC50 (fish): 0.8 mg/L
- Biodegradability: Inherently biodegradable

**Applications**: Throat lozenges, surface disinfectant, antiseptic

---

#### Octenidine Dihydrochloride

- **CAS Number**: 70775-75-6
- **PubChem CID**: 402617
- **Structure**: Bis-cationic bisphenyl biguanide with octamethylene linker
- **SMILES**: `CC(C)(C)c1ccc(NC(=N)NC(=N)NCCCCCCCCNC(=N)NC(=N)Nc2ccc(C(C)(C)C)cc2)cc1.[Cl-].[Cl-]`
- **Molecular Weight**: 623.8 Da

**Typical MIC Values (µg/mL)**:
- *S. aureus*: 0.25-2 (excellent—best in class)
- *E. coli*: 0.5-4
- *P. aeruginosa*: 2-16
- *C. albicans*: 0.5-4

**Safety Profile**:
- LD50 (oral, rat): 1850 mg/kg (excellent safety margin)
- Skin irritation: Mild
- Eye irritation: Mild

**Environmental Properties**:
- Biodegradability: Inherently biodegradable

**Applications**: Wound antiseptic, mucous membrane disinfection, surgical scrub

---

#### Dequalinium Chloride

- **CAS Number**: 522-51-0
- **ChEMBL ID**: CHEMBL1523
- **Structure**: Bis-quinolinium linked by decamethylene chain
- **SMILES**: `Cc1cc2c(N)cc(N)cc2[n+](CCCCCCCCCC[n+]3c4cc(N)cc(N)c4cc3C)c1.[Cl-].[Cl-]`
- **Molecular Weight**: 527.6 Da

**Typical MIC Values (µg/mL)**:
- *S. aureus*: 0.5-4
- *E. coli*: 4-16
- *P. aeruginosa*: 8-64
- *C. albicans*: 0.5-4 (excellent antifungal)

**Safety Profile**:
- LD50 (oral, rat): 150 mg/kg
- Skin irritation: Mild
- Eye irritation: Mild

**Environmental Properties**:
- Aquatic LC50 (fish): 0.5 mg/L
- Biodegradability: Inherently biodegradable

**Applications**: Throat lozenges, vaginal antiseptic, oral antiseptic

---

### Appendix B: Model Training Data Sources

#### ChEMBL Database

The primary source of antimicrobial activity data is ChEMBL, the European Molecular Biology Laboratory's open database of bioactive molecules.

**Data Used**:
- Quaternary ammonium compounds with reported MIC values
- Screening data against bacterial and fungal pathogens
- ADMET endpoint measurements

**Quality Considerations**:
- Only data from peer-reviewed sources included
- MIC values standardized to µg/mL
- Activity cliffs and outliers manually reviewed

#### REINVENT Pre-training

The REINVENT generative model was pre-trained on:
- ZINC database (~750 million drug-like molecules)
- ChEMBL bioactive compound set
- PubChem compound subset

This broad training enables the model to generate chemically valid, drug-like structures across diverse chemical space.

#### Quat-Specific Fine-tuning

The model was fine-tuned specifically for quaternary ammonium generation using:
- The 8 reference compounds and their analogs
- ChEMBL quaternary ammonium compounds with antimicrobial data
- Augmented data from structural enumeration of known quat scaffolds

### Appendix C: Algorithm Details for the Curious

#### REINVENT: Molecular Generation by Reinforcement Learning

REINVENT (Olivecrona et al., 2017) treats molecule generation as a sequence generation task. A recurrent neural network (RNN) generates SMILES strings token-by-token, learning the probability distribution of valid chemical structures.

**Architecture**:
- Encoder: Character-level embedding of SMILES tokens
- Generator: LSTM layers (Long Short-Term Memory)
- Decoder: Softmax output over vocabulary of SMILES tokens

**Reinforcement Learning**:
- Agent: The RNN generator being optimized
- Environment: Your specified scoring function
- Reward: Weighted combination of efficacy, safety, environmental, and SA scores
- Policy Gradient: REINFORCE algorithm with experience replay

**Key Parameters**:
- σ (sigma): Score threshold for considering a molecule "good" (default: 60)
- Replay buffer: Stores high-scoring molecules for experience replay
- KL divergence penalty: Prevents the agent from drifting too far from the prior distribution

**Reference**: Olivecrona, M. et al. (2017). "Molecular de-novo design through deep reinforcement learning." *Journal of Cheminformatics*, 9(1), 48.

#### ChemBERTa: Molecular Representation Learning

ChemBERTa (Chithrananda et al., 2020) is a BERT-style transformer model trained on SMILES strings using masked language modeling.

**Architecture**:
- 77 million parameters
- 12 transformer layers
- 768-dimensional embeddings

**Usage in Quat Generator Pro**:
- Converts SMILES to dense vector representations
- Enables similarity calculations between molecules
- Provides input features for MIC prediction neural network

**Reference**: Chithrananda, S. et al. (2020). "ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction." *NeurIPS Workshop on Machine Learning for Molecules*.

#### SA Score: Synthetic Accessibility

The SA Score algorithm (Ertl & Schuffenhauer, 2009) estimates synthetic feasibility on a scale of 1 (easy) to 10 (very difficult).

**Components**:
1. **Fragment Score**: Based on frequency of molecular fragments in synthesizable compounds
2. **Complexity Penalty**: Bertz complexity index measuring molecular information content
3. **Stereochemistry Penalty**: Cost per chiral center and stereo double bond
4. **Ring Penalties**: Macrocycles, spiro, and bridged systems

**Quat Extensions**:
The original algorithm was extended with quaternary nitrogen-specific considerations:
- Bonus for simple alkylation reactions
- Bonus for pre-formed heterocyclic quats
- Penalty for multiple quaternary centers

**Reference**: Ertl, P. & Schuffenhauer, A. (2009). "Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions." *Journal of Cheminformatics*, 1, 8.

---

### Appendix D: One-Page Quick Reference (Printable)

---

#### SCORE INTERPRETATION (0-100, higher = better)

| Score | Meaning | Action |
|-------|---------|--------|
| 80-100 | Excellent | Synthesize first |
| 65-79 | Good | Add to queue |
| 50-64 | Moderate | Consider modifications |
| 35-49 | Below average | Deprioritize |
| 0-34 | Poor | Do not pursue |

---

#### WEIGHT PRESETS

**Discovery Phase (default):** Efficacy 40%, Safety 30%, Environmental 15%, Synthesis 15%

**Safety-Focused:** Efficacy 35%, Safety 45%, Environmental 10%, Synthesis 10%

**Rapid Prototyping:** Efficacy 30%, Safety 25%, Environmental 15%, Synthesis 30%

**Green Chemistry:** Efficacy 30%, Safety 25%, Environmental 30%, Synthesis 15%

---

#### MIC INTERPRETATION

| MIC (µg/mL) | Activity | Reference |
|-------------|----------|-----------|
| ≤1 | Excellent | Octenidine-class |
| 1-4 | Very Good | Best commercial quats |
| 4-8 | Good | Standard disinfection |
| 8-32 | Moderate | Limited utility |
| >32 | Weak/Inactive | Not recommended |

---

#### FIRST RUN CHECKLIST

- [ ] Generate 100-200 molecules with default weights
- [ ] Filter: Safety >70, Efficacy >65
- [ ] Run batch benchmark against references
- [ ] Select 5-10 diverse scaffolds
- [ ] Prioritize by SA Score for synthesis order
- [ ] Test experimentally → recalibrate trust in predictions

---

#### KEY STRUCTURAL FEATURES

**Positive:** C12-C16 chain, benzyl group, pyridinium, LogP 3-5, MW 300-450

**Negative:** Chain >C18, multiple quats, LogP >6, nitro groups, halogenated aromatics

---

## Document Information

**Version**: 1.0  
**Last Updated**: January 2026  
**Intended Audience**: Research chemists specializing in quaternary ammonium antimicrobials  
**Document Type**: User instruction manual

**Feedback**: If you find errors or have suggestions for improving this manual, please report them at https://github.com/isidoregpt/QuatGenV1/issues

---

*Quat Generator Pro is a research tool intended to support and accelerate antimicrobial discovery. All predictions should be validated experimentally. This software does not replace professional judgment, regulatory compliance, or standard safety testing.*
