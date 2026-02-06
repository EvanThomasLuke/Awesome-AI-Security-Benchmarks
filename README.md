# Awesome-AI-Security-Benchmarks
List of AI Security benchmarks 



**Total unique benchmarks: ~175 (165 core cybersecurity + 10 AI security adjacent)**

---

## Complete Unified Table

| Benchmark | Year | Type | Links | Definition |
|-----------|------|------|-------|------------|
| **KNOWLEDGE & Q&A** |||||
| SecEval | 2023 | Dataset | [GitHub](https://github.com/XuanwuAI/SecEval) &#124; [HuggingFace](https://huggingface.co/datasets/XuanwuAI/SecEval) &#124; [Website](https://xuanwuai.github.io/SecEval/) | 2000+ multiple-choice questions across 9 security domains, GPT-4 generated from textbooks and standards |
| SecQA | 2023 | Dataset | [arXiv:2312.15838](https://arxiv.org/abs/2312.15838) &#124; [HuggingFace](https://huggingface.co/datasets/zefang-liu/secqa) | Multiple-choice questions from "Computer Systems Security" textbook with two complexity tiers |
| CyberMetric | 2024 | Dataset | [arXiv:2402.07688](https://arxiv.org/abs/2402.07688) &#124; [GitHub](https://github.com/cybermetric/CyberMetric) | RAG-generated MCQ datasets (80/500/2000/10000 variants) from NIST standards, research papers, and cybersecurity books |
| SECURE | 2024 | Dataset | [arXiv:2405.20441](https://arxiv.org/abs/2405.20441) &#124; [GitHub](https://github.com/aiforsec/SECURE) | 6 datasets (MAET, CWET, KCV, VOOD, RERT, CPST) for ICS cybersecurity evaluation |
| SEvenLLM-Bench | 2024 | Dataset | [arXiv:2405.03446](https://arxiv.org/abs/2405.03446) &#124; [GitHub](https://github.com/SEvenLLM/SEvenLLM) | 1,300-sample bilingual benchmark for CTI analytical capabilities with 28 tasks |
| CS-Eval | 2024 | Dataset | [arXiv:2411.16239](https://arxiv.org/abs/2411.16239) &#124; [GitHub](https://github.com/CS-EVAL/CS-Eval) | Comprehensive LLM benchmark for cybersecurity knowledge evaluation |
| SecBench | 2024 | Dataset | [arXiv:2412.20787](https://arxiv.org/abs/2412.20787) &#124; [HuggingFace](https://huggingface.co/datasets/secbench-hf/SecBench) | Multi-dimensional, multi-language benchmarking dataset for LLMs in cybersecurity |
| CSEBenchmark | 2025 | Dataset | [arXiv:2504.11783](https://arxiv.org/abs/2504.11783) &#124; [GitHub](https://github.com/NASP-THU/CSEBenchmark) | Fine-grained cyber-security evaluation based on 345 knowledge points (IEEE S&P 2025) |
| AttackQA | 2024 | Dataset | [arXiv:2408.16847](https://arxiv.org/abs/2408.16847) | Dataset for assisting cybersecurity operations using LLMs |
| CyberBench | 2024 | Dataset | [GitHub](https://github.com/jpmorganchase/CyberBench) | Multi-task benchmark over multiple cybersecurity NLP task datasets (JPMorganChase) |
| CySecBench | 2025 | Dataset | [arXiv:2501.01335](https://arxiv.org/abs/2501.01335) | Generative AI-based cybersecurity-focused prompt dataset (12,662 prompts) |
| AthenaBench | 2025 | Dataset | [arXiv:2511.01144](https://arxiv.org/abs/2511.01144) | CTI-focused dynamic evaluation tasks for LLMs |
| OCCULT | 2025 | Framework | [arXiv:2502.15797](https://arxiv.org/abs/2502.15797) | Offensive tactic knowledge evaluation framework |
| MMLU Computer Security | 2021 | Dataset | [HuggingFace](https://huggingface.co/datasets/cais/mmlu) | Standard MMLU benchmark's dedicated computer security subset |
| MMLU Security Studies | 2021 | Dataset | [HuggingFace](https://huggingface.co/datasets/cais/mmlu) | Standard MMLU benchmark's security studies subset |
| **CTF & OFFENSIVE SECURITY** |||||
| InterCode-CTF | 2023 | Environment | [arXiv:2306.14898](https://arxiv.org/abs/2306.14898) &#124; [GitHub](https://github.com/princeton-nlp/intercode) &#124; [Website](https://intercode-benchmark.github.io/) | 100 CTF tasks from PicoCTF with containerized bash execution environment |
| NYU CTF Bench | 2024 | Environment | [arXiv:2406.05590](https://arxiv.org/abs/2406.05590) &#124; [GitHub](https://github.com/NYU-LLM-CTF/NYU_CTF_Bench) &#124; [Website](https://nyu-llm-ctf.github.io/) | 200 challenges from CSAW CTF spanning crypto, forensics, web, reverse engineering, pwn |
| Cybench | 2024 | Environment | [arXiv:2408.08926](https://arxiv.org/abs/2408.08926) &#124; [GitHub](https://github.com/andyzorigin/cybench) &#124; [Website](https://cybench.github.io/) | 40 professional CTF tasks from HackTheBox, SekaiCTF, Glacier, HKCert |
| 3CB | 2024 | Environment | [arXiv:2410.09114](https://arxiv.org/abs/2410.09114) &#124; [Website](https://cybercapabilities.org/) | Advanced CTF challenges designed to be harder than Cybench |
| BountyBench | 2025 | Environment | [Website](https://bountybench.com/) &#124; [GitHub](https://github.com/bountybench/bountybench) | Real-world vulnerability detection, exploitation, and patching with dollar impact |
| RCTF2 | 2025 | Environment | [arXiv:2510.24317](https://arxiv.org/abs/2510.24317) | Robotics/cyber-physical sub-benchmark inside CAIBench (27 challenges) |
| CTFKnow | 2025 | Dataset | [arXiv:2501.09564](https://arxiv.org/abs/2501.09564) | CTF technical knowledge benchmark built from CTF writeups |
| CTF-Dojo | 2025 | Environment | [arXiv:2501.17535](https://arxiv.org/abs/2501.17535) | Large-scale executable runtime of containerized CTF challenges |
| AIRTBench | 2025 | Environment | [arXiv:2506.14682](https://arxiv.org/abs/2506.14682) | Autonomous AI red-teaming benchmark; 70 black-box CTFs |
| Crowdsourced CTF Elicitation | 2025 | Framework | [arXiv:2505.19915](https://arxiv.org/abs/2505.19915) | Evaluating AI cyber capabilities with crowdsourced elicitation (CTF-based) |
| **PENETRATION TESTING** |||||
| PentestGPT Benchmark | 2023 | Harness | [arXiv:2308.06782](https://arxiv.org/abs/2308.06782) &#124; [GitHub](https://github.com/GreyDGL/PentestGPT) | 182 sub-tasks from HackTheBox and VulnHub for penetration testing |
| AutoPenBench | 2024 | Environment | [arXiv:2410.03225](https://arxiv.org/abs/2410.03225) &#124; [GitHub](https://github.com/lucagioacchini/auto-pen-bench) | Benchmarking generative agents for pentesting with defined milestones |
| PentestEval | 2025 | Harness | [arXiv:2512.14233](https://arxiv.org/abs/2512.14233) | Stage-specific evaluation across entire pentesting lifecycle |
| Vulhub Benchmark | 2024 | Environment | [GitHub](https://github.com/vulhub/vulhub) | Docker-based vulnerable environments for automated pentesting |
| CHECKMATE | 2025 | Framework | [arXiv:2503.12735](https://arxiv.org/abs/2503.12735) | Planning + LLM agents for automated penetration testing |
| PenHeal | 2023 | Framework | [arXiv:2312.03015](https://arxiv.org/abs/2312.03015) | AI cybersecurity capabilities evaluation for pentesting and healing |
| AutoAttacker | 2024 | Framework | [arXiv:2403.01038](https://arxiv.org/abs/2403.01038) | Automated cyber-attack generation and evaluation benchmark |
| AI-Pentest-Benchmark | 2024 | Environment | [arXiv:2410.17141](https://arxiv.org/abs/2410.17141) &#124; [GitHub](https://github.com/isamu-isozaki/AI-Pentest-Benchmark) | 13 VulnHub VMs with 152 subtasks ("Towards Automated Penetration Testing") |
| TermiBench | 2025 | Environment | [arXiv:2509.09207](https://arxiv.org/abs/2509.09207) | Real-world pentesting benchmark; 510 hosts/30 CVEs; shell-focused |
| CTFTiny | 2025 | Dataset | [arXiv:2508.05674](https://arxiv.org/abs/2508.05674) | Curated 50 representative CTF challenges for rapid evaluation of offensive-security agents |
| CVE-Bench | 2025 | Environment | [arXiv:2503.17332](https://arxiv.org/abs/2503.17332) &#124; [GitHub](https://github.com/uiuc-kang-lab/cve-bench) | Real-world webapp vulnerability exploitation based on critical CVEs |
| XBOW Validation | 2025 | Environment | [Website](https://xbow.com/) | 104 web security challenges for autonomous offensive tools |
| CyberGym | 2025 | Environment | [arXiv:2506.02548](https://arxiv.org/abs/2506.02548) &#124; [HuggingFace](https://huggingface.co/datasets/sunblaze-ucb/cybergym) | 1,507 instances from 188 OSS projects with ASan/UBSan validation |
| Cyberattack Capabilities Framework | 2025 | Framework | [arXiv:2503.11917](https://arxiv.org/abs/2503.11917) &#124; [DeepMind Blog](https://deepmind.google/discover/blog/a-framework-for-evaluating-ai-cyberattack-capabilities/) | End-to-end attack chain evaluation with representative archetypes |
| **VULNERABILITY DETECTION** |||||
| VulDetectBench | 2024 | Dataset | [arXiv:2406.07595](https://arxiv.org/abs/2406.07595) &#124; [GitHub](https://github.com/Sweetaroo/VulDetectBench) | Vulnerability detection benchmark with 5 tasks (identify/classify/localize, etc.) |
| CyberSecEval v1 | 2023 | Harness | [arXiv:2312.04724](https://arxiv.org/abs/2312.04724) &#124; [GitHub](https://github.com/meta-llama/PurpleLlama) | Evaluates LLMs for insecure code generation propensity (Purple Llama) |
| CyberSecEval v2 | 2024 | Harness | [arXiv:2404.13161](https://arxiv.org/abs/2404.13161) &#124; [GitHub](https://github.com/meta-llama/PurpleLlama) | Extended benchmark with vulnerability exploitation and prompt injection tests |
| CyberSecEval v3 | 2024 | Harness | [arXiv:2408.01605](https://arxiv.org/abs/2408.01605) &#124; [GitHub](https://github.com/meta-llama/PurpleLlama) | Further advances in evaluating LLM cybersecurity risks |
| CyberSecEval v4 | 2025 | Meta-benchmark | [GitHub](https://github.com/meta-llama/PurpleLlama) &#124; [Website](https://meta-llama.github.io/PurpleLlama/CyberSecEval/) | Meta's latest suite; includes AutoPatchBench and CyberSOCEval |
| AutoPatchBench | 2025 | Dataset | [Blog](https://engineering.fb.com/2025/04/29/ai-research/autopatchbench-benchmark-ai-powered-security-fixes/) | Benchmark for automated repair of fuzzing-discovered vulnerabilities (part of CyberSecEval 4) |
| CyberSOCEval | 2025 | Dataset | [arXiv:2509.20166](https://arxiv.org/abs/2509.20166) &#124; [Meta](https://ai.meta.com/research/publications/cybersoceval-benchmarking-llms-capabilities-for-malware-analysis-and-threat-intelligence-reasoning/) | SOC-focused benchmarks for malware analysis + threat-intelligence reasoning (part of CyberSecEval 4) |
| SecLLMHolmes | 2024 | Framework | [arXiv:2405.19803](https://arxiv.org/abs/2405.19803) &#124; [GitHub](https://github.com/ai4cloudops/SecLLMHolmes) | Fully automated framework for evaluating LLM vulnerability detection |
| SecVulEval | 2025 | Dataset | [arXiv:2505.19828](https://arxiv.org/abs/2505.19828) | C/C++ vulnerability detection with statement-level granularity |
| VulnLLMEval | 2024 | Framework | [arXiv:2401.16185](https://arxiv.org/abs/2401.16185) | Framework for evaluating LLMs in vulnerability detection and patching |
| LLMSecCode | 2024 | Harness | [arXiv:2408.17894](https://arxiv.org/abs/2408.17894) | Benchmark for evaluating secure code practices in LLMs |
| eyeballvul | 2024 | Dataset | [arXiv:2407.08708](https://arxiv.org/abs/2407.08708) &#124; [GitHub](https://github.com/timothee-chauvin/eyeballvul) | Future-proof benchmark for vulnerability detection in the wild |
| VADER | 2025 | Framework | [arXiv:2505.19395](https://arxiv.org/abs/2505.19395) &#124; [GitHub](https://github.com/AfterQuery/vader) | Human-evaluated framework for Vulnerability Assessment, Detection, Explanation |
| SVEN | 2023 | Framework | [arXiv:2302.05319](https://arxiv.org/abs/2302.05319) &#124; [GitHub](https://github.com/eth-sri/sven) | Security hardening & adversarial testing for code LLMs (controlled code generation) |
| PrimeVul | 2024 | Dataset | [arXiv:2403.18624](https://arxiv.org/abs/2403.18624) &#124; [GitHub](https://github.com/DLVulDet/PrimeVul) | Filtered C/C++ function-level vulnerability detection dataset (ICSE 2025) |
| ARVO | 2024 | Dataset | [arXiv:2411.09278](https://arxiv.org/abs/2411.09278) | Project-level C/C++ vulnerability detection benchmark |
| Juliet Test Suite 1.3 | 2017 | Dataset | [NIST](https://samate.nist.gov/SARD/test-suites/111) | NIST's comprehensive test cases for C/C++ and Java vulnerability detection |
| SV-TrustEval-C | 2025 | Dataset | [arXiv:2505.20630](https://arxiv.org/abs/2505.20630) | Structure and semantic reasoning benchmark for C vulnerability analysis |
| VulGate | 2025 | Dataset | [arXiv:2508.16625](https://arxiv.org/abs/2508.16625) | Dataset with dedicated test sets for vulnerability detection generalization |
| Multi-Vuln Detection | 2025 | Dataset | [arXiv:2512.22306](https://arxiv.org/abs/2512.22306) | Long-context multi-label vulnerability benchmark across languages |
| Multi-Lang SVD | 2025 | Dataset | [arXiv:2503.01449](https://arxiv.org/abs/2503.01449) | Multi-language software vulnerability detection benchmark study |
| **SECURE CODE GENERATION** |||||
| CodeSecEval | 2024 | Harness | [arXiv:2407.02395](https://arxiv.org/abs/2407.02395) | Evaluating LLMs on secure code generation with CWE-based evaluation |
| CodeLMSec | 2024 | Harness | [arXiv:2302.04012](https://arxiv.org/abs/2302.04012) | Systematically evaluating security vulnerabilities in black-box code LMs |
| SecurityEval | 2022 | Dataset | [arXiv:2210.09263](https://arxiv.org/abs/2210.09263) &#124; [GitHub](https://github.com/s2e-lab/SecurityEval) | Mining vulnerability examples to evaluate ML-based code generation |
| SecCodePLT | 2024 | Harness | [arXiv:2410.11096](https://arxiv.org/abs/2410.11096) &#124; [GitHub](https://github.com/Virtue-Software/SecCodePLT) | Unified platform for evaluating code GenAI security with 27 CWEs |
| CWEval | 2025 | Harness | [arXiv:2501.08200](https://arxiv.org/abs/2501.08200) &#124; [GitHub](https://github.com/Co1lin/CWEval) | Outcome-driven evaluation for functionality and security of LLM code |
| SafeGenBench | 2025 | Dataset | [arXiv:2506.05692](https://arxiv.org/abs/2506.05692) | 558 security-sensitive test questions covering vulnerability taxonomy (ByteDance) |
| SecRepoBench | 2025 | Dataset | [arXiv:2504.21205](https://arxiv.org/abs/2504.21205) | Repository-level secure coding with 318 tasks across 27 repos |
| SecureAgentBench | 2025 | Environment | [arXiv:2509.22097](https://arxiv.org/abs/2509.22097) | 105 tasks for evaluating code agents on secure code generation |
| A.S.E | 2025 | Dataset | [arXiv:2508.18106](https://arxiv.org/abs/2508.18106) &#124; [GitHub](https://github.com/Tencent/AICGSecEval) | Repository-level benchmark using real CVEs (Tencent) |
| SEC-bench | 2025 | Environment | [arXiv:2506.11791](https://arxiv.org/abs/2506.11791) | Automated benchmarking of LLM agents on real-world security tasks |
| CASTLE | 2025 | Dataset | [arXiv:2503.09433](https://arxiv.org/abs/2503.09433) &#124; [GitHub](https://github.com/CASTLE-Benchmark/CASTLE-Benchmark) | Benchmarking dataset for static code analysis with ground-truth annotations |
| MT-Sec | 2025 | Dataset | [arXiv:2503.09700](https://arxiv.org/abs/2503.09700) | Benchmarking correctness and security in multi-turn code generation |
| DUALGUAGE | 2025 | Harness | [arXiv:2503.09380](https://arxiv.org/abs/2503.09380) | Automated joint security-functionality benchmarking |
| BaxBench | 2025 | Environment | [arXiv:2502.11844](https://arxiv.org/abs/2502.11844) &#124; [GitHub](https://github.com/logic-star-ai/baxbench) | 392 backend tasks validating functionality and exploits |
| SecCodeBench | 2025 | Harness | [GitHub](https://github.com/alibaba/sec-code-bench) | Alibaba's 37 test cases / 16 CWEs with dynamic PoC exploits |
| AutoPatchBench | 2025 | Component | [GitHub](https://github.com/meta-llama/PurpleLlama) | Automated vulnerability patching (component of CyberSecEval v4) |
| PatchEval | 2025 | Harness | [arXiv:2511.11019](https://arxiv.org/abs/2511.11019) | Multilingual benchmark for patching real-world vulnerabilities; 1,000 CVEs |
| PACEbench | 2025 | Framework | [arXiv:2510.11688](https://arxiv.org/abs/2510.11688) | Framework for practical AI cyber-exploitation capabilities |
| VulnRepairEval | 2025 | Framework | [arXiv:2509.03331](https://arxiv.org/abs/2509.03331) | Exploit-based evaluation for LLM vulnerability repair |
| **THREAT INTELLIGENCE & CTI** |||||
| CTIBench | 2024 | Dataset | [arXiv:2406.07599](https://arxiv.org/abs/2406.07599) &#124; [HuggingFace](https://huggingface.co/datasets/AI4Sec/cti-bench) | Evaluating LLMs on CTI tasks: attack patterns, threat actors, APT campaigns |
| CTI-MCQ | 2024 | Dataset | [arXiv:2406.07599](https://arxiv.org/abs/2406.07599) | Multiple-choice questions on attack patterns, APT campaigns, detection |
| CTI-RCM | 2024 | Dataset | [arXiv:2406.07599](https://arxiv.org/abs/2406.07599) | Root cause mapping correlating CVE records with CWE entities |
| SecKnowledge-Eval | 2025 | Dataset | [arXiv:2510.14113](https://arxiv.org/abs/2510.14113) | Evaluation datasets for complex cybersecurity tasks |
| ExCyTIn-Bench | 2025 | Environment | [arXiv:2507.14201](https://arxiv.org/abs/2507.14201) | Microsoft's cyber threat investigation agent benchmark |
| **MALWARE & SOC** |||||
| ACSE-Eval | 2025 | Dataset | [arXiv:2505.11565](https://arxiv.org/abs/2505.11565) | AWS Cloud Security Engineering Eval for cloud threat modeling |
| LogEval | 2024 | Dataset | [arXiv:2407.01896](https://arxiv.org/abs/2407.01896) | Benchmark suite for LLM log analysis tasks (parsing, anomaly detection, diagnosis, summarization) |
| CyberSOCEval | 2025 | Component | [GitHub](https://github.com/meta-llama/PurpleLlama) | Malware analysis and threat intel reasoning (component of CyberSecEval v4) |
| Sophos SOC Benchmark | 2024 | Harness | [Sophos Blog](https://news.sophos.com/en-us/2024/05/22/llms-and-the-future-of-soc-operations/) | Three SOC tasks: incident investigation, summarization, severity evaluation |
| LogLLM | 2024 | Harness | [arXiv:2411.08561](https://arxiv.org/abs/2411.08561) | Log-based anomaly detection using large language models |
| LogLM | 2024 | Dataset | [arXiv:2410.09352](https://arxiv.org/abs/2410.09352) | Task-based to instruction-based automated log analysis benchmark |
| DefenderBench | 2025 | Harness | [arXiv:2506.00739](https://arxiv.org/abs/2506.00739) &#124; [GitHub](https://github.com/microsoft/DefenderBench) | Microsoft toolkit spanning offense/defense/understanding |
| **PHISHING DETECTION** |||||
| PhishAgent Benchmark | 2024 | Harness | [arXiv:2408.02291](https://arxiv.org/abs/2408.02291) | Multimodal agent evaluation for phishing webpage detection |
| MLLM Phishing Benchmark | 2025 | Dataset | [arXiv:2503.01040](https://arxiv.org/abs/2503.01040) | Comprehensive security benchmark for multimodal LLMs |
| Open-Source LLM Phishing | 2025 | Harness | [arXiv:2503.01520](https://arxiv.org/abs/2503.01520) | 21 LLMs evaluated with 4 prompt engineering techniques |
| SME Phishing Detection | 2025 | Dataset | [arXiv:2503.08766](https://arxiv.org/abs/2503.08766) | LLM evaluation for small/medium enterprise phishing |
| PhishEmailLLM Dataset | 2025 | Dataset | [arXiv:2503.09270](https://arxiv.org/abs/2503.09270) | Meta-model approach for phishing email detection |
| APOLLO | 2024 | Harness | [arXiv:2402.16862](https://arxiv.org/abs/2402.16862) | GPT-based tool benchmark for phishing email detection |
| **PROMPT INJECTION** |||||
| AgentDojo | 2024 | Environment | [arXiv:2406.13352](https://arxiv.org/abs/2406.13352) &#124; [GitHub](https://github.com/ethz-spylab/agentdojo) | Dynamic environment for prompt injection attacks & defenses |
| INJECAGENT | 2024 | Dataset | [arXiv:2403.02691](https://arxiv.org/abs/2403.02691) &#124; [GitHub](https://github.com/uiuc-kang-lab/InjecAgent) | Benchmarking indirect prompt injections in tool-using agents |
| BIPIA | 2023 | Dataset | [arXiv:2312.14197](https://arxiv.org/abs/2312.14197) &#124; [GitHub](https://github.com/microsoft/BIPIA) | First benchmark for indirect prompt injection attacks |
| GenTel-Safe / GenTel-Bench | 2024 | Framework | [arXiv:2404.06531](https://arxiv.org/abs/2404.06531) | Unified benchmark and shielding framework for prompt injection |
| NotInject | 2024 | Dataset | [arXiv:2410.22770](https://arxiv.org/abs/2410.22770) &#124; [GitHub](https://github.com/SaFoLab-WISC/InjecGuard) | Over-defense/false-positive benchmark for prompt injection detectors (from InjecGuard) |
| PINT Benchmark | 2024 | Harness | [GitHub](https://github.com/lakeraai/pint-benchmark) | Lakera's benchmark for prompt injection detection systems |
| Tensor Trust | 2023 | Dataset | [arXiv:2311.01011](https://arxiv.org/abs/2311.01011) &#124; [GitHub](https://github.com/HumanCompatibleAI/tensor-trust) &#124; [Website](https://tensortrust.ai/) | 126K+ human-generated prompt injection attacks |
| Open-Prompt-Injection | 2024 | Harness | [arXiv:2402.12138](https://arxiv.org/abs/2402.12138) | Standardized framework for prompt injection vulnerability evaluation |
| safe-guard-prompt-injection | 2024 | Dataset | [arXiv:2402.13064](https://arxiv.org/abs/2402.13064) &#124; [HuggingFace](https://huggingface.co/datasets/Hack90/safe-guard-prompt-injection) | 10,296 prompt injection examples for guardrail evaluation |
| WASP | 2025 | Environment | [arXiv:2504.18575](https://arxiv.org/abs/2504.18575) &#124; [GitHub](https://github.com/facebookresearch/wasp) | Web Agent Security against Prompt injection attacks (Meta) |
| WAInjectBench | 2025 | Harness | [arXiv:2510.01354](https://arxiv.org/abs/2510.01354) | Benchmarking prompt injection detections for web agents |
| RAS-Eval | 2025 | Harness | [arXiv:2506.15253](https://arxiv.org/abs/2506.15253) | Security eval benchmark for LLM agents with tool execution |
| **JAILBREAK & RED TEAMING** |||||
| JailbreakBench | 2024 | Harness | [arXiv:2404.01318](https://arxiv.org/abs/2404.01318) &#124; [GitHub](https://github.com/JailbreakBench/jailbreakbench) &#124; [Website](https://jailbreakbench.github.io/) | 200 behaviors for standardized jailbreak evaluation |
| HarmBench | 2024 | Harness | [arXiv:2402.04249](https://arxiv.org/abs/2402.04249) &#124; [GitHub](https://github.com/centerforaisafety/HarmBench) | Standardized evaluation framework for automated red teaming |
| SafetyBench | 2024 | Harness | [GitHub](https://github.com/centerforaisafety/SafetyBench) | CAIS benchmark suite for comprehensive safety testing |
| ToxiGen | 2022 | Dataset | [arXiv:2203.09509](https://arxiv.org/abs/2203.09509) &#124; [GitHub](https://github.com/microsoft/ToxiGen) | Microsoft dataset for toxic/harmful content detection |
| StrongREJECT | 2024 | Dataset | [arXiv:2402.10260](https://arxiv.org/abs/2402.10260) &#124; [GitHub](https://github.com/alexandrasouly/strongreject) | 313 forbidden questions across 6 categories |
| WildJailbreak | 2024 | Dataset | [arXiv:2406.18510](https://arxiv.org/abs/2406.18510) &#124; [GitHub](https://github.com/allenai/wildjailbreak) | 261,534 conversations for training LLMs to be safe |
| SORRY-Bench | 2024 | Harness | [arXiv:2406.14598](https://arxiv.org/abs/2406.14598) &#124; [GitHub](https://github.com/SORRY-Bench/SORRY-Bench) | Systematically evaluating LLM safety refusal behaviors |
| ALERT | 2024 | Harness | [arXiv:2404.08676](https://arxiv.org/abs/2404.08676) | Comprehensive benchmark for LLM safety through red teaming |
| GPTFuzzer | 2023 | Harness | [arXiv:2309.10253](https://arxiv.org/abs/2309.10253) &#124; [GitHub](https://github.com/sherdencooper/GPTFuzz) | Auto-generated jailbreak prompts for red teaming LLMs |
| SG-Bench | 2024 | Harness | [arXiv:2410.21965](https://arxiv.org/abs/2410.21965) | Evaluating LLM safety generalization across diverse tasks |
| LatentJailbreak | 2023 | Dataset | [arXiv:2307.08487](https://arxiv.org/abs/2307.08487) | Benchmark for evaluating text safety and output robustness |
| AdvBench | 2023 | Dataset | [arXiv:2307.15043](https://arxiv.org/abs/2307.15043) | Adversarial behaviors dataset (foundation for many jailbreak benchmarks) |
| S-Eval | 2024 | Harness | [arXiv:2405.14191](https://arxiv.org/abs/2405.14191) | Automatic and adaptive test generation for LLM safety |
| SAFE | 2024 | Dataset | [arXiv:2404.18539](https://arxiv.org/abs/2404.18539) | Fine-grained safety dataset for LLMs beyond binary classification |
| DoNotAnswer | 2023 | Dataset | [arXiv:2308.13387](https://arxiv.org/abs/2308.13387) &#124; [GitHub](https://github.com/Libr-AI/do-not-answer) | Evaluating safeguards in LLMs with comprehensive refusal scenarios |
| AILuminate Jailbreak v0.5 | 2025 | Harness | [MLCommons](https://mlcommons.org/ailuminate/) | MLCommons jailbreak benchmark (industry standard) |
| SecReEvalBench | 2025 | Dataset | [arXiv:2505.07584](https://arxiv.org/abs/2505.07584) | Multi-turn security resilience evaluation benchmark |
| HELM Safety | 2024 | Component | [Stanford HELM](https://crfm.stanford.edu/helm/) | Safety evaluation component of Stanford's HELM |
| Agent Security Bench (ASB) | 2024 | Harness | [arXiv:2410.02644](https://arxiv.org/abs/2410.02644) &#124; [GitHub](https://github.com/agiresearch/ASB) | Formalizing attacks and defenses in LLM-based agents (ICLR 2025) |
| AgentHarm | 2024 | Dataset | [arXiv:2410.09024](https://arxiv.org/abs/2410.09024) &#124; [HuggingFace](https://huggingface.co/datasets/ai-safety-institute/AgentHarm) | UK AISI's 110 harmful agent tasks |
| b3 (Backbone Breaker Benchmark) | 2025 | Dataset | [arXiv:2510.22620](https://arxiv.org/abs/2510.22620) &#124; [Blog](https://www.lakera.ai/blog/the-backbone-breaker-benchmark) | Security evaluation for backbone LLMs used inside AI agents (agent-focused adversarial attacks) |
| **META-BENCHMARKS & FRAMEWORKS** |||||
| CAIBench | 2025 | Meta-benchmark | [arXiv:2510.24317](https://arxiv.org/abs/2510.24317) | Meta-benchmark integrating Base, Cybench, AutoPenBench, CTIBench |
| CyberPII-Bench | 2025 | Dataset | [arXiv:2510.24317](https://arxiv.org/abs/2510.24317) | PII anonymization evaluation in cybersecurity contexts |
| Cyber-Zero | 2025 | Framework | [arXiv:2508.00910](https://arxiv.org/abs/2508.00910) &#124; [GitHub](https://github.com/Cyber-Zero/Cyber-Zero) | Training framework with benchmark suites for EnIGMA+ |
| EnIGMA | 2024 | Environment | [arXiv:2409.16165](https://arxiv.org/abs/2409.16165) &#124; [GitHub](https://github.com/TorRient/EnIGMA) | Enhanced interactive generative model agent for CTF |
| **NETWORK & IDS DATASETS** |||||
| NSL-KDD | 2009 | Dataset | [UNB](https://www.unb.ca/cic/datasets/nsl.html) | Classic network intrusion detection dataset |
| CIC-IDS 2017 | 2017 | Dataset | [UNB](https://www.unb.ca/cic/datasets/ids-2017.html) | Intrusion detection evaluation dataset |
| CSE-CIC-IDS 2018 | 2018 | Dataset | [UNB](https://www.unb.ca/cic/datasets/ids-2018.html) | IDS/IPS dataset on AWS infrastructure |
| CIC-DDoS 2019 | 2019 | Dataset | [UNB](https://www.unb.ca/cic/datasets/ddos-2019.html) | DDoS attack evaluation dataset |
| CIRA-CIC-DoHBrw 2020 | 2020 | Dataset | [UNB](https://www.unb.ca/cic/datasets/dohbrw-2020.html) | DNS over HTTPS dataset |
| CIC-Bell DNS 2021 | 2021 | Dataset | [UNB](https://www.unb.ca/cic/datasets/) | DNS security dataset |
| CIC IoT Attack Dataset | 2023 | Dataset | [UNB](https://www.unb.ca/cic/datasets/) | IoT attack patterns dataset |
| CICIoV2024 | 2024 | Dataset | [UNB](https://www.unb.ca/cic/datasets/) | Internet of Vehicles security dataset |
| CICIoMT2024 | 2024 | Dataset | [UNB](https://www.unb.ca/cic/datasets/) | Medical IoT security dataset |
| IoT-DIAD | 2024 | Dataset | [UNB](https://www.unb.ca/cic/datasets/iot-diad-2024.html) | IoT intrusion detection and anomaly dataset |
| APT IIoT (CICADA) | 2024 | Dataset | [UNB](https://www.unb.ca/cic/datasets/) | Industrial IoT APT attack dataset |
| Datasense IIoT | 2025 | Dataset | [UNB](https://www.unb.ca/cic/datasets/) | Industrial IoT sensor data security dataset |
| BCCC-CIC-IDS2017 | 2017 | Dataset | [UNB BCCC](https://www.unb.ca/cic/datasets/) | Large-scale intrusion detection dataset |
| BCCC-CSE-CIC-IDS2018 | 2018 | Dataset | [UNB BCCC](https://www.unb.ca/cic/datasets/) | Extended large-scale IDS dataset |
| BCCC-VulSCs-2023 | 2023 | Dataset | [UNB BCCC](https://www.unb.ca/cic/datasets/) | Vulnerable smart contracts dataset |
| BCCC-SCsVuls-2024 | 2024 | Dataset | [UNB BCCC](https://www.unb.ca/cic/datasets/) | Smart contract vulnerabilities dataset |
| BCCC-cPacket-Cloud-DDoS-2024 | 2024 | Dataset | [UNB BCCC](https://www.unb.ca/cic/datasets/) | Cloud DDoS attacks dataset |
| BCCC-IoT-IDS-Zwave-2025 | 2025 | Dataset | [UNB BCCC](https://www.unb.ca/cic/datasets/) | Large-scale IoT-Zwave intrusion detection dataset |
| BCCC-DeFiFraudTrans-2025 | 2025 | Dataset | [UNB BCCC](https://www.unb.ca/cic/datasets/) | DeFi fraud transactions dataset |
| BCCC-Mal-NetMem-2025 | 2025 | Dataset | [UNB BCCC](https://www.unb.ca/cic/datasets/) | Multisource malware analysis using network traffic and memory |
| UNSW-NB15 | 2015/2024 | Dataset | [UNSW](https://research.unsw.edu.au/projects/unsw-nb15-dataset) | Network intrusion detection dataset (updated 2024) |
| MITRE ATLAS | 2023 | Knowledge Base | [MITRE](https://atlas.mitre.org/) | Attack success rates for AI models (knowledge base) |
| **AI SECURITY ADJACENT: ADVERSARIAL ML & ROBUSTNESS** |||||
| CIFAR-10 Adversarial | 2019 | Dataset | [IBM ART](https://github.com/Trusted-AI/adversarial-robustness-toolbox) | IBM ART benchmark for attack success rates |
| ADDC | 2024 | Dataset | ⚠️ *Unable to verify - Mahalle et al.* | Adversarial Diversity-Driven Dataset |
| DCA | 2024 | Harness | ⚠️ *Unable to verify - Mahalle et al.* | Aggressive events simulation for benchmark datasets |
| REA | 2024 | Harness | ⚠️ *Unable to verify - Mahalle et al.* | AI model threat handling evaluation |
| VMA | 2024 | Framework | ⚠️ *Unable to verify - Mahalle et al.* | AI model vulnerability identification benchmark |
| RobustBench | 2021 | Harness | [arXiv:2010.09670](https://arxiv.org/abs/2010.09670) &#124; [GitHub](https://github.com/RobustBench/robustbench) &#124; [Website](https://robustbench.github.io/) | Adversarial robustness leaderboard for image classifiers |
| AdversarialNLI (ANLI) | 2020 | Dataset | [arXiv:1910.14599](https://arxiv.org/abs/1910.14599) &#124; [GitHub](https://github.com/facebookresearch/anli) | Meta's adversarial robustness in NLI |
| **AI SECURITY ADJACENT: DEEPFAKE & MEDIA** |||||
| Faceswap-GAN | 2018 | Dataset | [GitHub](https://github.com/shaoanlu/faceswap-GAN) | Deepfake detection methods evaluation |
| VidTIMIT | 2012 | Dataset | [ConvAI](https://conradsanderson.id.au/vidtimit/) | Video-based deepfake detection benchmark |
| FaceForensics++ | 2019 | Dataset | [arXiv:1901.08971](https://arxiv.org/abs/1901.08971) &#124; [GitHub](https://github.com/ondyari/FaceForensics) | Large-scale deepfake detection benchmark |
| **MISCELLANEOUS** |||||
| AI Cyber Risk Benchmark | 2024 | Harness | [arXiv:2412.09878](https://arxiv.org/abs/2412.09878) | Evaluating automated exploitation capabilities |
| ITBench | 2025 | Environment | [arXiv:2502.03969](https://arxiv.org/abs/2502.03969) | Evaluating AI agents across IT automation tasks |
| SandboxEval | 2025 | Environment | [arXiv:2504.00018](https://arxiv.org/abs/2504.00018) | Test suite for LLM assessment environment safety |
| CyberLLMInstruct | 2025 | Dataset | [arXiv:2503.09144](https://arxiv.org/abs/2503.09144) | Dataset for analyzing safety of fine-tuned LLMs |
| DebugBench | 2024 | Harness | [arXiv:2401.08420](https://arxiv.org/abs/2401.08420) | Evaluating debugging capability of LLMs |
| AttackER | 2024 | Dataset | [arXiv:2408.05866](https://arxiv.org/abs/2408.05866) | Named entity recognition dataset for cyber-attack attribution |
| Hackphyr | 2024 | Environment | [arXiv:2409.11276](https://arxiv.org/abs/2409.11276) | Local fine-tuned LLM agent for network security |
| CyberPal.AI | 2024 | Dataset | [arXiv:2408.09304](https://arxiv.org/abs/2408.09304) | Expert-driven cybersecurity instructions benchmark |
| Truth Seeker Dataset | 2023 | Dataset | [UNB](https://www.unb.ca/cic/datasets/truthseeker-2023.html) | Misinformation detection in cybersecurity context |
| BinMetric | 2025 | Harness | [arXiv:2505.07360](https://arxiv.org/abs/2505.07360) | Binary analysis metrics and evaluation benchmark |
| OS-Harm | 2025 | Environment | [arXiv:2506.14866](https://arxiv.org/abs/2506.14866) | Safety of computer use agents with GUIs (150 tasks) |

---

## Summary Statistics

### By Category
| Category | Count |
|----------|-------|
| Knowledge & Q&A | 15 |
| CTF & Offensive | 10 |
| Penetration Testing | 13 |
| Vulnerability Detection | 19 |
| Secure Code Generation | 20 |
| Threat Intelligence & CTI | 5 |
| Malware & SOC | 7 |
| Phishing Detection | 6 |
| Prompt Injection | 12 |
| Jailbreak & Red Teaming | 20 |
| Meta-Benchmarks | 4 |
| Network & IDS Datasets | 22 |
| Miscellaneous | 12 |
| **Core Cybersecurity Benchmarks** | **~165** |
| *AI Security Adjacent: Adversarial ML* | *7* |
| *AI Security Adjacent: Deepfake & Media* | *3* |
| **TOTAL (incl. adjacent)** | **~175** |

---

