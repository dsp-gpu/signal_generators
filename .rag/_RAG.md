---
schema_version: 1
repo: signal_generators
version: 0.1.0
layer: compute
maturity: alpha
purpose: "TODO: AI-fill — назначение репо signal_generators"

modules:
  public:                               # auto: include/<repo>/*
    - generators
    - kernels
    - params
  internal:                             # auto: src/* кроме include
    - signal_generators

key_classes:                            # auto: top по test_params
  - fqn: signal_gen::SignalGeneratorFactory
    brief: "Создаёт генераторы по типу сигнала"
    maturity: alpha
    methods: 7
    test_params_rows: 10
    test_params: test_params/signal_gen_SignalGeneratorFactory.md
  - fqn: signal_gen::CwGeneratorROCm
    brief: "@ingroup grp_signal_generators"
    maturity: alpha
    methods: 11
    test_params_rows: 7
    test_params: test_params/signal_gen_CwGeneratorROCm.md
  - fqn: signal_gen::LfmGeneratorROCm
    brief: "TODO: AI-fill"
    maturity: alpha
    methods: 11
    test_params_rows: 7
    test_params: test_params/signal_gen_LfmGeneratorROCm.md
  - fqn: signal_gen::NoiseGeneratorROCm
    brief: "TODO: AI-fill"
    maturity: alpha
    methods: 11
    test_params_rows: 7
    test_params: test_params/signal_gen_NoiseGeneratorROCm.md
  - fqn: signal_gen::ScriptGeneratorROCm
    brief: "TODO: AI-fill"
    maturity: alpha
    methods: 24
    test_params_rows: 2
    test_params: test_params/signal_gen_ScriptGeneratorROCm.md
  - fqn: signal_gen::FormSignalGeneratorROCm
    brief: "@ingroup grp_signal_generators"
    maturity: alpha
    methods: 25
    test_params_rows: 1
    test_params: test_params/signal_gen_FormSignalGeneratorROCm.md
  - fqn: signal_gen::LfmGeneratorAnalyticalDelayROCm
    brief: "TODO: AI-fill"
    maturity: alpha
    methods: 18
    test_params_rows: 1
    test_params: test_params/signal_gen_LfmGeneratorAnalyticalDelayROCm.md
  - fqn: signal_gen::DelayedFormSignalGeneratorROCm
    brief: "TODO: AI-fill"
    maturity: alpha
    methods: 16
    test_params_rows: 1
    test_params: test_params/signal_gen_DelayedFormSignalGeneratorROCm.md
  - fqn: signal_gen::FormParams
    brief: "TODO: AI-fill"
    maturity: alpha
    methods: 4
    test_params_rows: 1
    test_params: test_params/signal_gen_FormParams.md
  - fqn: signal_gen::LfmParams
    brief: "TODO: AI-fill"
    maturity: alpha
    methods: 3
    test_params_rows: 1
    test_params: test_params/signal_gen_LfmParams.md
  - fqn: signal_gen::FormScriptGeneratorROCm
    brief: "TODO: AI-fill"
    maturity: alpha
    methods: 14
    test_params_rows: 0
    test_params: test_params/signal_gen_FormScriptGeneratorROCm.md
  - fqn: signal_gen::LfmConjugateGeneratorROCm
    brief: "TODO: AI-fill"
    maturity: alpha
    methods: 13
    test_params_rows: 0
    test_params: test_params/signal_gen_LfmConjugateGeneratorROCm.md
  - fqn: PyLfmAnalyticalDelayROCm
    brief: "TODO: AI-fill"
    maturity: alpha
    methods: 11
    test_params_rows: 0
    test_params: test_params/PyLfmAnalyticalDelayROCm.md
  - fqn: PyDelayedFormSignalGeneratorROCm
    brief: "TODO: AI-fill"
    maturity: alpha
    methods: 10
    test_params_rows: 0
    test_params: test_params/PyDelayedFormSignalGeneratorROCm.md
  - fqn: signal_gen::FormSignalGenerator
    brief: "GPU-генератор комплексных сигналов по формуле getX"
    maturity: alpha
    methods: 9
    test_params_rows: 0
    test_params: test_params/signal_gen_FormSignalGenerator.md
  - fqn: signal_gen::FormScriptGenerator
    brief: "DSL-генератор + on-disk кэш кернелов для формулы getX"
    maturity: alpha
    methods: 8
    test_params_rows: 0
    test_params: test_params/signal_gen_FormScriptGenerator.md
  - fqn: signal_gen::ScriptGenerator
    brief: "Compiles text DSL -> OpenCL kernel and executes on GPU"
    maturity: alpha
    methods: 8
    test_params_rows: 0
    test_params: test_params/signal_gen_ScriptGenerator.md
  - fqn: PyFormSignalGeneratorROCm
    brief: "TODO: AI-fill"
    maturity: alpha
    methods: 7
    test_params_rows: 0
    test_params: test_params/PyFormSignalGeneratorROCm.md
  - fqn: PyLfmAnalyticalDelay
    brief: "TODO: AI-fill"
    maturity: alpha
    methods: 7
    test_params_rows: 0
    test_params: test_params/PyLfmAnalyticalDelay.md
  - fqn: signal_gen::DelayedFormSignalGenerator
    brief: "GPU-генератор с дробной задержкой Farrow (Lagrange 48×5)"
    maturity: alpha
    methods: 7
    test_params_rows: 0
    test_params: test_params/signal_gen_DelayedFormSignalGenerator.md

test_params_summary:
  classes_with_params: 11
  methods_with_params: 19
  ready_for_autotest:  1
  partial_coverage:    40
  no_status:           0
  total_rows:          41

repo_stats:
  total_symbols: 434
  public_classes: 46
  total_files: 72

depends_on:                              # TODO: ручная разметка после deps таблицы
  internal: []
  external: []

used_by: []                              # TODO: AI-fill из других _RAG.md

python_modules:                          # TODO: auto from pybind_bindings
  - TODO

architecture_files:                       # auto: arch_files generator
  - .rag/arch/C2_container.md
  - .rag/arch/C3_component.md
  - .rag/arch/C4_code.md
tags:                                    # auto-inferred (RAG_CLAUDE_C4)
  - "#layer:compute"
  - "#repo:signal_generators"
  - "#namespace:signal_gen"
  - "#pattern:Factory:SignalGeneratorFactory"

notes: []                                # TODO: AI-fill из ai_summary

ai_generated_at: 2026-05-09T05:27:58Z
ai_model: TODO (auto-fields only, AI-brief pending)
ai_sections: []
parser_version: 1
---

# signal_generators

## Назначение
*(TODO: AI-fill через ollama qwen3:8b)*

## Ключевые классы
*(автогенерируется из YAML key_classes выше)*

## Дополнительная документация
- [../Doc/](../Doc/)

<!-- ⚙️ Auto-generated by generate_rag_manifest.py — отредактируй и закоммить. -->
