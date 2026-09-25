using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Jevlet.Core;

/// <summary>A typed question: the options are runtime-defined; the answer can only name one.</summary>
public sealed record Question(string Text, IReadOnlyList<string> Options, string Kind = "choice");

/// <summary>Calibrated probabilities over a question's options.</summary>
public sealed record Answer(IReadOnlyList<string> Options, IReadOnlyList<double> Probabilities)
{
    public int Best { get; } = Probabilities.Select((p, i) => (p, i)).MaxBy(x => x.p).i;
    public string Selected => Options[Best];
    public double Confidence => Probabilities[Best];
    public double this[string option] => Probabilities[Options.ToList().IndexOf(option)];
}

/// <summary>model.json written by jevlet.onnx_export: how to pack rows for this model.</summary>
public sealed record ModelManifest(
    string Format,
    string Backbone,
    bool Lowercase,
    IReadOnlyDictionary<string, int> TokenIds,
    int MaxPosition,
    int MaxPackedLen,
    int MaxStateTokens,
    int MaxQuestionTokens,
    int MaxOptionTokens,
    string AttentionTopology,
    string OptionPool,
    double Temperature,
    IReadOnlyDictionary<string, double> Temperatures,
    string SourceCheckpoint)
{
    public static ModelManifest Load(string path)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        return Parse(document.RootElement);
    }

    public static ModelManifest Parse(JsonElement root)
    {
        var format = root.GetProperty("format").GetString()!;
        if (format != "jevlet-onnx-1")
        {
            throw new InvalidDataException($"unsupported model format {format}");
        }
        return new ModelManifest(
            format,
            root.GetProperty("backbone").GetString()!,
            root.GetProperty("lowercase").GetBoolean(),
            root.GetProperty("token_ids").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetInt32()),
            root.GetProperty("max_position").GetInt32(),
            root.GetProperty("max_packed_len").GetInt32(),
            root.GetProperty("max_state_tokens").GetInt32(),
            root.GetProperty("max_question_tokens").GetInt32(),
            root.GetProperty("max_option_tokens").GetInt32(),
            root.GetProperty("attention_topology").GetString()!,
            root.GetProperty("option_pool").GetString()!,
            root.GetProperty("temperature").GetDouble(),
            root.GetProperty("temperatures").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetDouble()),
            root.GetProperty("source_checkpoint").GetString()!);
    }

    public double TemperatureFor(string kind) =>
        Temperatures.TryGetValue(kind, out var value) ? value
        : Temperatures.TryGetValue("default", out var fallback) ? fallback
        : Temperature;
}

/// <summary>One question branch inside a packed row.</summary>
public sealed record PackedQuestion(IReadOnlyList<(int Start, int Stop)> OptionSpans, IReadOnlyList<int> OptionEnds, int DecidePosition);

/// <summary>A state and its question branches as one sequence (PretrainedCollator for one row).</summary>
public sealed record PackedRow(long[] InputIds, long[] PositionIds, int[] BranchIds, IReadOnlyList<PackedQuestion> Questions);

/// <summary>
/// Port of <c>PretrainedCollator._pack_row</c> (block_bidir topology): the state once, then each
/// question as [QUESTION] text, [OPTION] body [END_OPTION] per option, and [DECIDE]. Branch
/// positions restart after the state; the state is trimmed only as far as the longest branch needs.
/// </summary>
public sealed class Packer(WordPieceTokenizer tokenizer, ModelManifest manifest)
{
    private const int StateBranch = -1;

    public PackedRow Pack(string state, IReadOnlyList<Question> questions)
    {
        if (manifest.AttentionTopology != "block_bidir")
        {
            throw new NotSupportedException($"topology {manifest.AttentionTopology}");
        }
        var ids = manifest.TokenIds;
        var questionBodies = questions.Select(q => Cap(tokenizer.Encode(q.Text), manifest.MaxQuestionTokens)).ToList();
        var optionBodies = questions
            .Select(q => q.Options.Select(o => OptionBody(tokenizer.Encode(o), ids["unk"])).ToList())
            .ToList();
        var longest = 0;
        for (var index = 0; index < questions.Count; index++)
        {
            var extent = 1 + questionBodies[index].Count + optionBodies[index].Sum(b => b.Count + 2) + 1;
            longest = Math.Max(longest, extent);
        }
        var room = manifest.MaxPosition - 1 - 1 - longest;
        var stateBody = Cap(tokenizer.Encode(state), Math.Max(0, Math.Min(manifest.MaxStateTokens, room)));

        var tokens = new List<long> { ids["cls"], ids["state"] };
        tokens.AddRange(stateBody.Select(id => (long)id));
        var stateLength = tokens.Count;
        var branches = Enumerable.Repeat(StateBranch, stateLength).ToList();
        var positions = Enumerable.Range(0, stateLength).Select(p => (long)p).ToList();
        var packed = new List<PackedQuestion>();
        for (var branch = 0; branch < questions.Count; branch++)
        {
            var offset = tokens.Count;
            var branchTokens = new List<long> { ids["question"] };
            branchTokens.AddRange(questionBodies[branch].Select(id => (long)id));
            var spans = new List<(int, int)>();
            var ends = new List<int>();
            foreach (var body in optionBodies[branch])
            {
                var start = offset + branchTokens.Count + 1;
                spans.Add((start, start + body.Count));
                branchTokens.Add(ids["option"]);
                branchTokens.AddRange(body.Select(id => (long)id));
                branchTokens.Add(ids["end_option"]);
                ends.Add(offset + branchTokens.Count - 1);
            }
            branchTokens.Add(ids["decide"]);
            var lastPosition = stateLength + branchTokens.Count - 1;
            if (lastPosition >= manifest.MaxPosition)
            {
                throw new InvalidOperationException(
                    $"question {branch} needs position {lastPosition}, over the backbone limit {manifest.MaxPosition}");
            }
            tokens.AddRange(branchTokens);
            branches.AddRange(Enumerable.Repeat(branch, branchTokens.Count));
            positions.AddRange(Enumerable.Range(stateLength, branchTokens.Count).Select(p => (long)p));
            packed.Add(new PackedQuestion(spans, ends, tokens.Count - 1));
        }
        if (tokens.Count > manifest.MaxPackedLen)
        {
            throw new InvalidOperationException($"packed row has {tokens.Count} tokens, over {manifest.MaxPackedLen}");
        }
        return new PackedRow(tokens.ToArray(), positions.ToArray(), branches.ToArray(), packed);
    }

    private static List<int> Cap(List<int> ids, int limit) => ids.Count > limit ? ids.GetRange(0, limit) : ids;

    private List<int> OptionBody(List<int> ids, int unknown)
    {
        var body = Cap(ids, manifest.MaxOptionTokens);
        return body.Count > 0 ? body : [unknown];
    }
}

/// <summary>The model: one packed ONNX pass answers every question about one state.</summary>
public sealed class DecisionEngine : IDisposable
{
    private readonly InferenceSession _session;
    public ModelManifest Manifest { get; }
    public Packer Packer { get; }

    private DecisionEngine(InferenceSession session, ModelManifest manifest, Packer packer)
    {
        _session = session;
        Manifest = manifest;
        Packer = packer;
    }

    /// <summary>"gpu" when DirectML runs the model, "cpu" otherwise.</summary>
    public string Device { get; private set; } = "cpu";

    /// <summary>
    /// Load model.onnx, model.json, and vocab.txt from one directory. The GPU (DirectML) is
    /// tried first; any failure there falls back to the CPU, which always works.
    /// </summary>
    public static DecisionEngine Load(string directory, bool preferGpu = true, int? threads = null)
    {
        var manifest = ModelManifest.Load(Path.Combine(directory, "model.json"));
        var tokenizer = WordPieceTokenizer.FromFile(Path.Combine(directory, "vocab.txt"), manifest.Lowercase);
        var model = Path.Combine(directory, "model.onnx");
        var packer = new Packer(tokenizer, manifest);
        if (preferGpu)
        {
            try
            {
                using var gpu = new SessionOptions
                {
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                    EnableMemoryPattern = false, // required by DirectML
                    ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                };
                gpu.AppendExecutionProvider_DML(0);
                return new DecisionEngine(new InferenceSession(model, gpu), manifest, packer) { Device = "gpu" };
            }
            catch (OnnxRuntimeException)
            {
                // No DirectML device (VM, remote session, old driver): use the CPU.
            }
        }
        using var cpu = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            // Physical cores: hyperthreads slowed this graph down in measurements (8 vs 16 threads).
            IntraOpNumThreads = threads ?? Math.Max(1, Environment.ProcessorCount / 2),
        };
        return new DecisionEngine(new InferenceSession(model, cpu), manifest, packer);
    }

    /// <summary>Raw logits per question (before temperature), in question order.</summary>
    public IReadOnlyList<float[]> Logits(PackedRow row)
    {
        var length = row.InputIds.Length;
        var mask = new bool[length * length];
        for (var query = 0; query < length; query++)
        {
            var queryState = row.BranchIds[query] < 0;
            for (var key = 0; key < length; key++)
            {
                var keyState = row.BranchIds[key] < 0;
                mask[query * length + key] = queryState ? keyState : keyState || row.BranchIds[key] == row.BranchIds[query];
            }
        }
        var optionCount = row.Questions.Sum(q => q.OptionSpans.Count);
        var pool = new float[optionCount * length];
        var decide = new float[optionCount * length];
        var option = 0;
        foreach (var question in row.Questions)
        {
            for (var index = 0; index < question.OptionSpans.Count; index++, option++)
            {
                if (Manifest.OptionPool == "end")
                {
                    pool[option * length + question.OptionEnds[index]] = 1f;
                }
                else
                {
                    var (start, stop) = question.OptionSpans[index];
                    for (var token = start; token < stop; token++)
                    {
                        pool[option * length + token] = 1f / (stop - start);
                    }
                }
                decide[option * length + question.DecidePosition] = 1f;
            }
        }
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(row.InputIds, [1, length])),
            NamedOnnxValue.CreateFromTensor("position_ids", new DenseTensor<long>(row.PositionIds, [1, length])),
            NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<bool>(mask, [1, 1, length, length])),
            NamedOnnxValue.CreateFromTensor("option_pool", new DenseTensor<float>(pool, [optionCount, length])),
            NamedOnnxValue.CreateFromTensor("decide_select", new DenseTensor<float>(decide, [optionCount, length])),
        };
        using var results = _session.Run(inputs);
        var flat = results[0].AsEnumerable<float>().ToArray();
        var perQuestion = new List<float[]>();
        var cursor = 0;
        foreach (var question in row.Questions)
        {
            perQuestion.Add(flat[cursor..(cursor + question.OptionSpans.Count)]);
            cursor += question.OptionSpans.Count;
        }
        return perQuestion;
    }

    /// <summary>Answer every question about one state in a single pass.</summary>
    public IReadOnlyList<Answer> Evaluate(string state, IReadOnlyList<Question> questions)
    {
        var logits = Logits(Packer.Pack(state, questions));
        var answers = new List<Answer>(questions.Count);
        for (var index = 0; index < questions.Count; index++)
        {
            var temperature = Manifest.TemperatureFor(questions[index].Kind);
            answers.Add(new Answer(questions[index].Options, Softmax(logits[index], temperature)));
        }
        return answers;
    }

    private static double[] Softmax(float[] logits, double temperature)
    {
        var scaled = logits.Select(l => l / temperature).ToArray();
        var max = scaled.Max();
        var exps = scaled.Select(s => Math.Exp(s - max)).ToArray();
        var sum = exps.Sum();
        return exps.Select(e => e / sum).ToArray();
    }

    public void Dispose() => _session.Dispose();
}
